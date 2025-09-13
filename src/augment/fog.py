# src/augment/fog.py
import cv2
import numpy as np

# -----------------------------
# 低频随机场（fBM：多八度叠加）
# -----------------------------
def rand_perlin(h, w, scale=128, octaves=2, persistence=0.5, lacunarity=2.0, seed=None):
    """
    生成 [0,1] 的低频随机场，近似 Perlin/Simplex，用于雾团不均匀性。
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    base = np.zeros((h, w), np.float32)
    freq = 1.0 / max(1, scale)
    amp = 1.0
    norm = 0.0
    for _ in range(max(1, octaves)):
        gh = max(1, int(h * freq))
        gw = max(1, int(w * freq))
        g = rng.rand(gh + 1, gw + 1).astype(np.float32)

        ys = np.linspace(0, gh, h, endpoint=False)
        xs = np.linspace(0, gw, w, endpoint=False)
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        y1 = np.clip(y0 + 1, 0, gh)
        x1 = np.clip(x0 + 1, 0, gw)
        wy = ys - y0
        wx = xs - x0

        g00 = g[y0][:, x0]
        g01 = g[y0][:, x1]
        g10 = g[y1][:, x0]
        g11 = g[y1][:, x1]
        top = g00 * (1 - wx) + g01 * wx
        bottom = g10 * (1 - wx) + g11 * wx
        val = top * (1 - wy[:, None]) + bottom * wy[:, None]

        base += amp * val
        norm += amp
        amp *= persistence
        freq *= lacunarity

    base /= max(1e-6, norm)
    base = (base - base.min()) / max(1e-6, (base.max() - base.min()))
    return base.astype(np.float32)

def _ensure_3c(x: np.ndarray) -> np.ndarray:
    """单通道扩成三通道。"""
    return x if x.ndim == 3 else np.stack([x, x, x], axis=-1)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _guided_filter(guide_gray_u8: np.ndarray, src_float: np.ndarray, radius=8, eps=1e-3) -> np.ndarray:
    """
    导向滤波（edge-preserving）保边平滑；若无 ximgproc 则退化为双边滤波近似。
    - guide_gray_u8: uint8 单通道（导向）
    - src_float: float32 单通道 [0,1]
    """
    try:
        gf = cv2.ximgproc.guidedFilter(guide_gray_u8, (src_float * 255).astype(np.uint8), radius, eps * 255 * 255)
        return gf.astype(np.float32) / 255.0
    except Exception:
        d = max(3, radius * 2 + 1)
        out = cv2.bilateralFilter(src_float.astype(np.float32), d, 12, 12)
        return out.astype(np.float32)

# -----------------------------
# 预设强度（或使用 MOR 控制）
# -----------------------------
FOG_PRESETS = {
    "light":  dict(beta=(0.03, 0.06),  airlight=(0.82, 0.93), glow=(0.12, 0.22), contrast_drop=(0.06, 0.12)),
    "medium": dict(beta=(0.06, 0.12),  airlight=(0.86, 0.96), glow=(0.18, 0.34), contrast_drop=(0.10, 0.18)),
    "heavy":  dict(beta=(0.12, 0.22),  airlight=(0.90, 0.99), glow=(0.28, 0.48), contrast_drop=(0.15, 0.26)),
}

def _rand_range(lo, hi, rng) -> float:
    return float(lo + (hi - lo) * rng.rand())

# -----------------------------
# 增强型道路雾合成器
# -----------------------------
class EnhancedFogSynthesizer:
    """
    基于大气散射模型的道路雾合成：
      I = J * t + A * (1 - t) ， t = exp(-β·d)
    关键：地平线软融合、远点约束、空气光自适应与平滑、边缘保留传输图、
          全局大气幕、深度递增模糊、柔和光晕、局部对比衰减。
    """
    def __init__(self,
                 level: str = "medium",
                 mor: float | None = None,         # 气象能见度（米），如 500/150/50；若给定优先生效
                 y_h_ratio: float = 0.42,          # 地平线相对高度（0~1，越小越靠上）
                 vanishing_x_ratio: float = 0.5,   # 远点水平位置（0~1）
                 perlin_scale_ratio: float = 0.18, # 雾团尺度占图像宽度比例（~0.12~0.25）
                 perlin_octaves: int = 2,
                 sky_boost: float = 1.25,          # 天空增强（>1）
                 road_damp: float = 0.9,           # 路面抑制（0.85~0.95）
                 edge_guided: bool = True,         # 传输图保边
                 horizon_softness: float = 0.06,   # 地平线软化系数（相对高度，0.04~0.10）
                 depth_blur_max: float = 3.5,      # 远处最大模糊半径（px，随分辨率调）
                 global_veil: float = 0.06,        # 全局大气幕强度（0~0.12）
                 seed: int | None = None):
        self.level = level
        self.mor = mor
        self.y_h_ratio = y_h_ratio
        self.vx_ratio = vanishing_x_ratio
        self.perlin_scale_ratio = perlin_scale_ratio
        self.perlin_octaves = perlin_octaves
        self.sky_boost = sky_boost
        self.road_damp = road_damp
        self.edge_guided = edge_guided
        self.horizon_softness = horizon_softness
        self.depth_blur_max = depth_blur_max
        self.global_veil = global_veil
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    # -------- 空气光（自适应 & 平滑） --------
    def _airlight_from_image(self, img_f32: np.ndarray) -> np.ndarray:
        h, w = img_f32.shape[:2]
        band_h = max(10, int(0.12 * h))
        top = img_f32[:band_h]
        lum = 0.299 * top[:, :, 2] + 0.587 * top[:, :, 1] + 0.114 * top[:, :, 0]
        thr = np.quantile(lum, 0.9)
        mask = lum >= thr
        A_rgb = (top.mean(axis=(0, 1)) if mask.sum() < 100 else top[mask].mean(axis=0)).astype(np.float32)
        # 轻微色偏
        tint = self.rng.uniform(-0.02, 0.02, size=3).astype(np.float32)
        A_rgb = np.clip(A_rgb + tint, 0.7, 1.0)
        # 垂直渐变 + 轻微横向渐变
        vgrad = np.linspace(1.0, 0.85, h, dtype=np.float32)[:, None]
        xgrad = np.linspace(0.95, 1.05, w, dtype=np.float32)[None, :]
        A_map = _ensure_3c(vgrad) * A_rgb[None, None, :] * _ensure_3c(xgrad)
        # 导向滤波进一步柔化“天-楼”边界
        guide = (img_f32.mean(axis=2) * 255).astype(np.uint8)
        for c in range(3):
            A_map[:, :, c] = _guided_filter(guide, A_map[:, :, c].astype(np.float32), radius=16, eps=1e-3)
        return np.clip(A_map, 0.7, 1.0)

    # -------- 深度先验（地平线软融合 + 远点约束） --------
    def _depth_proxy(self, h: int, w: int):
        y_h = int(self.y_h_ratio * h)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        # 透视：靠近地平线更远
        y_off = np.maximum(yy - y_h, 1.0)
        d_persp = 1.0 / y_off

        # 远点约束：向远点径向增远
        vx = float(self.vx_ratio * w)
        vy = float(y_h)
        r = np.sqrt((xx - vx) ** 2 + (yy - vy) ** 2) + 1.0
        d_vanish = 1.0 / r

        # 归一 + 混合
        d = 0.7 * (d_persp / d_persp.max()) + 0.3 * (d_vanish / d_vanish.max())
        d = (d - d.min()) / max(1e-6, (d.max() - d.min()))

        # 地平线软融合（S型），避免硬分界带
        softness = max(1e-3, self.horizon_softness) * h
        sky_weight = _sigmoid((y_h - yy) / softness).astype(np.float32)  # 上部~1，下部~0
        d *= (1.0 + (self.sky_boost - 1.0) * sky_weight) * (self.road_damp ** (1.0 - sky_weight))
        return np.clip(d, 0, 1), y_h, sky_weight

    # -------- β 空间扰动（雾团） --------
    def _beta_map(self, h: int, w: int, base_beta: float) -> np.ndarray:
        scale = max(16, int(self.perlin_scale_ratio * w))
        noise = rand_perlin(h, w, scale=scale, octaves=self.perlin_octaves, seed=self.rng.randint(1e9))
        return (base_beta * (0.85 + 0.35 * noise)).astype(np.float32)

    # -------- 传输图（可选保边） --------
    def _transmission(self, beta_map: np.ndarray, depth: np.ndarray, guide_gray_u8: np.ndarray) -> np.ndarray:
        t = np.exp(-beta_map * depth)
        t = np.clip(t, 0.05, 1.0)
        if self.edge_guided:
            t = _guided_filter(guide_gray_u8, t.astype(np.float32), radius=8, eps=1e-3)
            t = np.clip(t, 0.05, 1.0)
        return t

    # -------- 柔和光晕（软掩码） --------
    def _glow(self, img_f32: np.ndarray, strength: float) -> np.ndarray:
        gray = cv2.cvtColor((img_f32 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        thr = np.clip(gray.mean() + 0.6 * gray.std(), 0.65, 0.9)
        hard = (gray > thr).astype(np.float32)
        k = int(9 + 20 * strength) | 1
        soft = cv2.GaussianBlur(hard, (k, k), k * 0.35).astype(np.float32)
        soft = np.clip(soft, 0, 1)
        k2 = int(max(7, (img_f32.shape[0] + img_f32.shape[1]) * (0.003 + 0.01 * strength))) | 1
        blur = cv2.GaussianBlur(img_f32, (k2, k2), k2 * 0.25)
        return np.clip(img_f32 * (1 - soft[..., None]) + (img_f32 + strength * blur) * soft[..., None], 0, 1)

    # -------- 深度递增模糊（远处更糊，模拟散射） --------
    def _depth_blur(self, hazy_f32: np.ndarray, depth: np.ndarray, strength: float) -> np.ndarray:
        r = depth * self.depth_blur_max * (0.5 + strength)  # 半径字段
        r = np.clip(r, 0.0, self.depth_blur_max * 1.5)
        out = hazy_f32.copy()
        # 三档近似，避免逐像素变核开销
        bands = [0.33, 0.66, 1.0]
        prev = np.zeros_like(depth)
        for b in bands:
            mask = ((depth >= prev) & (depth < b)).astype(np.float32)
            if mask.sum() < 100:
                prev = np.full_like(depth, b)
                continue
            rad = int(max(1, np.mean(r[mask > 0]) * 1.5)) | 1
            if rad <= 1:
                prev = np.full_like(depth, b)
                continue
            blurred = cv2.GaussianBlur(hazy_f32, (rad, rad), rad * 0.5)
            m3 = _ensure_3c(cv2.GaussianBlur(mask, (rad | 1, rad | 1), rad * 0.5))
            out = out * (1 - m3) + blurred * m3
            prev = np.full_like(depth, b)
        return np.clip(out, 0, 1)

    # -------- 局部对比衰减（保边） --------
    def _local_contrast_fade(self, img_f32: np.ndarray, amount: float) -> np.ndarray:
        ycrcb = cv2.cvtColor((img_f32 * 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        d = int(5 + amount * 20) | 1
        y_smooth = cv2.bilateralFilter(y, d, 25 + amount * 50, 25 + amount * 50)
        y_mix = cv2.addWeighted(y, 1.0 - amount, y_smooth, amount, 0.0)
        out = cv2.cvtColor(cv2.merge([y_mix, cr, cb]), cv2.COLOR_YCrCb2BGR).astype(np.float32) / 255.0
        return out

    # -------- 主入口 --------
    def synthesize(self, bgr_uint8: np.ndarray, level: str | None = None):
        """
        输入：BGR uint8
        输出：(hazy_bgr_uint8, meta_dict)
        meta 包含 beta_map/A_map/depth/y_h/t 等可视化或训练用信息。
        """
        img = bgr_uint8.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        if level is not None:
            self.level = level

        # 强度选择：优先 MOR
        if self.mor is not None and self.mor > 0:
            base_beta = 3.912 / float(self.mor)      # Koschmieder 近似
            glow_rng  = (0.12, 0.45)
            cdrop_rng = (0.08, 0.22)
            a_rng     = (0.86, 0.98)
        else:
            p = FOG_PRESETS[self.level]
            base_beta = _rand_range(*p["beta"], self.rng)
            glow_rng  = p["glow"]
            cdrop_rng = p["contrast_drop"]
            a_rng     = p["airlight"]

        # 深度（含软融合权重） + β 噪声
        depth, y_h, sky_weight = self._depth_proxy(h, w)
        beta_map = self._beta_map(h, w, base_beta)

        # 空气光（自适应+平滑），再统一拉亮到预设均值范围
        A_map = self._airlight_from_image(img)
        scale = _rand_range(*a_rng, self.rng) / max(1e-6, A_map.mean())
        A_map = np.clip(A_map * scale, 0.75, 1.0)

        # 传输图（保边）
        guide_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        t = self._transmission(beta_map, depth, guide_gray)
        t3 = _ensure_3c(t)

        # 基础散射合成
        hazy = img * t3 + A_map * (1.0 - t3)

        # 全局大气幕（上重下轻）
        gv = self.global_veil * (0.6 + 0.4 * sky_weight)  # 0~global_veil
        hazy = np.clip(hazy * (1.0 - _ensure_3c(gv)) + A_map * _ensure_3c(gv), 0, 1)

        # 柔和光晕
        glow = _rand_range(*glow_rng, self.rng)
        hazy = self._glow(hazy, glow)

        # 深度递增模糊
        hazy = self._depth_blur(hazy, depth, strength=base_beta)

        # 局部对比衰减
        cdrop = _rand_range(*cdrop_rng, self.rng)
        hazy = self._local_contrast_fade(hazy, cdrop)

        # 轻微色温/伽马/噪声（更接近传感器）
        tint = (1.0 + self.rng.uniform(-0.015, 0.02, size=3)).astype(np.float32)
        hazy = np.clip(hazy * tint[None, None, :], 0, 1)
        if self.rng.rand() < 0.35:
            gamma = 1.0 + self.rng.uniform(-0.04, 0.05)
            hazy = np.clip(hazy ** gamma, 0, 1)
        if self.rng.rand() < 0.3:
            noise = self.rng.normal(0, 0.0035, size=hazy.shape).astype(np.float32)
            hazy = np.clip(hazy + noise, 0, 1)

        return (hazy * 255.0 + 0.5).astype(np.uint8), {
            "beta_map": beta_map,
            "A_map": A_map,
            "depth": depth,
            "y_h": y_h,
            "t": t,
        }
