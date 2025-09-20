import yaml
from pathlib import Path
from copy import deepcopy

_DEFAULTS = {
    "camera": {
        "source": 0,
        "width": 1280,
        "height": 720,
        "fps_request": 30,
        "backend": "auto",
    },
    "preview": {
        "show_fps": True,
        "compare": {
            "enable": True,
            "layout": "h",
            "label_raw": "RAW",
            "label_proc": "PROC",
            "divider_px": 4,
        },
        "record": {
            "enable": False,
            "path": "out_compare.mp4",
            "fps": 30,
        },
    },
    "preprocess": {
        "enabled": False,
        "chain": [],
        "auto_gate": {
            "enable_low_contrast_gate": False,
            "contrast_thresh": 20.0,
        },
    },
    "detect": {
        "enabled": False,
        "backend": "ultralytics",
        "model": "yolov8n.pt",
        "device": "auto",
        "conf_thres": 0.25,
        "iou_thres": 0.7,
        "max_det": 100,
        "classes_keep": [],
    },
    "tracking": {
        "enabled": False,
        "backend": "sort",
        "max_staleness": 1.0,
        "min_hits": 3,
        "iou_threshold": 0.3,
        "speed_window": 0.75,
    },
    "geometry": {
        "enabled": False,
        "projector": {
            "type": "homography",
            "image_points": [],
            "world_points": [],
            "origin": [0.0, 0.0],
            "max_distance": 1_000_000.0,
        },
    },
    "vis": {
        "draw": {
            "det": True,
            "thickness": 2,
            "font_scale": 0.6,
        },
    },
    "runtime": {
        "async": {
            "enabled": False,
            "capture": {
                "queue_size": 4,
                "drop_oldest": True,
            },
            "infer": {
                "workers": 1,
                "profile_interval": 60,
            },
            "output": {
                "queue_size": 2,
            },
            "hot_reload": {
                "enabled": False,
                "watch_interval": 2.0,
            },
        },
    },
}

def _merge(a: dict, b: dict):
    """merge b into a (recursive)"""
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def _project_root():
    # 以当前文件所在目录往上找 project 根
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "configs").exists():
            return p
    return Path.cwd()

def resolve_config_path(path: str | None = None) -> Path:
    """Resolve configuration file path relative to project root."""
    root = _project_root()
    if path:
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
    else:
        cfg_path = root / "configs" / "default.yaml"
    return cfg_path


def load_config(path: str | None = None) -> dict:
    cfg_path = resolve_config_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件未找到：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    # 把 None 的分支替换成空 dict，避免 .get() 崩
    def _none_to_dict(x):
        if x is None: return {}
        if isinstance(x, dict):
            return {k: _none_to_dict(v) for k, v in x.items()}
        return x
    user_cfg = _none_to_dict(user_cfg)

    return _merge(_DEFAULTS, user_cfg)
