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

def load_config(path: str | None = None) -> dict:
    root = _project_root()
    cfg_path = Path(path) if path else (root / "configs" / "default.yaml")
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
