import argparse, os
from pathlib import Path
import cv2
from src.augment.fog import EnhancedFogSynthesizer as FogSynthesizer


def process_folder(inp, outp, levels=("light","medium","heavy"), limit=None):
    inp = Path(inp); outp = Path(outp)
    outp.mkdir(parents=True, exist_ok=True)
    files = [p for p in inp.rglob("*") if p.suffix.lower() in [".jpg",".png",".jpeg"]]
    if limit: files = files[:limit]

    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            print("Skip unreadable:", p)
            continue
        for lv in levels:
            synth = FogSynthesizer(
                level=lv,
                y_h_ratio=0.42,
                perlin_scale_ratio=0.18,
                perlin_octaves=2,
                horizon_softness=0.07,
                global_veil=0.5,
                depth_blur_max=4.0
            )
            hazy, meta = synth.synthesize(img)
            rel = p.relative_to(inp)
            out_dir = outp / lv / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / rel.name), hazy)
        if i % 20 == 0:
            print(f"[{i}/{len(files)}] {p}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="清晰道路图片目录")
    ap.add_argument("--output", required=True, help="输出雾化图片目录")
    ap.add_argument("--levels", default="light,medium,heavy")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    lv = [s.strip() for s in args.levels.split(",") if s.strip()]
    process_folder(args.input, args.output, lv, limit=(args.limit or None))
