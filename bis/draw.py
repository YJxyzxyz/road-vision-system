import cv2
from typing import List, Tuple
from src.detect.types import Detection

def draw_detections(img, dets: List[Detection], thickness=2, font_scale=0.6):
    color_tbl = _color_table()
    for d in dets:
        c = color_tbl[d.cls_id % len(color_tbl)]
        p1 = (int(d.x1), int(d.y1)); p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, c, thickness)
        label = f"{d.cls_name} {d.conf:.2f}"
        _draw_label(img, label, p1, c, font_scale, thickness)

def _draw_label(img, text, topleft, color, font_scale, thickness):
    x,y = topleft
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th), _ = cv2.getTextSize(text, font, font_scale, max(1, thickness))
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 4), font, font_scale, (255,255,255), max(1, thickness-1), cv2.LINE_AA)

def _color_table() -> Tuple[Tuple[int,int,int], ...]:
    return (
        (255, 128,  64), (  0, 255, 255), ( 80, 175,  76),
        (255,   0, 255), (  0, 128, 255), (255,  64,  64),
        ( 64, 255,  64), (128, 128, 255), (255, 200,   0),
        (  0, 255, 128),
    )
