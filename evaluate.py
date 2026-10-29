# -*- coding: utf-8 -*-
"""
评测脚本：在合成测试集(labels.json)上量化检测效果。
指标：
- 二分类（有缺陷 vs 正常）：Precision / Recall / F1
- 每类缺陷：TP/FP/FN 与召回
匹配规则：检测框与真值框 IoU>=0.3 视为命中；命中但类型判错计入该类的漏判/误报。
"""
import os
import json
import cv2
from detector import detect, iou

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
TYPES = ["scratch", "stain", "missing"]
IOU_T = 0.3


def evaluate():
    with open(os.path.join(DATA, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)

    tp = fp = tn = fn = 0
    per = {t: {"tp": 0, "fp": 0, "fn": 0} for t in TYPES}

    for fname, lab in labels.items():
        img = cv2.imread(os.path.join(DATA, fname))
        dets = detect(img)
        gt = lab["defect"]
        box = lab.get("bbox")

        if gt == "none":
            if dets:
                fp += 1
            else:
                tn += 1
            continue

        best, best_type = 0.0, None
        for d in dets:
            if box:
                s = iou(d["bbox"], box)
                if s > best:
                    best, best_type = s, d["type"]
        if best >= IOU_T:
            tp += 1
            if best_type == gt:
                per[gt]["tp"] += 1
            else:
                per[gt]["fn"] += 1
                per[best_type]["fp"] += 1
        else:
            fn += 1
            per[gt]["fn"] += 1

    def prf(t_, p_, f_):
        p = t_ / (t_ + p_) if (t_ + p_) else 0
        r = t_ / (t_ + f_) if (t_ + f_) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        return p, r, f1

    print("=== 二分类（有缺陷 / 正常）===")
    P, R, F = prf(tp, fp, fn)
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precision={P:.3f}  Recall={R:.3f}  F1={F:.3f}")
    print("\n=== 每类缺陷 ===")
    for t in TYPES:
        s = per[t]
        p, r, f1 = prf(s["tp"], s["fp"], s["fn"])
        print(f"{t:8s}  TP={s['tp']} FP={s['fp']} FN={s['fn']}  P={p:.3f} R={r:.3f} F1={f1:.3f}")

    return {"binary": {"P": round(P, 3), "R": round(R, 3), "F1": round(F, 3)}, "per_class": per}


if __name__ == "__main__":
    evaluate()
