# -*- coding: utf-8 -*-
"""
合成「工业金属表面缺陷」数据集生成器
- 生成正常图 + 三类缺陷：划痕(scratch)/污点(stain)/缺损(missing)
- 输出到 dataset/ 目录，并写 labels.json（含真值 bbox，供 evaluate.py 评测）
- 用途：给缺陷检测 demo 提供可复现的测试素材（无需真实产线摄像头）
"""
import os
import json
import random
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
IMG = 256

random.seed(2027)
np.random.seed(2027)


def make_plate():
    """生成一张带拉丝纹理的金属底板（模拟工业零件表面）"""
    base = np.full((IMG, IMG), 150, np.uint8)
    noise = np.random.normal(0, 5, (IMG, IMG)).astype(np.int16)
    # 横向拉丝纹理
    streak = np.tile(np.sin(np.linspace(0, 40 * np.pi, IMG)).astype(np.float32), (IMG, 1)) * 3
    img = base.astype(np.int16) + noise + streak.astype(np.int16)
    return np.clip(img, 90, 210).astype(np.uint8)


def add_scratch(img):
    """在图上画一条细长的划痕，返回 (图, bbox)"""
    x = random.randint(20, IMG - 20)
    y = random.randint(20, IMG - 20)
    angle = random.uniform(0, np.pi)
    length = random.randint(60, 170)
    pts = []
    cx, cy = x, y
    for t in np.linspace(0, length, 14):
        jx = cx + t * np.cos(angle) + random.randint(-5, 5)
        jy = cy + t * np.sin(angle) + random.randint(-5, 5)
        pts.append((int(jx), int(jy)))
    pts = np.array(pts)
    out = img.copy()
    color = int(random.choice([55, 225]))
    cv2.polylines(out, [pts], False, color, random.randint(1, 2), cv2.LINE_AA)
    x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
    return out, [x0, y0, x1 - x0, y1 - y0]


def add_stain(img):
    """在图上画一块不规则污点，返回 (图, bbox)"""
    cx, cy = random.randint(45, IMG - 45), random.randint(45, IMG - 45)
    a, b = random.randint(10, 26), random.randint(10, 26)
    out = img.copy()
    cv2.ellipse(out, (cx, cy), (a, b), random.randint(0, 180), 0, 360, int(random.choice([45, 215])), -1)
    return out, [cx - a, cy - b, 2 * a, 2 * b]


def add_missing(img):
    """模拟边角缺损（崩角），返回 (图, bbox)"""
    corner = random.choice([(0, 0), (IMG, 0), (0, IMG), (IMG, IMG)])
    size = random.randint(28, 58)
    out = img.copy()
    if corner == (0, 0):
        pts = np.array([[0, 0], [size, 0], [0, size]])
    elif corner == (IMG, 0):
        pts = np.array([[IMG, 0], [IMG, size], [IMG - size, 0]])
    elif corner == (0, IMG):
        pts = np.array([[0, IMG], [0, IMG - size], [size, IMG]])
    else:
        pts = np.array([[IMG, IMG], [IMG, IMG - size], [IMG - size, IMG]])
    cv2.fillPoly(out, [pts], int(random.choice([65, 235])))
    x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
    return out, [x0, y0, x1 - x0, y1 - y0]


def gen(count_per=40):
    os.makedirs(DATA, exist_ok=True)
    labels = {}
    types = [("none", None), ("scratch", add_scratch), ("stain", add_stain), ("missing", add_missing)]
    idx = 0
    for tname, fn in types:
        for _ in range(count_per):
            img = make_plate()
            bbox = None
            if fn:
                img, bbox = fn(img)
            out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            fname = f"{tname}_{idx:03d}.png"
            cv2.imwrite(os.path.join(DATA, fname), out)
            labels[fname] = {"defect": tname, "bbox": bbox}
            idx += 1
    with open(os.path.join(DATA, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"已生成 {idx} 张图 -> {DATA}")


if __name__ == "__main__":
    gen(40)
