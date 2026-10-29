# -*- coding: utf-8 -*-
"""
工业表面缺陷检测核心算法（经典计算机视觉，无需训练模型）
主流程：灰度 -> 中值滤波 -> 与原图做差分 -> 阈值化得到"异常区掩膜" -> 形态学
       -> 轮廓发现 -> 按面积/长宽比/是否贴边 判别缺陷类型。
为什么用「中值差分」而非纯 Canny：
  - 崩角/污点是实心填充区，靠"颜色偏离背景"就能整块抓出，不依赖边缘闭合；
  - 划痕是细线，做形态学膨胀加粗后再提轮廓，避免被面积阈值误删。
全部步骤可解释，面试能逐句讲清楚。
"""
import cv2
import numpy as np

IMG = 256
MIN_AREA = 80                     # 小于此面积的连通域视为噪声
DIFF_T = 35                       # |原图-全图中值| 超过此值判为异常（正常拉丝纹理仅偏离~8）
BORDER = 14                       # 距图像边缘小于此值视为「贴边」（崩角特征）


def detect(img_bgr):
    """输入 BGR 图，返回检测列表 [{'bbox':[x,y,w,h], 'type', 'confidence'}]"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 用「全图中位数」做差分：污点/崩角/划痕都显著偏离全局背景，
    # 而正常拉丝纹理只偏离 ~8，因此阈值能干净地分出缺陷区，且不会因
    # 局部中值把大块实心缺陷"抵消"掉（这是纯局部中值法的盲区）。
    gm = int(np.median(gray))
    diff = cv2.absdiff(gray, np.full_like(gray, gm))
    _, thr = cv2.threshold(diff, DIFF_T, 255, cv2.THRESH_BINARY)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(thr, k_small, iterations=1)       # 加粗细线（划痕）
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_small, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # 用旋转包围盒(minAreaRect)算真实长宽比：斜向划痕的轴对齐包围盒接近
        # 正方形，但旋转后是细长条，据此才能正确识别划痕、不被误判成污点。
        rw, rh = cv2.minAreaRect(c)[1]
        aspect = max(rw, rh) / max(1, min(rw, rh))      # 真实长宽比
        extent = area / max(1, w * h)                  # 延展度（实心度）
        near_border = (x < BORDER or y < BORDER or x + w > IMG - BORDER or y + h > IMG - BORDER)

        if aspect > 2.5:
            dtype, conf = "scratch", min(0.99, 0.6 + 0.05 * aspect)
        elif near_border and w * h > 400:
            dtype, conf = "missing", 0.82
        elif extent > 0.6:
            dtype, conf = "stain", min(0.99, 0.6 + extent * 0.35)
        else:
            dtype = "missing" if near_border else "stain"
            conf = 0.78 if near_border else 0.55
        dets.append({"bbox": [int(x), int(y), int(w), int(h)], "type": dtype, "confidence": round(conf, 2)})
    return dets


def iou(a, b):
    """两 bbox [x,y,w,h] 的 IoU"""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union
