# -*- coding: utf-8 -*-
"""
演示脚本：对单张/整批图像跑缺陷检测，输出带标注框的结果图 + 自包含 HTML 报告。
HTML 把原图/结果图以 base64 内嵌，可直接发给 HR 或放进作品集。
"""
import os
import base64
import cv2
from detector import detect

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
OUT = os.path.join(HERE, "output")
COLORS = {"scratch": (0, 0, 255), "stain": (0, 165, 255), "missing": (255, 0, 0)}


def annotate(img, dets):
    out = img.copy()
    for d in dets:
        x, y, w, h = d["bbox"]
        c = COLORS.get(d["type"], (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        cv2.putText(out, f"{d['type']} {d['confidence']}", (x, max(8, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_batch():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for f in sorted(os.listdir(DATA)):
        if not f.endswith(".png") or f == "labels.json":
            continue
        img = cv2.imread(os.path.join(DATA, f))
        dets = detect(img)
        ann = annotate(img, dets)
        cv2.imwrite(os.path.join(OUT, f), ann)
        rows.append((f, dets))

    # 生成 HTML 报告
    cards = []
    for f, dets in rows:
        o = os.path.join(OUT, f)
        orig = b64(os.path.join(DATA, f))
        ann = b64(o)
        tags = " · ".join(f"{d['type']}({d['confidence']})" for d in dets) or "未检出缺陷"
        cards.append(f"""
        <div class="card">
          <div class="imgs">
            <img src="data:image/png;base64,{orig}"><img src="data:image/png;base64,{ann}">
          </div>
          <div class="meta">{f} — <b>{tags}</b></div>
        </div>""")
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>工业表面缺陷检测 · 演示报告</title>
<style>
 body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#f5f6f8;color:#222}}
 header{{background:#1f2a44;color:#fff;padding:18px 24px}} header h1{{margin:0;font-size:18px}}
 header p{{margin:4px 0 0;opacity:.8;font-size:13px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:18px}}
 .card{{background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
 .imgs{{display:flex}} .imgs img{{width:50%;display:block}}
 .meta{{padding:8px 10px;font-size:12px;border-top:1px solid #eee}}
</style></head><body>
<header><h1>工业表面缺陷检测 · 演示报告</h1>
<p>经典 CV（Canny + 形态学 + 轮廓分析）· 共 {len(rows)} 张 · 左:原图 右:检测标注</p></header>
<div class="grid">{''.join(cards)}</div></body></html>"""
    rep = os.path.join(OUT, "report.html")
    with open(rep, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"已处理 {len(rows)} 张，标注图 -> {OUT}，报告 -> {rep}")


if __name__ == "__main__":
    run_batch()
