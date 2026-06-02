#!/usr/bin/env python3
"""
从 t-SNE 坐标计算包络圆，生成带 envelope_circles 的 JSON。
不做降维，直接用现有坐标。
"""

import json
import numpy as np
from collections import defaultdict

DATA_PATH = '../play_cluster_data.json'
OUT_PATH = '../play_cluster_data_with_envelopes.json'

with open(DATA_PATH, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} plays (existing t-SNE coords)")

# ── 包络圆 ────────────────────────────────────────────────────
def min_enclosing_circle(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return float(pts[0,0]), float(pts[0,1]), 0.0 if len(pts) == 1 else (0, 0, 0)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    r = np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1)).max()
    return float(cx), float(cy), float(r)

group_points = defaultdict(list)
for d in data:
    group_points[d['char_group']].append((d['x'], d['y']))

envelope_circles = {}
for g, pts in group_points.items():
    if len(pts) >= 5 and g != '其他':
        cx, cy, r = min_enclosing_circle(pts)
        r = float(r) * 1.08 + 0.15
        envelope_circles[g] = {
            'cx': round(cx, 4), 'cy': round(cy, 4),
            'r': round(r, 4), 'count': len(pts)
        }
        print(f"  {g}: ({cx:.2f},{cy:.2f}) r={r:.2f} n={len(pts)}")

# ── 输出 ──────────────────────────────────────────────────────
output = {
    'method': 't-SNE',
    'plays': data,
    'envelope_circles': envelope_circles
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False)

print(f"\nDone: {OUT_PATH}")
print(f"Envelope circles for {len(envelope_circles)} groups")
