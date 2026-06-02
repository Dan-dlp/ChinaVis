#!/usr/bin/env python3
"""
Jaccard 距离 + MDS 降维：更适合「剧本角色集合重叠度」这种集合型数据。

Jaccard 距离 = 1 - |A ∩ B| / |A ∪ B|
MDS (Multidimensional Scaling) 在 Jaccard 距离矩阵上投影到 2D。
"""

import json
import numpy as np
from sklearn.manifold import MDS
from collections import defaultdict

DATA_PATH = '../play_cluster_data.json'
OUT_PATH = '../play_cluster_data_jaccard.json'

# ── 1. 加载 ──────────────────────────────────────────────────
with open(DATA_PATH, 'r') as f:
    data = json.load(f)
n = len(data)
print(f"Loaded {n} plays")

# ── 2. 建角色集合 ───────────────────────────────────────────
char_sets = []
for d in data:
    chars = set(c.strip() for c in d['top_chars'].split('、') if c.strip())
    char_sets.append(chars)

# ── 3. 计算 Jaccard 距离矩阵 ────────────────────────────────
#     上三角，对称，对角=0
print("Computing Jaccard distance matrix...")
dist = np.zeros((n, n), dtype=np.float64)

for i in range(n):
    si = char_sets[i]
    for j in range(i + 1, n):
        sj = char_sets[j]
        intersection = len(si & sj)
        union = len(si | sj)
        if union > 0:
            d = 1.0 - intersection / union
        else:
            d = 1.0
        dist[i, j] = d
        dist[j, i] = d

print(f"  Distance range: [{dist.min():.3f}, {dist.max():.3f}]")
print(f"  Mean distance: {dist.mean():.3f}")

# ── 4. MDS 到 2D ────────────────────────────────────────────
print("Running MDS (metric, 2 components)...")
mds = MDS(
    n_components=2,
    metric=True,
    dissimilarity='precomputed',
    random_state=42,
    max_iter=500,
    n_init=4,
    eps=1e-6,
    normalized_stress='auto'
)
coords = mds.fit_transform(dist)

stress = mds.stress_
print(f"  MDS stress: {stress:.4f}")
print(f"  (Lower is better; < 0.2 = good, < 0.1 = excellent)")

# ── 5. 更新坐标 ─────────────────────────────────────────────
for i, d in enumerate(data):
    d['x'] = float(coords[i, 0])
    d['y'] = float(coords[i, 1])

# ── 6. 包络圆 ───────────────────────────────────────────────
def min_enclosing_circle(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 0, 0, 0
    if len(pts) == 1:
        return float(pts[0, 0]), float(pts[0, 1]), 0.0
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

# ── 7. 写文件 ────────────────────────────────────────────────
output = {
    'method': 'Jaccard Distance + MDS',
    'stress': round(stress, 4),
    'plays': data,
    'envelope_circles': envelope_circles
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False)

compat = '../play_cluster_data_jaccard_flat.json'
with open(compat, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

print(f"\nDone: {OUT_PATH}")
print(f"Flat: {compat}")
print(f"Envelope circles: {len(envelope_circles)} groups")
