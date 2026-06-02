#!/usr/bin/env python3
"""
TruncatedSVD 降维（替代 PCA）：对稀疏角色共现矩阵做 SVD，
生成 play_cluster_data_svd.json，为每个 char_group 计算包络圆。

TruncatedSVD (≈LSA) 对高维稀疏二元矩阵效果远好于普通 PCA。
"""

import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
from collections import Counter

DATA_PATH = '../play_cluster_data.json'
OUT_PATH = '../play_cluster_data_svd.json'

# ── 1. 加载 ──────────────────────────────────────────────────
with open(DATA_PATH, 'r') as f:
    data = json.load(f)
print(f"Loaded {len(data)} plays")

# ── 2. 构建稀疏特征矩阵 ─────────────────────────────────────
all_chars_set = set()
for d in data:
    chars = [c.strip() for c in d['top_chars'].split('、') if c.strip()]
    all_chars_set.update(chars)

char_list = sorted(all_chars_set)
char_index = {c: i for i, c in enumerate(char_list)}
n_features = len(char_list)
n_samples = len(data)
print(f"Features: {n_features}, Samples: {n_samples}")

# 直接构建 numpy 稠密矩阵（SVD 输入）
X = np.zeros((n_samples, n_features), dtype=np.float32)
for i, d in enumerate(data):
    chars = [c.strip() for c in d['top_chars'].split('、') if c.strip()]
    for c in chars:
        if c in char_index:
            X[i, char_index[c]] = 1.0

# ── 3. TF-IDF 风格加权（可选，提升区分度）────────────────
#     给稀有角色更高权重（类似 IDF）
doc_freq = np.array((X > 0).sum(axis=0)).flatten()
idf = np.log((n_samples + 1) / (doc_freq + 1)) + 1  # smooth IDF
X_weighted = X * idf

print(f"Sparsity: {1 - np.count_nonzero(X) / X.size:.2%}")

# ── 4. TruncatedSVD ──────────────────────────────────────────
#     n_components=50 先做高维，再取前2维（效果往往更好）
svd = TruncatedSVD(n_components=min(50, n_features - 1), random_state=42)
X_svd = svd.fit_transform(X_weighted)

print(f"\nSVD explained variance (top 5): {svd.explained_variance_ratio_[:5]}")
print(f"  Component 1: {svd.explained_variance_ratio_[0]:.3%}")
print(f"  Component 2: {svd.explained_variance_ratio_[1]:.3%}")
print(f"  Total (50 comps): {sum(svd.explained_variance_ratio_):.3%}")

# ── 5. 取前 2 维做坐标 ──────────────────────────────────────
for i, d in enumerate(data):
    d['x'] = float(X_svd[i, 0])
    d['y'] = float(X_svd[i, 1])

# ── 6. 包络圆 ────────────────────────────────────────────────
def min_enclosing_circle(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return float(pts[0, 0]), float(pts[0, 1]), 0.0 if len(pts) == 1 else (0, 0, 0)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    r = np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1)).max()
    return float(cx), float(cy), float(r)

group_points = {}
for d in data:
    g = d['char_group']
    group_points.setdefault(g, []).append((d['x'], d['y']))

envelope_circles = {}
for g, pts in group_points.items():
    if len(pts) >= 5 and g != '其他':
        cx, cy, r = min_enclosing_circle(pts)
        r = float(r) * 1.10 + 0.2
        envelope_circles[g] = {
            'cx': round(cx, 4), 'cy': round(cy, 4),
            'r': round(r, 4), 'count': len(pts)
        }
        print(f"  {g}: ({cx:.2f},{cy:.2f}) r={r:.2f} n={len(pts)}")

# ── 7. 写文件 ────────────────────────────────────────────────
output = {
    'method': 'TruncatedSVD (LSA)',
    'explained_variance': [float(v) for v in svd.explained_variance_ratio_[:2]],
    'plays': data,
    'envelope_circles': envelope_circles
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False)

compat = '../play_cluster_data_svd_flat.json'
with open(compat, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

print(f"\nDone: {OUT_PATH}")
print(f"Flat: {compat}")
print(f"Envelope circles: {len(envelope_circles)} groups")
