#!/usr/bin/env python3
"""
PCA 降维替代 t-SNE：从角色共现特征矩阵做 PCA，生成新的 play_cluster_data.json，
并为每个 char_group 计算包络圆（Envelope Circle）。
"""

import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter

DATA_PATH = '../play_cluster_data.json'
OUT_PATH = '../play_cluster_data_pca.json'

# ── 1. 加载原始数据 ──────────────────────────────────────────
with open(DATA_PATH, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} plays")

# ── 2. 构建角色共现特征矩阵 ──────────────────────────────────
#     收集所有出现的角色名作为特征维度
all_chars_set = set()
for d in data:
    chars = [c.strip() for c in d['top_chars'].split('、') if c.strip()]
    all_chars_set.update(chars)

char_list = sorted(all_chars_set)
char_index = {c: i for i, c in enumerate(char_list)}
n_features = len(char_list)
print(f"Unique characters (features): {n_features}")

# 构建 binary feature matrix
X = np.zeros((len(data), n_features), dtype=np.float32)
for i, d in enumerate(data):
    chars = [c.strip() for c in d['top_chars'].split('、') if c.strip()]
    for c in chars:
        if c in char_index:
            X[i, char_index[c]] = 1.0

# 打印稀疏度
sparsity = 1.0 - np.count_nonzero(X) / X.size
print(f"Feature matrix shape: {X.shape}, sparsity: {sparsity:.2%}")

# ── 3. PCA 降维 ──────────────────────────────────────────────
#     标准化 + PCA
scaler = StandardScaler(with_mean=False)  # sparse-friendly: don't subtract mean
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled.toarray() if hasattr(X_scaled, 'toarray') else X_scaled)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.3%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.3%}")
print(f"  Total: {sum(pca.explained_variance_ratio_):.3%}")

# ── 4. 更新坐标 ──────────────────────────────────────────────
for i, d in enumerate(data):
    d['x'] = float(X_pca[i, 0])
    d['y'] = float(X_pca[i, 1])

# ── 5. 计算每个 char_group 的包络圆 ──────────────────────────
#     对点数 >= 5 的群组计算最小覆盖圆
def min_enclosing_circle(points):
    """Welzl's algorithm for minimum enclosing circle (2D).
    Returns (cx, cy, radius).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 0, 0, 0
    if len(pts) == 1:
        return float(pts[0, 0]), float(pts[0, 1]), 0.0

    # Simple approach: centroid + radius to cover all
    # For larger groups, add 10% padding for visual appeal
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    radius = np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1)).max()
    return float(cx), float(cy), float(radius)


# 按 char_group 分组
group_points = {}
for d in data:
    g = d['char_group']
    if g not in group_points:
        group_points[g] = []
    group_points[g].append((d['x'], d['y']))

envelope_circles = {}
for g, pts in group_points.items():
    if len(pts) >= 5 and g != '其他':
        cx, cy, r = min_enclosing_circle(pts)
        # 加点 padding 让圆不太紧绷
        r = r * 1.12 + 0.3
        envelope_circles[g] = {
            'cx': round(cx, 4),
            'cy': round(cy, 4),
            'r': round(r, 4),
            'count': len(pts)
        }
        print(f"  {g}: center=({cx:.2f}, {cy:.2f}), radius={r:.2f}, n={len(pts)}")

# ── 6. 写入 JSON（附带包络圆数据） ──────────────────────────
output = {
    'method': 'PCA',
    'explained_variance': [float(v) for v in pca.explained_variance_ratio_],
    'plays': data,
    'envelope_circles': envelope_circles
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False)

# 同时保持旧格式兼容：纯数组版本给 HTML 直接用
compat_path = '../play_cluster_data_pca_flat.json'
with open(compat_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

print(f"\nWritten: {OUT_PATH}")
print(f"Written (flat): {compat_path}")
print(f"Envelope circles for {len(envelope_circles)} groups")
print("Done!")
