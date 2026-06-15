#!/usr/bin/env python3
"""
生成情感标记密度 CSV — Panel E（河流图）数据源
=================================================
输入: structured/scripts/*.json  (每剧本的完整结构化数据)
输出: public/data/drama_structure/{id}_emotion.csv

每行 = 一个场次
列: scene_index, normalized_time, total_lines,
     sing_urgent, sing_mid, sing_slow, exclaim, weep, laugh
值为该场次中该类情感标记行数 / 场次总行数 (密度)
"""

import json
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path('/Users/dan/Desktop/VISeer/ChinaVIS')
SCRIPTS_DIR = PROJECT_ROOT / 'structured' / 'scripts'
OUTPUT_DIR = PROJECT_ROOT / 'frontend' / 'app' / 'public' / 'data' / 'drama_structure'

# ── 按关键词自动分类 ──
def classify(line_type: str) -> str | None:
    """返回 'sing_urgent' | 'sing_mid' | 'sing_slow' | 'exclaim' | 'weep' | 'laugh' | None"""
    lt = line_type.strip()

    # 叫头（独立判断，避免被唱腔关键词误匹配）
    if lt in ('叫头', '三叫头', '叫板'):
        return 'exclaim'

    # 哭
    if lt in ('哭', '哭头'):
        return 'weep'

    # 笑
    if lt in ('笑',):
        return 'laugh'

    # 非情感行：跳过
    if lt in ('白', '同白', '内白', '念', '同念', '内念',
              '引子', '点绛唇', '数板', 'stage_direction', ''):
        return None

    # ── 唱腔子类：按关键词分三档 ──
    # 急: 快板、流水、散板、紧板、滚板、垛板、跺板、索板、哭板
    urgent_kw = ('快板', '流水', '散板', '紧板', '滚板', '垛板', '跺板',
                 '索板', '哭板', '倒板', '乱导板', '摇扳', '摇唱', '块板',
                 '快二六', '快原板', '快流水', '快三眼板', '叠板', '小快板')
    if any(kw in lt for kw in urgent_kw):
        return 'sing_urgent'

    # 慢: 慢板、三眼、小慢板、大松板
    slow_kw = ('慢板', '慢三眼', '小慢板', '大松板', '慢二六', '慢原板', '慢流水')
    if any(kw in lt for kw in slow_kw):
        return 'sing_slow'

    # 中: 其余所有唱腔（原板、二六、摇板、导板、正板、平板、吹腔、梆子、南梆子、高拨子、四平调、风入松、回龙、汉调、碰板、顶板、联弹 等）
    return 'sing_mid'


# ── 处理 ──
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

unmatched = Counter()
total_dramas = 0

for json_file in sorted(SCRIPTS_DIR.glob('*.json')):
    drama_id = json_file.stem
    total_dramas += 1

    with open(json_file, encoding='utf-8') as f:
        drama = json.load(f)

    scenes = drama.get('scenes', [])
    if not scenes:
        continue

    rows = []
    for si, scene in enumerate(scenes):
        lines = scene.get('lines', [])

        # 过滤掉 stage_direction，统计有效情感行和总行
        valid_lines = [l for l in lines if l.get('line_type') != 'stage_direction']
        total = len(valid_lines)
        if total == 0:
            continue

        counts = Counter()
        for line in valid_lines:
            cat = classify(line.get('line_type', ''))
            if cat:
                counts[cat] += 1

        rows.append({
            'scene_index': si + 1,
            'normalized_time': si / (len(scenes) - 1) if len(scenes) > 1 else 0.0,
            'total_lines': total,
            'sing_urgent': counts['sing_urgent'] / total,
            'sing_mid': counts['sing_mid'] / total,
            'sing_slow': counts['sing_slow'] / total,
            'exclaim': counts['exclaim'] / total,
            'weep': counts['weep'] / total,
            'laugh': counts['laugh'] / total,
        })

    if not rows:
        continue

    # 写入 CSV
    csv_path = OUTPUT_DIR / f'{drama_id}_emotion.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'scene_index', 'normalized_time', 'total_lines',
            'sing_urgent', 'sing_mid', 'sing_slow', 'exclaim', 'weep', 'laugh'
        ])
        writer.writeheader()
        writer.writerows(rows)

print(f"完成: {total_dramas} 个剧本 → {OUTPUT_DIR}")
print(f"\n数据列: scene_index, normalized_time, total_lines, sing_urgent, sing_mid, sing_slow, exclaim, weep, laugh")
print(f"值范围: 0.0 ~ 1.0 (密度 = 该类行数 / 场次有效行数)")
