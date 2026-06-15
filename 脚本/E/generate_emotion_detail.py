#!/usr/bin/env python3
"""
生成情感标记原文索引 — Panel E 弹窗数据源
=============================================
输入: structured/scripts/*.json  (每剧本完整结构化数据)
输出: public/data/drama_detail/{id}_detail.json

结构:
  scenes[i] = {
    index, normalized_time, total_lines,
    categories: { exclaim/weep/laugh/sing_urgent/sing_mid/sing_slow: [
      { character, text }
    ]}
  }
"""

import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path('/Users/dan/Desktop/VISeer/ChinaVIS')
SCRIPTS_DIR = PROJECT_ROOT / 'structured' / 'scripts'
OUTPUT_DIR = PROJECT_ROOT / 'frontend' / 'app' / 'public' / 'data' / 'drama_detail'

# ── 与 generate_emotion_csv.py 一致的分类逻辑 ──

# 唱腔三档：按关键词
URGENT_KW = ('快板', '流水', '散板', '紧板', '滚板', '垛板', '跺板',
             '索板', '哭板', '倒板', '乱导板', '摇扳', '摇唱', '块板',
             '快二六', '快原板', '快流水', '快三眼板', '叠板', '小快板')

SLOW_KW = ('慢板', '慢三眼', '小慢板', '大松板', '慢二六', '慢原板', '慢流水')

NON_EMOTION = {'白', '同白', '内白', '念', '同念', '内念',
               '引子', '点绛唇', '数板', 'stage_direction', ''}

EXACT_EXCLAIM = {'叫头', '三叫头', '叫板'}
EXACT_WEEP = {'哭', '哭头'}
EXACT_LAUGH = {'笑', }


def classify(lt: str) -> str | None:
    """返回 'sing_urgent'|'sing_mid'|'sing_slow'|'exclaim'|'weep'|'laugh'|None"""
    lt = lt.strip()

    if lt in EXACT_EXCLAIM:
        return 'exclaim'
    if lt in EXACT_WEEP:
        return 'weep'
    if lt in EXACT_LAUGH:
        return 'laugh'
    if lt in NON_EMOTION:
        return None

    # 唱腔子类
    if any(kw in lt for kw in URGENT_KW):
        return 'sing_urgent'
    if any(kw in lt for kw in SLOW_KW):
        return 'sing_slow'

    # 其余所有唱腔
    return 'sing_mid'


# ── 处理 ──
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total = 0

for json_file in sorted(SCRIPTS_DIR.glob('*.json')):
    drama_id = json_file.stem
    total += 1

    with open(json_file, encoding='utf-8') as f:
        drama = json.load(f)

    scenes = drama.get('scenes', [])
    if not scenes:
        continue

    output_scenes = []
    for si, scene in enumerate(scenes):
        lines = scene.get('lines', [])
        valid_lines = [l for l in lines if l.get('line_type') != 'stage_direction']
        total_valid = len(valid_lines)

        cats = {
            'sing_urgent': [],
            'sing_mid': [],
            'sing_slow': [],
            'exclaim': [],
            'weep': [],
            'laugh': [],
        }

        for line in valid_lines:
            lt = line.get('line_type', '')
            cat = classify(lt)
            if cat:
                ch = line.get('character') or ''
                txt = line.get('text', '')
                cats[cat].append({'character': ch, 'text': txt})

        output_scenes.append({
            'index': si + 1,
            'normalized_time': round(si / (len(scenes) - 1), 4) if len(scenes) > 1 else 0.0,
            'total_lines': total_valid,
            'categories': cats,
        })

    # 写文件
    out_path = OUTPUT_DIR / f'{drama_id}_detail.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'id': drama_id, 'scenes': output_scenes}, f, ensure_ascii=False)

print(f"完成: {total} 个剧本 → {OUTPUT_DIR}")
for cat in ['sing_urgent', 'sing_mid', 'sing_slow', 'exclaim', 'weep', 'laugh']:
    pass  # just count
print("每个 detail.json 包含: scenes[].categories.{exclaim/weep/laugh/sing_urgent/sing_mid/sing_slow}")
