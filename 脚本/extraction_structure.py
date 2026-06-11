import pathlib
import json
import math
from collections import Counter

import numpy as np
import pandas as pd


# =========================
# Config
# =========================

WINDOW_SIZE = 20
STEP_SIZE = 10

IGNORE_LINE_TYPES = ["stage_direction"]

# line_type categories (semantic grouping)
LINE_TYPE_CATEGORIES = {
    "白": "speech",
    "同白": "chorus",
    "念": "recitation",
    "叫头": "exclamation",
    "笑": "laughter",
    "哭头": "weeping",
}

SCRIPT_DIR = pathlib.Path("structured/scripts")
OUT_DIR = pathlib.Path("output/data/drama_structure")


# =========================
# Utils
# =========================


def safe_ratio(a, b):
    return a / b if b != 0 else 0


def shannon_entropy(items):
    if len(items) == 0:
        return 0

    counter = Counter(items)
    total = len(items)

    entropy = 0
    for c in counter.values():
        p = c / total
        entropy -= p * math.log2(p)

    return entropy


def consecutive_streaks(sequence, target):
    streaks = []
    current = 0

    for item in sequence:
        if item == target:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0

    if current > 0:
        streaks.append(current)

    return streaks


def classify_line_type(lt):
    if lt in LINE_TYPE_CATEGORIES:
        return LINE_TYPE_CATEGORIES[lt]
    return "musical"


def extract_window_features(
    window,
    start,
    end,
    total_lines,
    style_family_list,
    musical_style_list,
    non_musical_line_types,
):
    window_len = len(window)
    center = (start + end) / 2
    if total_lines >= WINDOW_SIZE:
        first_center = WINDOW_SIZE / 2
        last_center = total_lines - WINDOW_SIZE / 2
        denom = last_center - first_center
        normalized_time = ((center - first_center) / denom) if denom != 0 else 0.5
    else:
        normalized_time = 0.5

    line_type_categories = [classify_line_type(x["line_type"]) for x in window]
    lt_counter = Counter(line_type_categories)

    musical_ratio = safe_ratio(lt_counter.get("musical", 0), window_len)
    speech_ratio = safe_ratio(lt_counter.get("speech", 0), window_len)
    recitation_ratio = safe_ratio(lt_counter.get("recitation", 0), window_len)
    chorus_ratio = safe_ratio(lt_counter.get("chorus", 0), window_len)
    exclamation_ratio = safe_ratio(lt_counter.get("exclamation", 0), window_len)
    laughter_ratio = safe_ratio(lt_counter.get("laughter", 0), window_len)
    weeping_ratio = safe_ratio(lt_counter.get("weeping", 0), window_len)

    raw_lt_ratios = {}
    for lt in non_musical_line_types:
        count = sum(1 for x in window if x["line_type"] == lt)
        raw_lt_ratios[lt] = safe_ratio(count, window_len)

    characters = [x["character"] for x in window]
    unique_characters = set(characters)
    character_count = len(unique_characters)

    switches = 0
    for i in range(len(characters) - 1):
        if characters[i] != characters[i + 1]:
            switches += 1

    role_switch_frequency = safe_ratio(switches, len(characters) - 1)
    speaker_entropy = shannon_entropy(characters)
    char_counter = Counter(characters)
    dominant_character_ratio = max(char_counter.values()) / len(characters)

    musical_streaks = consecutive_streaks(line_type_categories, "musical")
    avg_musical_streak = np.mean(musical_streaks) if musical_streaks else 0
    max_musical_streak = max(musical_streaks) if musical_streaks else 0

    style_families_in_window = [x["style_family"] for x in window if x["style_family"]]
    sf_counter = Counter(style_families_in_window)

    style_family_ratios = {}
    for sf in style_family_list:
        style_family_ratios[f"family_{sf}"] = safe_ratio(
            sf_counter.get(sf, 0), window_len
        )

    mode_switches = 0
    for i in range(len(style_families_in_window) - 1):
        if style_families_in_window[i] != style_families_in_window[i + 1]:
            mode_switches += 1
    mode_switch_rate = safe_ratio(mode_switches, len(style_families_in_window) - 1)

    musical_styles_in_window = [
        x["musical_style"] for x in window if x["musical_style"]
    ]
    ms_counter = Counter(musical_styles_in_window)

    musical_style_ratios = {}
    for ms in musical_style_list:
        musical_style_ratios[f"style_{ms}"] = safe_ratio(
            ms_counter.get(ms, 0), window_len
        )

    musical_style_entropy = shannon_entropy(musical_styles_in_window)
    narrative_strength = np.mean([x["narrative_strength"] for x in window])

    row = {
        "window_start": start,
        "window_end": end,
        "normalized_time": normalized_time,
        "musical_ratio": musical_ratio,
        "speech_ratio": speech_ratio,
        "recitation_ratio": recitation_ratio,
        "chorus_ratio": chorus_ratio,
        "exclamation_ratio": exclamation_ratio,
        "laughter_ratio": laughter_ratio,
        "weeping_ratio": weeping_ratio,
        "role_switch_frequency": role_switch_frequency,
        "character_count": character_count,
        "speaker_entropy": speaker_entropy,
        "dominant_character_ratio": dominant_character_ratio,
        "avg_musical_streak": avg_musical_streak,
        "max_musical_streak": max_musical_streak,
        "mode_switch_rate": mode_switch_rate,
        "musical_style_entropy": musical_style_entropy,
        "narrative_strength": narrative_strength,
    }
    row.update(style_family_ratios)
    row.update(musical_style_ratios)
    for lt in non_musical_line_types:
        row[f"linetype_{lt}"] = raw_lt_ratios[lt]

    return row


# =========================
# Process a single drama
# =========================


def process_drama(drama_path):
    drama_id = drama_path.stem

    with open(drama_path, "r", encoding="utf-8") as f:
        drama = json.load(f)

    # ---------------------
    # Flatten line sequence & discover types
    # ---------------------

    all_lines = []

    scene_narrative_map = {}

    if "statistics" in drama and "narrative_curve" in drama["statistics"]:
        curve = drama["statistics"]["narrative_curve"]
        for i, val in enumerate(curve):
            scene_narrative_map[i] = val

    all_style_families = set()
    all_musical_styles = set()
    all_raw_line_types = set()

    for scene in drama["scenes"]:
        scene_idx = scene.get("scene_index", 0)
        narrative_strength = scene_narrative_map.get(scene_idx, 0)

        for line in scene["lines"]:
            line_type = line.get("line_type", "")

            if line_type in IGNORE_LINE_TYPES:
                continue

            style_family = line.get("style_family", "") or ""
            musical_style = line.get("musical_style", "") or ""

            if style_family:
                all_style_families.add(style_family)
            if musical_style:
                all_musical_styles.add(musical_style)
            if line_type:
                all_raw_line_types.add(line_type)

            all_lines.append(
                {
                    "scene_index": scene_idx,
                    "character": line.get("character", "UNKNOWN"),
                    "line_type": line_type,
                    "musical_style": musical_style,
                    "style_family": style_family,
                    "text": line.get("text", ""),
                    "narrative_strength": narrative_strength,
                }
            )

    total_lines = len(all_lines)

    if total_lines == 0:
        print(f"  [{drama_id}] SKIP: no usable lines after filtering")
        return

    style_family_list = sorted(all_style_families)
    musical_style_list = sorted(all_musical_styles)
    non_musical_line_types = sorted(
        lt
        for lt in all_raw_line_types
        if lt not in IGNORE_LINE_TYPES and classify_line_type(lt) != "musical"
    )

    print(
        f"  [{drama_id}] lines={total_lines}  families={len(style_family_list)}  "
        f"styles={len(musical_style_list)}  lt={len(non_musical_line_types)}"
    )

    # ---------------------
    # Sliding Window Extraction
    # ---------------------

    results = []

    if total_lines < WINDOW_SIZE:
        print(
            f"  [{drama_id}] SHORT: only {total_lines} usable lines (< {WINDOW_SIZE}), "
            "using full drama as one feature window"
        )
        results.append(
            extract_window_features(
                all_lines,
                0,
                total_lines,
                total_lines,
                style_family_list,
                musical_style_list,
                non_musical_line_types,
            )
        )
    else:
        for start in range(0, total_lines - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            window = all_lines[start:end]
            results.append(
                extract_window_features(
                    window,
                    start,
                    end,
                    total_lines,
                    style_family_list,
                    musical_style_list,
                    non_musical_line_types,
                )
            )

    # ---------------------
    # Save
    # ---------------------

    df = pd.DataFrame(results)

    core_cols = [
        "window_start",
        "window_end",
        "normalized_time",
        "musical_ratio",
        "speech_ratio",
        "recitation_ratio",
        "chorus_ratio",
        "exclamation_ratio",
        "laughter_ratio",
        "weeping_ratio",
        "role_switch_frequency",
        "character_count",
        "speaker_entropy",
        "dominant_character_ratio",
        "avg_musical_streak",
        "max_musical_streak",
        "mode_switch_rate",
        "musical_style_entropy",
        "narrative_strength",
    ]

    family_cols = sorted(
        [c for c in df.columns if c.startswith("family_")],
        key=lambda x: df[x].sum(),
        reverse=True,
    )
    style_cols = sorted(
        [c for c in df.columns if c.startswith("style_")],
        key=lambda x: df[x].sum(),
        reverse=True,
    )
    linetype_cols = sorted(
        [c for c in df.columns if c.startswith("linetype_")],
        key=lambda x: df[x].sum(),
        reverse=True,
    )

    ordered_cols = core_cols + family_cols + style_cols + linetype_cols
    for c in df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)

    df = df[ordered_cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{drama_id}_features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"    -> {out_path} ({len(ordered_cols)} cols)")


# =========================
# Main
# =========================


def main():
    json_files = sorted(SCRIPT_DIR.glob("*.json"))
    print(f"Processing {len(json_files)} dramas in {SCRIPT_DIR}/\n")

    for f in json_files:
        process_drama(f)

    print(f"\nDone. Output in {OUT_DIR}/")


if __name__ == "__main__":
    main()
