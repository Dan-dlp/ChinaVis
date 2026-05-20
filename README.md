# ChinaVIS 2026 — 京剧剧本数据集

> ChinaVis 2026 可视化竞赛 · 赛道1-I · 上海科技大学 Viseer 可视化分析与交互课题组

---

## 项目概述

本仓库包含针对 **ChinaVis 2026 数据可视分析与人文创意赛（赛道1-I）** 处理的结构化京剧剧本数据集。

原始数据为跨来源、跨流派的京剧剧本 PDF，共 **1473 个剧本**，经过系统化文本提取与结构化解析后，生成本仓库中的 `structured/` 数据集，供后续可视分析使用。

---

## 仓库结构

```
ChinaVIS/
├── structured/                  # 结构化数据集（主要数据）
│   ├── index.json               # 全量剧本索引（轻量摘要）
│   ├── scripts/                 # 每个剧本的完整结构化 JSON（1473 个文件）
│   ├── graph.json               # 全局角色关系网络
│   ├── nlp_corpus.json          # NLP 全文语料库
│   ├── drama_topics.json        # 每个剧本的主题权重分布
│   ├── topics_meta.json         # 25 个主题的元数据
│   ├── version_groups.json      # 同名剧本多版本对照
│   └── pending_pdfs.json        # 处理过程日志（已完成）
├── 1-I_answerSheet (1).docx     # 答卷模板
├── ChinaVis2026_赛题I_比赛说明.docx  # 赛题说明
└── README.md
```

> 原始 PDF 文件（1473 个，共约 620MB）不包含在本仓库中。

---

## structured/ 数据详解

### 1. `index.json` — 全量索引

**格式**：`{ "total": 1473, "dramas": [ ... ] }`

每条 drama 记录包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 8位唯一编号（如 `"01001001"`） |
| `title` | string | 剧目正名 |
| `title_normalized` | string | 去括号规范化标题 |
| `alt_titles` | array | 别名列表 |
| `source_collection` | string | 来源文献（戏考 / 京剧汇编 / 京剧丛刊 / 录音唱片本 等） |
| `source_id` | string | 来源文件夹编码（如 `"01000000"`） |
| `pages` | int | PDF 页数 |
| `scene_count` | int | 场次数 |
| `character_count` | int | 角色数 |
| `role_types` | array | 出现的行当类型（如 `["老生","旦","净","丑"]`） |
| `narrative_curve` | array | 叙事强度曲线（每场的唱词密度，float 数组） |
| `musical_arc` | array | 音乐风格序列（每场主要唱腔，如 `"西皮"/"二黄"/null`） |
| `topics` | array | 主题分布（`[{topic_id, weight}, ...]`，共 25 个主题） |
| `drama_type` | string | 剧目类型（历史战争 / 家庭伦理 / 公案 / 神话 等） |

---

### 2. `scripts/` — 单剧本完整 JSON（1473 个文件）

文件名即剧本 ID，如 `01001001.json`（空城计）。

**顶层字段**：

| 字段 | 说明 |
|---|---|
| `id` / `title` / `alt_titles` | 基本信息，同 index |
| `source_collection` / `source_id` / `source_note` | 来源详情 |
| `characters` | 角色列表，含 `name`、`role_type`、`gender` |
| `plot_summary` | 剧情简介 |
| `notes` | 注释（演出史、流派说明等） |
| `full_text` | 提取的完整原文（前约 6000 字） |
| `topics` | 主题权重分布 |
| `drama_type` | 剧目类型 |
| `statistics` | 全剧统计信息（见下） |
| `scenes` | 场次列表（见下） |

**`statistics` 字段**：

```json
{
  "scene_count": 6,
  "character_count": 7,
  "role_type_dist": { "老生": 2, "净": 2, "小生": 1, "丑": 2 },
  "narrative_curve": [0.0, 0.0, 0.275, 0.0, ...],
  "musical_arc": ["西皮", null, "西皮", null, ...],
  "cooccurrence": { "诸葛亮|司马懿": 3, "司马懿|司马昭": 5, ... }
}
```

**`scenes[i]` 场次结构**：

```json
{
  "scene_index": 2,
  "label": "【第二场】",
  "characters_present": ["钟会", "旗牌", "四将", "邓艾"],
  "lines": [ ... ],
  "interactions": [],
  "stats": { ... }
}
```

**`lines[j]` 台词行结构**（核心）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `character` | string\|null | 说话角色名；`null` 表示舞台指示 |
| `line_type` | string | 台词类型（见下方类型说明） |
| `musical_style` | string\|null | 具体唱腔名（如 `"西皮摇板"`） |
| `style_family` | string\|null | 唱腔大类（`"西皮"` / `"二黄"` / `"南梆子"` 等） |
| `text` | string | 台词内容 |

**`line_type` 说明**：

| 类型值 | 含义 |
|---|---|
| `白` | 说白（口语对白） |
| `念` | 韵白（有节奏的念词） |
| `唱` | 唱腔（配乐演唱） |
| `同白` / `同念` / `同唱` | 多角色同时念/唱 |
| `内白` / `内唱` | 场外（幕后）说白/唱 |
| `引子` / `点绛唇` | 上场时的固定曲牌 |
| `哭头` / `叫头` / `叫板` | 特定戏曲表演程式 |
| `西皮摇板` / `二黄散板` 等 | 具体唱腔形式（西皮/二黄 + 板式） |
| `stage_direction` | 舞台指示（上/下场、动作说明等），`character=null` |

示例：

```json
{
  "character": "钟会",
  "line_type": "点绛唇",
  "musical_style": null,
  "style_family": null,
  "text": "兵伐蜀地，建立功绩，施妙计，惯用心机，要把成都取。"
},
{
  "character": "钟会",
  "line_type": "念",
  "musical_style": null,
  "style_family": null,
  "text": "髫年称早慧，曾作秘书郎。妙计平西蜀，要比张子房。"
}
```

---

### 3. `graph.json` — 全局角色关系网络

**格式**：`{ "nodes": [...], "links": [...] }`

- **nodes**（3581 个角色节点）：

```json
{ "id": "诸葛亮", "role_type": "老生", "gender": "male", "drama_count": 93 }
```

- **links**（17927 条共现边）：

```json
{
  "source": "司马懿",
  "target": "诸葛亮",
  "weight": 7,
  "drama_count": 4,
  "drama_ids": ["01001001", "70006105", "14003003", "02004004"]
}
```

`weight` 为总共现次数，`drama_count` 为共现剧本数。

---

### 4. `nlp_corpus.json` — NLP 全文语料库

**格式**：数组，1473 条记录，每条：

```json
{ "id": "01001001", "title": "空城计", "text": "完整提取文本..." }
```

可直接用于 TF-IDF、词向量、LLM 分析等。

---

### 5. `drama_topics.json` — 主题权重

**格式**：`{ "剧本id": [{topic_id, weight}, ...] }`

每个剧本分配 5 个最相关主题（共 25 个主题，基于 LDA 主题模型）。

---

### 6. `topics_meta.json` — 主题元数据

25 个主题，每条：

```json
{
  "topic_id": 2,
  "label": "刘备·诸葛亮·张飞",
  "keywords": ["刘备", "诸葛亮", "张飞", "关羽", "赵云", ...],
  "top_dramas": ["03065006", "04003001", "80000036", ...]
}
```

---

### 7. `version_groups.json` — 多版本对照

**格式**：`{ "剧目名": [版本1, 版本2, ...] }`

共 **216 组** 存在多个版本的剧目，每个版本记录：

```json
{
  "id": "01001001",
  "source_collection": "戏考",
  "scene_count": 6,
  "narrative_curve": [0.0, 0.0, 0.275, ...]
}
```

适用于 Q4（叙事结构跨版本比较）分析。

---

## 数据来源说明

### PDF 编号体系（8位数字）

格式：`XXYYYZZZ`

| 段 | 含义 |
|---|---|
| `XX`（位1-2） | 来源大类：`01`-`14` 按流派/来源分类；`70` 特定演员/版本；`80` 综合经典；`90`/`94` 其他 |
| `YYY`（位3-5） | 子类编码（具体剧目集或演员组） |
| `ZZZ`（位6-8） | 序号 |

### 来源文献

| 来源编码 | 文献名称 | 剧本数 |
|---|---|---|
| `01000000` | 中国京剧戏考 | 448 |
| `02000000` | 京剧汇编（二） | 194 |
| `03000000` | 京剧汇编 | 360 |
| `04000000` | 京剧丛刊 | 71 |
| `05000000` | 其他汇编 | 67 |
| `70xxx000` | 特定演员/版本合集 | 227 |
| `80000000` | 录音唱片本 | 34 |
| `90000000` / `94000000` | 其他来源 | 17 |

---

## 五道赛题与数据对应

| 题目 | 核心数据 | 关键字段 |
|---|---|---|
| **Q1** 行当推断与历史演变 | `scripts/` 角色行 | `characters[].role_type`、`line_type`、`lines[].text` |
| **Q2** 角色关系网络 | `graph.json`、`scripts/` 场次共现 | `links.weight`、`statistics.cooccurrence` |
| **Q3** 主题提取与比较 | `drama_topics.json`、`topics_meta.json` | `topics[].weight`、`keywords` |
| **Q4** 叙事结构分析 | `index.json`、`version_groups.json` | `narrative_curve`、`musical_arc`、`scene_count` |
| **Q5** 三要素综合可视分析 | 全部数据联合使用 | — |

---

## 团队信息

**上海科技大学 Viseer 可视化分析与交互课题组**  
队长：段林沛  
参赛人数：6 人  
赛道：ChinaVis 2026 赛道1-I 数据可视分析与人文创意赛
