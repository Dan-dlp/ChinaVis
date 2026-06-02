#!/usr/bin/env python3
"""
京剧剧本降维聚类 Pipeline
- 特征：LDA主题(30维) + 角色行当(8维) + 叙事特征(5维) = 43维
- 降维：UMAP → 2D
- 分类：关键词规则 → 8大类 + 二级系列群
- 输出：Script Dimension Reduction/drama_embeddings.json
"""

import json, os, re, warnings
import numpy as np
warnings.filterwarnings('ignore')

BASE = '/sessions/exciting-affectionate-dijkstra/mnt/ChinaVIS/structured'
OUT_DIR = '/sessions/exciting-affectionate-dijkstra/mnt/ChinaVIS/Script Dimension Reduction'
os.makedirs(OUT_DIR, exist_ok=True)

print("Step 1: 加载数据...")
with open(f'{BASE}/index.json') as f:
    idx = json.load(f)
dramas = idx['dramas']

scripts = {}
for d in dramas:
    fp = f"{BASE}/scripts/{d['id']}.json"
    if os.path.exists(fp):
        with open(fp) as f:
            scripts[d['id']] = json.load(f)

print(f"  加载 {len(scripts)} 个剧本")

# ─────────────────────────────────────────────
print("Step 2: 分词 + 构建语料...")
import jieba
jieba.setLogLevel(60)  # 静默

# 停用词（简易）
STOPWORDS = set('的了是在有我你他她它们这那个一不也都很就与和或但')
STOPWORDS.update(['同白','第一场','第二场','第三场','第四场','第五场',
                  '上场','下场','白','念','唱','内白','引子','摇板',
                  'scripts','xikao','http','com','www','2013','Powered',
                  'TCPDF','tcpdf','play','中国','京剧','戏考','整理',
                  '什么','如何','这里','那里','一个','出来','进来',
                  '起来','过来','过去','下去','上去','说道'])

def tokenize(text):
    # 只保留2字以上的中文词
    words = jieba.cut(text)
    return [w for w in words if len(w) >= 2 and w not in STOPWORDS
            and re.match(r'^[一-鿿]+$', w)]

ids = list(scripts.keys())
corpus_words = []
for sid in ids:
    text = scripts[sid].get('full_text', '')
    corpus_words.append(tokenize(text))

print(f"  分词完成，平均词数: {np.mean([len(w) for w in corpus_words]):.0f}")

# ─────────────────────────────────────────────
print("Step 3: 构建词典，TF-IDF 矩阵...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 把词列表转回字符串给sklearn
corpus_str = [' '.join(words) for words in corpus_words]

tfidf = TfidfVectorizer(max_features=8000, min_df=3, max_df=0.85)
X_tfidf = tfidf.fit_transform(corpus_str)
print(f"  TF-IDF 矩阵: {X_tfidf.shape}")

# ─────────────────────────────────────────────
print("Step 4: 重新跑 LDA (30 topics)...")
lda = LatentDirichletAllocation(
    n_components=30,
    max_iter=30,
    learning_method='online',
    random_state=42,
    n_jobs=-1
)
X_lda = lda.fit_transform(X_tfidf)  # (1473, 30)
print(f"  LDA 完成，shape: {X_lda.shape}")

# 打印主题关键词
feature_names = tfidf.get_feature_names_out()
print("\n  LDA 主题预览:")
for ti in range(30):
    top_words = [feature_names[i] for i in lda.components_[ti].argsort()[-8:][::-1]]
    print(f"    T{ti:02d}: {' '.join(top_words)}")


# ─────────────────────────────────────────────
print("\nStep 5: 角色行当特征 (8维)...")

# 归并到8大类行当
ROLE_MAP = {
    '老生':'laosheng', '生':'laosheng', '正生':'laosheng', '冠生':'laosheng', '巾生':'laosheng',
    '小生':'xiaosheng',
    '旦':'dan', '正旦':'dan', '花旦':'dan', '贴旦':'dan', '小旦':'dan', '五旦':'dan',
    '老旦':'laodan',
    '武旦':'wudan', '刀马旦':'wudan',
    '净':'jing', '副净':'jing', '红净':'jing', '武净':'jing',
    '红生':'hongsheng',
    '丑':'chou', '彩旦':'chou', '丑旦':'chou', '武丑':'chou',
    '武生':'wusheng',
}
ROLE_DIMS = ["laosheng","xiaosheng","dan","laodan","wudan","jing","hongsheng","chou","wusheng"]

def role_features(drama_data):
    counts = {r: 0 for r in ROLE_DIMS}
    chars = drama_data.get('characters', [])
    total = max(len(chars), 1)
    for c in chars:
        rt = c.get('role_type','')
        mapped = ROLE_MAP.get(rt)
        if mapped:
            counts[mapped] += 1
    return [counts[r] / total for r in ROLE_DIMS]

X_role = np.array([role_features(scripts[sid]) for sid in ids])
print(f"  角色特征 shape: {X_role.shape}")

# ─────────────────────────────────────────────
print("Step 6: 叙事特征 (5维)...")

# 查出所有剧本的 drama_info
drama_map = {d['id']: d for d in dramas}

def narrative_features(sid):
    d = drama_map.get(sid, {})
    s = scripts.get(sid, {})
    # 场次数 (归一化到0-1，max~30)
    scenes = min(d.get('scene_count', 1), 30) / 30
    # 页数/篇幅 (max~50)
    pages = min(s.get('pages', d.get('pages', 1)), 50) / 50
    # 角色数 (max~20)
    chars = min(d.get('character_count', 1), 20) / 20
    # 唱腔：西皮比例
    arc = d.get('musical_arc', [])
    total_arc = max(len(arc), 1)
    xipi = sum(1 for x in arc if x == '西皮') / total_arc
    # 二黄比例
    erhuang = sum(1 for x in arc if x == '二黄') / total_arc
    return [scenes, pages, chars, xipi, erhuang]

X_narr = np.array([narrative_features(sid) for sid in ids])
print(f"  叙事特征 shape: {X_narr.shape}")

# ─────────────────────────────────────────────
print("Step 7: 合并特征，加权拼接...")
from sklearn.preprocessing import StandardScaler

# 权重：LDA主题最重要，角色次之，叙事辅助
# 标准化各组特征后加权
scaler_lda = StandardScaler()
scaler_role = StandardScaler()
scaler_narr = StandardScaler()

Xl = scaler_lda.fit_transform(X_lda)   * 1.5   # 30维，权重1.5
Xr = scaler_role.fit_transform(X_role) * 1.2   # 8维，权重1.2
Xn = scaler_narr.fit_transform(X_narr) * 0.8   # 5维，权重0.8

X_combined = np.hstack([Xl, Xr, Xn])
print(f"  合并特征 shape: {X_combined.shape}")

# ─────────────────────────────────────────────
print("Step 8: UMAP 降维...")
import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=20,
    min_dist=0.15,
    metric='cosine',
    random_state=42
)
X_2d = reducer.fit_transform(X_combined)
print(f"  UMAP 完成，shape: {X_2d.shape}")
print(f"  x范围: [{X_2d[:,0].min():.2f}, {X_2d[:,0].max():.2f}]")
print(f"  y范围: [{X_2d[:,1].min():.2f}, {X_2d[:,1].max():.2f}]")


# ─────────────────────────────────────────────
print("\nStep 9: 8类分类规则...")

# 基于角色名关键词字典做半监督分类
# 规则：先用关键词匹配title+characters，未命中的用LDA最强主题投票

CATEGORY_RULES = {
    "历史风云剧": {
        "keywords": [
            # 三国
            "刘备","诸葛亮","关羽","张飞","曹操","孙权","赵云","周瑜","黄忠","马超",
            "司马懿","吕布","貂蝉","孙尚香","孔明","孟德","云长","翼德",
            # 隋唐
            "秦琼","程咬金","尉迟恭","罗成","李世民","李渊","李密","单雄信","瓦岗",
            "秦叔宝","徐茂公",
            # 东汉/刘秀
            "刘秀","姚期","岑彭","马武","邓禹","王莽",
            # 三国赵氏孤儿等春秋战国
            "屠岸贾","程婴","公孙杵臼","韩厥","赵盾","伍员","申包胥","专诸",
            # 其他历史战争
            "花云","徐达","朱元璋","戚继光","班超","苏武",
        ],
        "strong_topics": [19, 28, 29, 5, 18, 0, 22],  # 三国/隋唐/东汉
    },
    "英雄侠义剧": {
        "keywords": [
            # 水浒
            "宋江","李逵","林冲","武松","鲁智深","燕青","时迁","石秀","梁山","晁盖",
            # 岳飞/杨家将
            "岳飞","牛皋","岳云","兀术","韩世忠","王佐","岳鹏举",
            "穆桂英","杨宗保","杨延昭","佘太君","孟良","焦赞","杨继业","杨八姐",
            "杨六郎","杨五郎","柴郡主",
            # 沙陀/李克用
            "李克用","李存孝","史敬思","王彦章","朱温",
            # 其他侠义
            "黄天霸","鲁智深","花和尚",
        ],
        "strong_topics": [26, 7, 3, 6],
    },
    "宫廷权谋剧": {
        "keywords": [
            "包拯","展昭","公孙策","王朝","马汉","张龙","赵虎",  # 包拯但权谋向
            "严嵩","邹应龙","嘉靖","海瑞","刘瑾","贾桂","鸣凤",
            "太监","皇上","万岁","娘娘","国母","李艳妃","徐延昭",
            "二进宫","万历","徐彦昭","杨波","李良",
            "唐明皇","杨贵妃","高力士","安禄山","玄宗",
            "武则天","上官婉儿","赵匡胤","赵普","柴荣","高怀德",
        ],
        "strong_topics": [21, 23, 16, 1, 25],
    },
    "公案清官剧": {
        "keywords": [
            "包拯","秦香莲","陈世美","韩琦","颜查散","马蓉",
            "苏三","王金龙","崇公道","况钟",
            "衙役","知县","县令","大老爷","师爷","书吏",
            "冤枉","喊冤","状纸","判案","断案","升堂",
        ],
        "strong_topics": [1, 27, 4, 25],
    },
    "爱情婚姻剧": {
        "keywords": [
            "王宝钏","薛平贵","薛仁贵","柳迎春","寒窑",
            "红娘","崔莺莺","张珙","杜丽娘","柳梦梅","春香",
            "杜十娘","李甲","荀灌娘","貂蝉",
            "王昭君","文成公主",
            "梁山伯","祝英台","白蛇","许仙",
            "秦香莲",  # 这里也有婚姻线
        ],
        "strong_topics": [15, 20, 8],
    },
    "神话神魔剧": {
        "keywords": [
            "孙悟空","猪八戒","唐僧","沙僧","唐三藏","如来","观音",
            "姜子牙","哪吒","杨戬","雷震子","金吒","木吒",
            "沉香","刘彦昌","华岳三娘","二郎神",
            "嫦娥","月宫","天官","玉帝","王母","蟠桃",
        ],
        "strong_topics": [12, 11, 20],
    },
    "家庭伦理剧": {
        "keywords": [
            "媳妇","婆婆","公婆","养母","继母","后母",
            "孝子","孝女","养子","养女","孤儿",
            "王春娥","薛保","孙玉姣",
        ],
        "strong_topics": [8, 13, 27, 2],
    },
    "生活风俗剧": {
        "keywords": [
            "店家","酒保","店小二","车夫","渔翁","樵夫",
            "庄稼","乡民","村民","老翁","老妪",
            "卖油郎","花鼓","花灯","打鱼","卖水",
        ],
        "strong_topics": [13, 14, 2, 17],
    },
}

# LDA主题到大类的映射（作为fallback）
TOPIC_TO_CATEGORY = {
    0: "历史风云剧", 3: "英雄侠义剧", 5: "历史风云剧", 6: "英雄侠义剧",
    7: "英雄侠义剧", 10: "历史风云剧", 11: "神话神魔剧", 12: "神话神魔剧",
    15: "爱情婚姻剧", 16: "宫廷权谋剧", 18: "历史风云剧", 19: "历史风云剧",
    20: "爱情婚姻剧", 21: "宫廷权谋剧", 22: "宫廷权谋剧", 23: "宫廷权谋剧",
    24: "生活风俗剧", 25: "宫廷权谋剧", 26: "英雄侠义剧", 27: "公案清官剧",
    28: "历史风云剧", 29: "历史风云剧", 1: "公案清官剧", 4: "家庭伦理剧",
    8: "爱情婚姻剧", 9: "英雄侠义剧", 13: "家庭伦理剧", 14: "生活风俗剧",
    17: "生活风俗剧",
}

def classify_drama(sid, lda_vec):
    s = scripts.get(sid, {})
    title = s.get('title', '')
    chars = s.get('characters', [])
    char_names = set(c.get('name', '') for c in chars)
    full_text = s.get('full_text', '')[:500]  # 只看前500字判断

    # 提取文本中的人名（简单：查关键词）
    text_to_check = title + ' ' + ' '.join(char_names) + ' ' + full_text

    scores = {cat: 0.0 for cat in CATEGORY_RULES}

    # 关键词匹配
    for cat, rules in CATEGORY_RULES.items():
        for kw in rules['keywords']:
            if kw in text_to_check:
                scores[cat] += 1.5

    # LDA主题投票（top3主题）
    top_topics = np.argsort(lda_vec)[-3:][::-1]
    for ti in top_topics:
        cat = TOPIC_TO_CATEGORY.get(int(ti))
        if cat:
            scores[cat] += lda_vec[ti] * 3.0

    # 强主题加权
    for cat, rules in CATEGORY_RULES.items():
        for ti in rules.get('strong_topics', []):
            scores[cat] += lda_vec[ti] * 2.0

    best = max(scores, key=scores.get)
    return best

categories = [classify_drama(sid, X_lda[i]) for i, sid in enumerate(ids)]
from collections import Counter
cat_dist = Counter(categories)
print("  8类分布:")
for cat, cnt in sorted(cat_dist.items(), key=lambda x:-x[1]):
    print(f"    {cat}: {cnt}")


# ─────────────────────────────────────────────
print("\nStep 10: 修正分类（以原drama_type为先验）...")

# 原类型→新8类基础映射
BASE_MAP = {
    "爱情婚姻": "爱情婚姻剧",
    "家庭伦理": "家庭伦理剧",
    "历史宫廷": "宫廷权谋剧",
    "冤案公案": "公案清官剧",
    "神话传说": "神话神魔剧",
    "其他":     "生活风俗剧",
}
# 历史战争 → 历史风云剧 or 英雄侠义剧（需拆分）
# 忠义智谋 → 历史风云剧 or 英雄侠义剧（需拆分）

HERO_KEYWORDS = set([
    # 水浒
    "宋江","李逵","林冲","武松","鲁智深","燕青","时迁","石秀","梁山","晁盖","鲁达",
    # 岳飞/杨家将
    "岳飞","牛皋","岳云","兀术","韩世忠","王佐",
    "穆桂英","杨宗保","杨延昭","佘太君","孟良","焦赞","杨继业","杨八姐","杨六郎","柴郡主",
    # 沙陀/李克用
    "李克用","李存孝","史敬思","王彦章","朱温",
    # 侠义绿林
    "黄天霸","施公","豪杰",
])

def refined_classify(sid, orig_type, lda_vec):
    s = scripts.get(sid, {})
    chars = s.get('characters', [])
    char_names = set(c.get('name', '') for c in chars)
    title = s.get('title', '')
    check = title + ' ' + ' '.join(char_names)

    # 先用base_map处理已知类
    if orig_type in BASE_MAP:
        return BASE_MAP[orig_type]

    # 历史战争 & 忠义智谋 → 按关键词拆
    if orig_type in ("历史战争", "忠义智谋"):
        hero_score = sum(1 for kw in HERO_KEYWORDS if kw in check)
        if hero_score >= 1:
            return "英雄侠义剧"
        # 用LDA辅助：岳飞/水浒topic
        if lda_vec[26] + lda_vec[3] + lda_vec[7] + lda_vec[6] > lda_vec[19] + lda_vec[28]:
            return "英雄侠义剧"
        return "历史风云剧"

    return "历史风云剧"  # fallback

categories_v2 = []
for i, sid in enumerate(ids):
    orig = drama_map.get(sid, {}).get('drama_type', '其他')
    cat = refined_classify(sid, orig, X_lda[i])
    categories_v2.append(cat)

cat_dist2 = Counter(categories_v2)
print("  修正后8类分布:")
for cat, cnt in sorted(cat_dist2.items(), key=lambda x:-x[1]):
    print(f"    {cat}: {cnt}")


# ─────────────────────────────────────────────
print("\nStep 11: 二级系列分组...")

# 每个一级类下的系列，用LDA主题权重+关键词识别
SERIES_RULES = {
    "历史风云剧": [
        ("三国群",   [19,28,29], ["刘备","诸葛亮","曹操","关羽","张飞","周瑜","孙权","赵云","司马懿","孔明","吕布"]),
        ("隋唐群",   [5],  ["秦琼","程咬金","尉迟恭","罗成","李世民","李渊","单雄信","徐茂公","瓦岗"]),
        ("东汉群",   [18], ["刘秀","姚期","岑彭","马武","王莽"]),
        ("赵氏孤儿群",[0], ["屠岸贾","程婴","公孙杵臼","韩厥"]),
        ("春秋战国群",[],  ["伍员","申包胥","专诸","勾践","范蠡","孙膑","庞涓","苏秦","张仪"]),
    ],
    "英雄侠义剧": [
        ("水浒群",   [3],  ["宋江","李逵","林冲","武松","鲁智深","燕青","梁山","晁盖"]),
        ("岳飞群",   [26], ["岳飞","牛皋","岳云","兀术","韩世忠","疯僧"]),
        ("杨家将群", [7],  ["穆桂英","杨宗保","杨延昭","佘太君","孟良","焦赞","杨继业","杨八姐","杨六郎"]),
        ("沙陀群",   [6],  ["李克用","李存孝","史敬思","王彦章","朱温"]),
    ],
    "宫廷权谋剧": [
        ("唐宫群",   [23], ["唐明皇","杨贵妃","高力士","安禄山","玄宗"]),
        ("明廷群",   [21,25], ["严嵩","邹应龙","嘉靖","海瑞","刘瑾","贾桂","鸣凤","徐延昭","李艳妃","二进宫","万历"]),
        ("宋廷群",   [16], ["赵匡胤","赵普","柴荣","高怀德","潘仁美"]),
        ("三国宫廷", [19,28], ["曹操","孙权","董卓","王允"]),
    ],
    "公案清官剧": [
        ("包公群",   [1],  ["包拯","展昭","公孙策","王朝","马汉","张龙","赵虎"]),
        ("苏三群",   [27], ["苏三","王金龙","崇公道"]),
        ("秦香莲群", [],   ["秦香莲","陈世美"]),
    ],
    "爱情婚姻剧": [
        ("西厢记群", [15], ["红娘","崔莺莺","张珙","张生"]),
        ("牡丹亭群", [15], ["杜丽娘","柳梦梅","春香"]),
        ("薛平贵群", [20], ["王宝钏","薛平贵","寒窑"]),
        ("其他爱情", [],   []),
    ],
    "神话神魔剧": [
        ("西游群",   [12], ["孙悟空","猪八戒","唐僧","沙僧","如来","观音"]),
        ("封神群",   [11], ["姜子牙","哪吒","杨戬","雷震子","金吒","木吒"]),
        ("其他神话", [],   ["沉香","刘彦昌","嫦娥","玉帝"]),
    ],
    "家庭伦理剧": [
        ("教子劝善", [13], ["薛保","王春娥","孙玉姣"]),
        ("其他伦理", [],   []),
    ],
    "生活风俗剧": [
        ("市井百态", [14,17], ["店家","酒保","车夫","樵夫"]),
    ],
}

def get_series(sid, cat, lda_vec):
    rules = SERIES_RULES.get(cat, [])
    s = scripts.get(sid, {})
    chars = s.get('characters', [])
    char_names = set(c.get('name', '') for c in chars)
    title = s.get('title', '')
    check = title + ' ' + ' '.join(char_names)

    best_series = None
    best_score = 0

    for series_name, topic_ids, keywords in rules:
        if series_name.startswith("其他"):
            continue
        score = 0
        for kw in keywords:
            if kw in check:
                score += 2
        for ti in topic_ids:
            score += lda_vec[ti] * 3
        if score > best_score:
            best_score = score
            best_series = series_name

    # 如果没有匹配，归到"其他+类名"
    if best_series is None or best_score < 0.5:
        # 找该类的fallback系列
        for series_name, _, _ in rules:
            if series_name.startswith("其他"):
                return series_name
        return f"其他{cat[:2]}"

    return best_series

series_list = []
for i, sid in enumerate(ids):
    cat = categories_v2[i]
    series = get_series(sid, cat, X_lda[i])
    series_list.append(series)

# 打印系列分布
series_dist = Counter(series_list)
print("  二级系列分布（top20）:")
for s, c in series_dist.most_common(20):
    print(f"    {s}: {c}")


# ─────────────────────────────────────────────
print("\nStep 12: 输出 JSON...")

# 归一化坐标到 [-1, 1]
x_arr = X_2d[:, 0]
y_arr = X_2d[:, 1]
x_norm = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min()) * 2 - 1
y_norm = (y_arr - y_arr.min()) / (y_arr.max() - y_arr.min()) * 2 - 1

# 构建输出
points = []
for i, sid in enumerate(ids):
    s = scripts.get(sid, {})
    d = drama_map.get(sid, {})
    points.append({
        "id": sid,
        "title": s.get('title', d.get('title', '')),
        "x": round(float(x_norm[i]), 4),
        "y": round(float(y_norm[i]), 4),
        "category": categories_v2[i],
        "series": series_list[i],
        "orig_type": d.get('drama_type', ''),
        "scene_count": d.get('scene_count', 0),
        "character_count": d.get('character_count', 0),
        "pages": s.get('pages', 0),
        # top3 LDA topics
        "top_topics": [int(t) for t in np.argsort(X_lda[i])[-3:][::-1]],
    })

# LDA主题元数据（重新跑的）
topic_meta = []
for ti in range(30):
    top_words = [feature_names[j] for j in lda.components_[ti].argsort()[-10:][::-1]]
    topic_meta.append({
        "topic_id": ti,
        "keywords": top_words
    })

output = {
    "total": len(points),
    "categories": sorted(set(categories_v2)),
    "category_counts": dict(cat_dist2),
    "series_counts": dict(series_dist),
    "topic_meta": topic_meta,
    "points": points
}

out_path = f"{OUT_DIR}/drama_embeddings.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"  输出完成：{out_path}")
print(f"  总点数: {len(points)}")
print("Done!")

