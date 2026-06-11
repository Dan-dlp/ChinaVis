#!/usr/bin/env python3
"""
京剧剧本降维聚类 Pipeline V6
======================================
V6：监督 UMAP + 布局后处理（hull_layout_postprocess 模块）。
先用监督 UMAP 保持各系列内部紧凑，
再做系列压缩 + 半径感知的包络圆碰撞分离，
保证不同系列的包络圆 0 重叠、0 套叠。

参数：
  - Supervised UMAP: target_weight=0.75, min_dist=0.08, n_neighbors=20
  - 后处理参数见 hull_layout_postprocess.py
  - 不画包络圆："其他XX" + 市井百态（纯散点）

输入：structured/scripts/*.json + frontend public/data/index.json
输出：frontend/app/public/data/filter/drama_embeddings.json (直接覆盖)
"""

import json, os, re, pickle, warnings, math
import numpy as np
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')

import jieba
jieba.setLogLevel(60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, LabelEncoder
import umap

# ── 路径配置 ──────────────────────────────────────────────
BASE = '/Users/dan/Desktop/VISeer/ChinaVIS/structured'
FRONTEND_DATA = '/Users/dan/Desktop/VISeer/ChinaVIS/frontend/app/public/data'
OUT  = '/Users/dan/Desktop/VISeer/ChinaVIS/structured/Script Dimension Reduction'
os.makedirs(OUT, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────
with open(f'{FRONTEND_DATA}/index.json') as f:
    idx = json.load(f)
drama_map = {d['id']: d for d in idx['dramas']}
scripts = {}
for d in idx['dramas']:
    fp = f"{BASE}/scripts/{d['id']}.json"
    if os.path.exists(fp):
        with open(fp) as f2:
            scripts[d['id']] = json.load(f2)
ids = list(scripts.keys())
print(f"加载 {len(ids)} 个剧本")

# ── 一级分类字典 ──────────────────────────────────────────
CHAR_RULES = {
    "神话神魔剧": [
        "孙悟空","猪八戒","唐僧","沙僧","唐三藏","玉龙太子","弼马温",
        "金角大王","银角大王","九灵元圣","金钱豹","牛魔王","铁扇公主",
        "姜子牙","哪吒","杨戬","雷震子","金吒","木吒","土行孙","妲己",
        "闻太师","黄飞虎","申公豹","广成子","通天教主",
        "沉香","华岳三娘","二郎神","法海",
        "嫦娥","玉帝","王母","太白金星","麻姑","铁拐李","吕洞宾","张果老",
        "何仙姑","钟离权","韩湘子","曹国舅","蓝采和","东方朔","南极老人",
        "天官","福星","禄星","寿星","多福星君","多禄星君","多寿星君","天女",
        "目莲","刘清提","水母娘娘","九尾仙姬","妙善",
    ],
    "公案清官剧": [
        "包拯","展昭","公孙策","王朝","马汉","张龙","赵虎","颜查散",
        "苏三","崇公道","王金龙",
        "施世纶","施公","黄天霸","贺天保","卢志义","常保",
        "窦娥","张驴儿","蔡母","秦香莲","陈世美",
        "况钟","白玉堂","欧阳春","丁兆蕙","卢方",
    ],
    "英雄侠义剧": [
        "宋江","李逵","林冲","武松","鲁智深","燕青","时迁","石秀","晁盖",
        "花荣","阮小七","阮小五","阮小二","呼延灼","卢俊义","吴用","柴进","秦明","史进",
        "岳飞","牛皋","岳云","韩世忠","王佐","疯僧","岳鹏举","高宠",
        "穆桂英","杨宗保","杨延昭","佘太君","孟良","焦赞","杨继业",
        "杨八姐","杨六郎","杨五郎","杨四郎","柴郡主","杨排风","杨延辉","佘赛花",
        "李克用","李存孝","史敬思","王彦章","朱温","李嗣源","安敬思",
        "施必显","施碧霞","李春芳","华月娥","华子林",
    ],
    "历史风云剧": [
        "刘备","曹操","关羽","张飞","诸葛亮","孙权","周瑜","赵云","黄忠",
        "马超","司马懿","吕布","孙尚香","庞统","徐庶","鲁肃","张辽","夏侯惇",
        "甘宁","太史慈","陆逊","姜维","邓艾","孔明","孟德","翼德","云长",
        "貂蝉","董卓","王允","马谡","王平","魏延",
        "秦琼","程咬金","尉迟恭","罗成","李渊","单雄信","徐茂公","秦叔宝",
        "薛仁贵","薛丁山","薛刚","薛猛","樊梨花","张士贵","李元霸","裴元庆",
        "宇文化及","杨广","隋炀帝",
        "刘秀","姚期","岑彭","马武","邓禹","吴汉","杜茂","姚刚","郭荣","刘忠","王霸",
        "屠岸贾","程婴","公孙杵臼","韩厥","赵盾",
        "伍员","伍子胥","申包胥","专诸","勾践","范蠡","孙膑","庞涓",
        "廉颇","蔺相如","荆轲","燕太子丹","项羽","韩信","范增","虞姬","张良",
        "屈原","伍奢","夫差","西施","苏秦","张仪",
        "姬昌","姬发","武王","纣王","苏护","苏全忠",
        "楚庄王","唐狡","晋文公","介子推","伯嚭","文种","郑庄公","共叔段","武姜",
        "刘邦","项梁","蒯彻","萧何","曹参","陈平","张苍",
        "周幽王","褒姒","申侯",
        "卫青","霍去病","李广","班超","苏武",
        "戚继光","花云","徐达","常遇春","袁崇焕","崇祯","郑成功","史可法","李自成","吴三桂",
        "赵匡胤","柴荣","赵普","高怀德","郑子明","石守信","曹彬","潘美",
        "赵构","宋钦宗","宋徽宗","完颜宗弼","兀术",
        "祢衡","刘表","黄祖","管宁","华歆",
        "徐文炳","徐文亮",
    ],
    "宫廷权谋剧": [
        "严嵩","邹应龙","嘉靖","刘瑾","贾桂","鸣凤","徐延昭","李艳妃",
        "万历","正德","李良","杨波","徐彦昭","王振","严世藩",
        "唐明皇","杨贵妃","高力士","安禄山","虢国夫人","上官婉儿","李隆基",
        "武则天","吕雉","吕产","吕禄","霍光",
        "潘仁美","潘洪","赵光义","贺后","宋仁宗","庞吉",
        "乾隆","和珅","纪晓岚","皇太极","李世民",
    ],
    "爱情婚姻剧": [
        "红娘","崔莺莺","张珙","杜丽娘","柳梦梅","春香","杜宝",
        "白素贞","许仙","许汉文","小青",
        "梁山伯","祝英台","王宝钏","薛平贵","寒窑","魏虎",
        "王昭君","呼韩邪","杜十娘","李甲","陈妙常","潘必正","赵五娘","蔡伯喈",
        "文成公主","禄东赞","苏小妹","秦观","荀灌娘","陈三两",
        "李慧娘","裴稚卿","陈杏元","梅良玉",
        "方卿","陈翠娥","田玉川","胡凤莲","郦珊柯","卫玉",
        "何玉凤","慕金龙","金桂英","骆宏勋",
    ],
    "家庭伦理剧": [
        "王春娥","薛保","薛子奇","薛倚","孙玉姣","王金友",
        "秋胡","罗敷","罗氏","汤勤","赵艳容","哑奴",
        "李三娘","刘智远","吴汉","王兰英","闵父","闵母","闵觉","闵远","孟母",
        "贾宝玉","林黛玉","薛宝钗","王熙凤","贾母","贾政","晴雯","紫鹃",
        "袭人","鸳鸯","平儿","迎春","探春","惜春","秦可卿","刘姥姥",
        "贾琏","贾珍","贾环","秦钟","智能",
    ],
    "其他剧本": [],
}

KW_RULES = {
    "神话神魔剧": [
        "取经","花果山","天宫","蟠桃","仙界","天庭","阴曹","地府","阎王","冥府",
        "修炼成仙","得道成仙","炼丹","法宝","神器","宝莲","莲花宝座",
        "妖怪","除妖","降妖","伏魔","斩妖","封神","封神榜","八卦炉",
    ],
    "公案清官剧": [
        "升堂","问斩","铡","冤枉","喊冤","状纸","状告","断案","审案",
        "冤狱","冤案","平反","昭雪","知府","知县","县令","刑部",
        "砍头","斩首","发配","充军","流放","破案",
    ],
    "英雄侠义剧": [
        "梁山","绿林","好汉","豪杰","江湖","占山为王","聚义","劫富济贫",
        "武林","比武招亲","擂台","侠义",
    ],
    "历史风云剧": [
        "兵马","出征","征讨","交战","厮杀","攻城","守城","大军",
        "将帅","帅印","点兵","三军","军师","行军","围城","破阵",
        "兵权","率兵","发兵","两军对阵","战役",
    ],
    "宫廷权谋剧": [
        "朝廷","朝政","奸臣","宦官","弄权","党争","密谋","宫廷","宫闱",
        "皇位","夺位","篡位","御旨","圣旨","龙颜","谋反","诛杀忠良",
    ],
    "爱情婚姻剧": [
        "相思","私奔","成亲","洞房","花轿","订婚","定情","私定终身",
        "情郎","情缘","爱慕","倾心","钟情","一见钟情","抛彩球","招亲",
        "儿女情长","鸳鸯","才子佳人","私会","幽会",
    ],
    "家庭伦理剧": [
        "婆婆","媳妇","公婆","孝子","孝女","养子","养母","继母","后母",
        "悍妇","妒妇","妒忌","养育","教子","劝善","孝道","骨肉",
        "妻离子散","家道中落","弃妇","弃子","认亲","寻亲","大观园","荣国府","贾府",
    ],
    "其他剧本": [
        "打鱼","卖水","卖油","卖花","店小二","酒保","车夫","樵夫",
        "渔翁","农夫","庄稼","市井","民间","乡村","百姓","小民",
    ],
}

PRIORITY = {
    "神话神魔剧":8,"公案清官剧":7,"英雄侠义剧":6,"历史风云剧":5,
    "宫廷权谋剧":4,"爱情婚姻剧":3,"家庭伦理剧":2,"其他剧本":1,
}

def classify(sid):
    s = scripts.get(sid, {})
    char_names = set(c.get('name','').strip() for c in s.get('characters',[])) - {'龙套',''}
    text = s.get('title','') + ' ' + s.get('plot_summary','')
    scores = {cat: 0.0 for cat in CHAR_RULES}
    for cat, names in CHAR_RULES.items():
        for name in names:
            if name in char_names: scores[cat] += 3
    for cat, kws in KW_RULES.items():
        for kw in kws:
            if kw in text: scores[cat] += 1
    best = max(scores.values())
    if best == 0: return "其他剧本"
    return max([c for c,v in scores.items() if v==best], key=lambda c:PRIORITY[c])

categories = [classify(sid) for sid in ids]
print("一级分类完成：", dict(Counter(categories)))

# ── 二级系列分组 ──────────────────────────────────────────
SERIES_RULES = {
    "历史风云剧":[
        ("三国群",    ["刘备","曹操","关羽","张飞","诸葛亮","孙权","周瑜","赵云","司马懿","吕布","孔明"]),
        ("隋唐群",    ["秦琼","程咬金","尉迟恭","罗成","李渊","单雄信","徐茂公","薛仁贵","薛丁山","薛刚"]),
        ("东汉群",    ["刘秀","姚期","岑彭","马武","邓禹","吴汉"]),
        ("赵氏孤儿群",["屠岸贾","程婴","公孙杵臼","韩厥","赵盾"]),
        ("春秋战国群",["项羽","韩信","虞姬","范增","张良","荆轲","廉颇","蔺相如","伍员","伍子胥","勾践","范蠡","孙膑","庞涓","楚庄王","晋文公"]),
    ],
    "英雄侠义剧":[
        ("水浒群",    ["宋江","李逵","林冲","武松","鲁智深","燕青","梁山","晁盖","花荣","阮小七"]),
        ("岳飞群",    ["岳飞","牛皋","岳云","韩世忠","王佐","疯僧","高宠"]),
        ("杨家将群",  ["穆桂英","杨宗保","杨延昭","佘太君","孟良","焦赞","杨继业","杨六郎","杨八姐","佘赛花"]),
        ("沙陀群",    ["李克用","李存孝","史敬思","王彦章","朱温","安敬思"]),
        ("天宝图群",  ["施必显","施碧霞","李春芳","华月娥","华子林"]),
    ],
    "宫廷权谋剧":[
        ("明廷群",    ["严嵩","邹应龙","嘉靖","刘瑾","贾桂","鸣凤","徐延昭","李艳妃","万历"]),
        ("唐宫群",    ["唐明皇","杨贵妃","高力士","安禄山","李隆基","虢国夫人"]),
        ("汉宫群",    ["吕雉","吕产","吕禄","霍光","张苍","陈平"]),
        ("宋廷群",    ["潘仁美","赵光义","贺后","宋仁宗","庞吉"]),
    ],
    "公案清官剧":[
        ("包公群",    ["包拯","展昭","公孙策","王朝","马汉","张龙","赵虎"]),
        ("施公群",    ["施世纶","施公","黄天霸","贺天保","卢志义"]),
        ("苏三群",    ["苏三","王金龙","崇公道"]),
        ("秦香莲群",  ["秦香莲","陈世美"]),
    ],
    "爱情婚姻剧":[
        ("西厢牡丹群",["红娘","崔莺莺","张珙","杜丽娘","柳梦梅","春香"]),
        ("薛平贵群",  ["王宝钏","薛平贵","寒窑","魏虎"]),
        ("白蛇群",    ["白素贞","许仙","小青","法海"]),
        ("梁祝群",    ["梁山伯","祝英台"]),
    ],
    "神话神魔剧":[
        ("西游群",    ["孙悟空","猪八戒","唐僧","沙僧","如来","观音","唐三藏"]),
        ("封神群",    ["姜子牙","哪吒","杨戬","雷震子","金吒","木吒","土行孙","妲己","闻太师","黄飞虎"]),
        ("劈山群",    ["沉香","华岳三娘","二郎神"]),
        ("祝寿群",    ["麻姑","铁拐李","吕洞宾","天官","福星","禄星","寿星","多福星君"]),
    ],
    "家庭伦理剧":[
        ("红楼群",    ["贾宝玉","林黛玉","薛宝钗","王熙凤","贾母","贾政","晴雯","紫鹃","袭人","贾琏"]),
        ("教子群",    ["王春娥","薛保","孙玉姣","孟母","薛子奇"]),
        ("其他伦理",  []),
    ],
    "其他剧本":[
        ("市井百态",  []),
    ],
}

def get_series(sid, cat):
    rules = SERIES_RULES.get(cat, [])
    s = scripts.get(sid, {})
    cn = set(c.get('name','') for c in s.get('characters',[]))
    best, best_score = None, 0
    for sname, kws in rules:
        if sname in ("其他伦理","市井百态") or sname.startswith("其他"): continue
        score = sum(2 for k in kws if k in cn)
        if score > best_score: best_score, best = score, sname
    if best is None or best_score == 0:
        for sname, _ in rules:
            if sname in ("其他伦理","市井百态") or sname.startswith("其他"): return sname
        return f"其他{cat[:2]}"
    return best

series_list = [get_series(ids[i], categories[i]) for i in range(len(ids))]

# ── 特征工程 ──────────────────────────────────────────────

# 1. LDA（清洗URL）
STOPWORDS = set('的了是在有我你他她它们这那个一不也都很就与和或但')
STOPWORDS.update(['同白','第一场','第二场','第三场','第四场','第五场',
                  'scripts','xikao','http','com','www','2013','Powered','TCPDF','tcpdf','play','中国','京剧','戏考'])

def clean_tok(text):
    lines = [l for l in text.split('\n') if 'http' not in l and 'TCPDF' not in l]
    return [w for w in jieba.cut('\n'.join(lines))
            if len(w)>=2 and w not in STOPWORDS and re.match(r'^[一-鿿]+$',w)]

corpus_str = [' '.join(clean_tok(scripts[s].get('full_text',''))) for s in ids]
tfidf = TfidfVectorizer(max_features=8000, min_df=3, max_df=0.85)
X_tfidf = tfidf.fit_transform(corpus_str)
lda = LatentDirichletAllocation(n_components=30, max_iter=30,
                                 learning_method='online', random_state=42, n_jobs=-1)
X_lda = lda.fit_transform(X_tfidf)
print(f"LDA done {X_lda.shape}")

# 2. 角色行当
ROLE_MAP = {'老生':'ls','生':'ls','正生':'ls','冠生':'ls','巾生':'ls','小生':'xs',
            '旦':'d','正旦':'d','花旦':'d','贴旦':'d','小旦':'d','五旦':'d',
            '老旦':'ld','武旦':'wd','净':'j','副净':'j','红净':'j','武净':'j',
            '红生':'hs','丑':'c','彩旦':'c','丑旦':'c','武丑':'c','武生':'ws'}
ROLE_DIMS = ['ls','xs','d','ld','wd','j','hs','c','ws']
def role_feat(sid):
    chars=scripts[sid].get('characters',[]); t=max(len(chars),1)
    cnt={r:0 for r in ROLE_DIMS}
    for ch in chars:
        m=ROLE_MAP.get(ch.get('role_type',''))
        if m: cnt[m]+=1
    return [cnt[r]/t for r in ROLE_DIMS]
X_role = np.array([role_feat(s) for s in ids])

# 3. 叙事特征
def narr_feat(sid):
    d=drama_map.get(sid,{}); s=scripts.get(sid,{})
    arc=d.get('musical_arc',[]); t=max(len(arc),1)
    return [min(d.get('scene_count',1),30)/30, min(s.get('pages',1),50)/50,
            min(d.get('character_count',1),20)/20,
            sum(1 for x in arc if x=='西皮')/t,
            sum(1 for x in arc if x=='二黄')/t]
X_narr = np.array([narr_feat(s) for s in ids])

# 4. 角色名二值特征（系列分离核心信号）
SERIES_CHARS = {
    "三国":     ["刘备","曹操","关羽","张飞","诸葛亮","孙权","周瑜","赵云","司马懿","吕布","黄忠","马超","貂蝉","董卓"],
    "隋唐":     ["秦琼","程咬金","尉迟恭","罗成","李渊","单雄信","徐茂公","薛仁贵","薛丁山","薛刚","李元霸","裴元庆"],
    "东汉":     ["刘秀","姚期","岑彭","马武","邓禹","吴汉","杜茂","姚刚"],
    "赵氏孤儿": ["屠岸贾","程婴","公孙杵臼","韩厥","赵盾"],
    "春秋战国": ["项羽","韩信","虞姬","范增","张良","荆轲","廉颇","蔺相如","伍员","伍子胥","勾践","范蠡","孙膑","庞涓","苏秦","张仪","楚庄王","晋文公","介子推","周幽王","褒姒","姬昌","姬发"],
    "明清历史": ["崇祯","袁崇焕","戚继光","郑成功","史可法","李自成","吴三桂","皇太极","乾隆","和珅"],
    "水浒":     ["宋江","李逵","林冲","武松","鲁智深","燕青","时迁","晁盖","花荣","阮小七"],
    "岳飞":     ["岳飞","牛皋","岳云","韩世忠","王佐","疯僧","高宠"],
    "杨家将":   ["穆桂英","杨宗保","杨延昭","佘太君","孟良","焦赞","杨继业","杨六郎","杨八姐","佘赛花"],
    "沙陀":     ["李克用","李存孝","史敬思","王彦章","朱温","安敬思"],
    "明廷":     ["严嵩","邹应龙","嘉靖","刘瑾","贾桂","鸣凤","徐延昭","李艳妃","万历"],
    "唐宫":     ["唐明皇","杨贵妃","高力士","安禄山","李隆基"],
    "汉宫":     ["吕雉","霍光","张苍","陈平","吕产","吕禄"],
    "宋廷":     ["潘仁美","赵光义","贺后","宋仁宗","庞吉"],
    "包公":     ["包拯","展昭","公孙策","王朝","马汉","张龙","赵虎"],
    "施公":     ["施世纶","施公","黄天霸","贺天保","卢志义"],
    "苏三":     ["苏三","崇公道","王金龙"],
    "西厢牡丹": ["红娘","崔莺莺","张珙","杜丽娘","柳梦梅","春香"],
    "薛平贵":   ["王宝钏","薛平贵","寒窑","魏虎"],
    "白蛇":     ["白素贞","许仙","小青","法海"],
    "梁祝":     ["梁山伯","祝英台"],
    "西游":     ["孙悟空","猪八戒","唐僧","沙僧","如来","观音"],
    "封神":     ["姜子牙","哪吒","杨戬","雷震子","金吒","木吒","妲己","闻太师","黄飞虎"],
    "劈山":     ["沉香","二郎神","华岳三娘"],
    "祝寿":     ["麻姑","铁拐李","吕洞宾","天官","福星","禄星","寿星","多福星君"],
    "红楼":     ["贾宝玉","林黛玉","薛宝钗","王熙凤","贾母","贾政","晴雯","紫鹃","袭人","贾琏"],
    "教子":     ["王春娥","薛保","孙玉姣","孟母","薛子奇"],
}

char_vocab = {}
for chars in SERIES_CHARS.values():
    for name in chars:
        if name not in char_vocab:
            char_vocab[name] = len(char_vocab)

def char_features(sid):
    chars = set(c.get('name','') for c in scripts[sid].get('characters',[]))
    vec = np.zeros(len(char_vocab))
    for name, idx in char_vocab.items():
        if name in chars: vec[idx] = 1.0
    return vec

X_char = np.array([char_features(sid) for sid in ids])
print(f"角色名特征 {X_char.shape}")

# ── 5. 合并特征（V5）────────────────────────────────────
Xl = StandardScaler().fit_transform(X_lda)  * 1.5
Xr = StandardScaler().fit_transform(X_role) * 1.0
Xn = StandardScaler().fit_transform(X_narr) * 0.8
Xc = X_char * 6.0
X_combined = np.hstack([Xl, Xr, Xn, Xc])
print(f"合并特征 {X_combined.shape}")
print(f"  权重: LDA×1.5  Role×1.0  Narr×0.8  Char×6.0")

# ── V5：监督 UMAP + hull 排斥后处理 ──────────────────────
le = LabelEncoder()
y = le.fit_transform(series_list)
print("Supervised UMAP (V5: target_weight=0.75, min_dist=0.08, n_neighbors=20)...")
reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.08,
                    metric='cosine', random_state=42,
                    target_metric='categorical', target_weight=0.75)
X_2d = reducer.fit_transform(X_combined, y=y)

x_n = (X_2d[:,0]-X_2d[:,0].min())/(X_2d[:,0].max()-X_2d[:,0].min())*2-1
y_n = (X_2d[:,1]-X_2d[:,1].min())/(X_2d[:,1].max()-X_2d[:,1].min())*2-1

# ── V6 布局后处理：系列压缩 + 包络圆碰撞分离 ─────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hull_layout_postprocess import layout_postprocess, SKIP_HULL_SERIES

points = []
for i,sid in enumerate(ids):
    s=scripts.get(sid,{}); d=drama_map.get(sid,{})
    points.append({"id":sid,"title":s.get('title',d.get('title','')),"x":float(x_n[i]),
                   "y":float(y_n[i]),"category":categories[i],"series":series_list[i],
                   "scene_count":d.get('scene_count',0),"character_count":d.get('character_count',0),
                   "pages":s.get('pages',0)})

series_hull = layout_postprocess(points)

SERIES_CAT = {}
for i,sid in enumerate(ids):
    SERIES_CAT[series_list[i]] = categories[i]
skipped_hulls = sorted(SKIP_HULL_SERIES & set(series_list))

# ── 输出 JSON ─────────────────────────────────────────────

output = {"total":len(points),"categories":sorted(set(categories)),
          "category_counts":dict(Counter(categories)),
          "series_hull":series_hull,"series_cat":SERIES_CAT,"points":points}

out_path = f'{FRONTEND_DATA}/filter/drama_embeddings.json'
with open(out_path,'w',encoding='utf-8') as f:
    json.dump(output,f,ensure_ascii=False,indent=2)

print(f"\n✓ 输出: {out_path}")
print(f"  总点数: {len(points)}，包络圆: {len(series_hull)} 个")
print(f"  跳过包络圆: {skipped_hulls}")

# ── 质量报告 ──────────────────────────────────────────────
print("\n=== 空间分布报告 ===")
xs = [p['x'] for p in points]
ys = [p['y'] for p in points]
print(f"  x: [{min(xs):.3f}, {max(xs):.3f}]")
print(f"  y: [{min(ys):.3f}, {max(ys):.3f}]")

# 中心区域密度
center_count = sum(1 for p in points if abs(p['x']) < 0.3 and abs(p['y']) < 0.3)
print(f"  中心区域 (|x|<0.3, |y|<0.3): {center_count}/{len(points)} ({center_count/len(points)*100:.1f}%)")

# 检查重叠程度
hull_items = list(series_hull.items())
overlap_count = 0
for i in range(len(hull_items)):
    for j in range(i+1, len(hull_items)):
        n1, h1 = hull_items[i]
        n2, h2 = hull_items[j]
        dx = h1['cx'] - h2['cx']
        dy = h1['cy'] - h2['cy']
        cd = math.sqrt(dx*dx + dy*dy)
        if cd - (h1['r'] + h2['r']) < 0:
            overlap_count += 1
total_pairs = len(hull_items) * (len(hull_items) - 1) // 2
print(f"  重叠 hull 对: {overlap_count}/{total_pairs} ({overlap_count/total_pairs*100:.1f}%)")
