"""把 results/modern/ 的 CSV 彙整成一頁 HTML：modern_summary.html

用法：python build_modern_summary.py [artifact.html]（先跑完 modern_lda.py）
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from build_summary import STANDALONE

ROOT = Path(__file__).parent
RES = ROOT / "results" / "modern"
PERIODS = ["1949年前", "1949–1976", "1977–1999", "2000年后"]

# 主題命名：根據前 15 個詞、最常用的小說和代表段落；status：clear / partial / unclear
NAMES = {
    "k20": [
        ("五四抒情·爱与痛苦", "clear"), ("北方乡土口语", "partial"), ("当代都市男女", "clear"),
        ("工厂与城市管理", "clear"), ("朝廷·衙门", "clear"), ("洋场客厅·太太小姐", "clear"),
        ("家庭与乡镇日常", "partial"), ("侠义·江湖（评书）", "clear"), ("当代企业·商场", "clear"),
        ("学堂·国家·留学", "clear"), ("老舍的北平", "clear"), ("演义与鸳鸯蝴蝶（旧小说腔）", "clear"),
        ("农业合作化", "clear"), ("当代乡村·基层", "clear"), ("哲思·宇宙·人类", "clear"),
        ("战场·部队", "clear"), ("革命·组织话语", "clear"), ("家族史·身体（新历史小说）", "clear"),
        ("抗日游击", "clear"), ("风景描写", "partial"),
    ],
    "k40": [
        ("家族·亲属称谓", "partial"), ("家宴·亲戚往来", "clear"), ("哲思·存在", "clear"),
        ("江湖奇侠·神怪", "clear"), ("五四抒情·爱与痛苦", "clear"), ("当代企业·商场", "clear"),
        ("上海工商业改造", "clear"), ("穷人·粗口（北方乡土）", "clear"), ("特务·监狱·情报", "partial"),
        ("改革官场", "clear"), ("婚姻·家庭", "clear"), ("红军·土地革命", "clear"),
        ("张恨水的客厅", "clear"), ("合作化（浩然）", "clear"), ("舞台·演出", "clear"),
        ("学校·教师", "clear"), ("宗族·祠堂", "clear"), ("周立波的湖南乡村", "clear"),
        ("侠义·评书", "clear"), ("四川市镇·商铺", "clear"), ("抗日游击", "clear"),
        ("医院·疾病·死亡", "clear"), ("老舍的北平", "clear"), ("动作描写", "partial"),
        ("朝廷·皇帝", "clear"), ("北方农家院落", "partial"), ("民国文人议论", "clear"),
        ("草原·狼", "clear"), ("国家·时局", "clear"), ("干部·组织话语", "clear"),
        ("新文艺·知识青年", "clear"), ("土司与剿匪（边地）", "partial"), ("赵树理的晋东南", "clear"),
        ("身体·感官（先锋小说）", "clear"), ("战场·部队", "clear"), ("革命激情", "clear"),
        ("演义·兵马", "clear"), ("当代乡村·县城", "clear"), ("风景描写", "partial"),
        ("都市男女（上海）", "clear"),
    ],
}

FINDINGS = [
    ("主题 = 题材 + 作家 + 时代话语",
     "和唐诗不同，小说的主题常常就是一位作家的语言：老舍的北平、张恨水的客厅、赵树理的晋东南、周立波的湖南乡村。作家的人名去掉了，方言词、称谓和叙述腔调还在。"),
    ("年代是最强的信号",
     "合作化、抗日游击、上海工商业改造几乎只出现在 1949–1976（约 3 倍）；武侠、演义、张恨水式客厅只在 1949 年前；当代企业、草原在 2000 年后最集中（企业主题是平均的 5 倍）。"),
    ("十七年文学的“话语”可以量化",
     "“同志、革命、群众、组织、思想、毛主席”自成主题，1949–1976 是平均的约 2 倍；到了 1977–1999，同样的“领导、干部、问题”换成改革官场（张平、柯云路）。"),
    ("战争写法分成两种",
     "抗日游击（鬼子、伪军、民兵、炮楼）和正规战场（连长、团长、阵地）在 20 个和 40 个主题里都是两个独立主题，前者是乡村群众战争，后者是解放战争和朝鲜战争的部队叙事。"),
    ("40 个主题拆出的是作家和写法",
     "20→40 时，合作化拆成浩然和赵树理；旧小说腔拆成神怪武侠和三国演义；革命话语拆成红军、干部组织、革命激情、特务监狱；新历史小说拆成宗族、身体感官、疾病死亡、边地等六个。主题越多，越接近“一位作家或一种写法一个主题”。"),
    ("有些主题是文体",
     "风景描写、动作描写、北方口语、旧小说腔，这些主题跨越题材，代表的是叙述方式。研究题材时可以忽略它们，研究风格时它们反而是重点。"),
    ("模型能找出语料的问题",
     "第一次训练时，“魏巍”“张爱玲《金锁记》”在武侠、都市、抒情等毫不相干的主题里都有高比例。打开文件才发现是两部网络言情小说；顺着查下去，又发现《故事新编》的文件其实是《呐喊》，《棋王》是张系国的同名小说。"),
    ("还有漏网的名字",
     "铁锁、登高（赵树理）、神童（张系国）等仍在主题词里。jieba 没认出来的名字只能靠人工补充；这也提醒我们，主题词表要配合原文阅读。"),
]


def read_csv(name):
    with open(RES / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_run(rid):
    names = NAMES[rid]
    k = len(names)
    words = [[] for _ in range(k)]
    for r in read_csv(f"topic_words_{rid}.csv"):
        words[int(r["topic"])].append([r["word"], round(float(r["prob"]), 5)])
    passages = [[] for _ in range(k)]
    for r in read_csv(f"typical_passages_{rid}.csv"):
        passages[int(r["topic"])].append(
            {"author": r["author"], "title": r["title"], "year": r["year"], "prob": float(r["topic_prob"]), "excerpt": r["excerpt"]}
        )
    periods = {int(r["topic"]): r for r in read_csv(f"period_topics_{rid}.csv")}
    novels = read_csv(f"novel_topics_{rid}.csv")
    topics = []
    for t, (name, status) in enumerate(names):
        best = sorted(novels, key=lambda r: -float(r[f"topic_{t}"]))[:5]
        topics.append({
            "id": t, "name": name, "status": status,
            "share": float(periods[t]["share"]),
            "periods": [float(periods[t][p]) for p in PERIODS],
            "words": words[t],
            "novels": [[f"{r['author']}《{r['title']}》", float(r[f"topic_{t}"])] for r in best],
            "passages": passages[t],
        })
    return {"id": rid, "k": k, "topics": topics}


def split_matrix():
    top20 = np.array([int(r["top_topic"]) for r in read_csv("doc_topics_k20.csv")])
    top40 = np.array([int(r["top_topic"]) for r in read_csv("doc_topics_k40.csv")])
    m = np.zeros((40, 20))
    np.add.at(m, (top40, top20), 1)
    counts = m.sum(1)
    return {"rows": (m / np.maximum(counts, 1)[:, None]).round(3).tolist(), "counts": counts.astype(int).tolist()}


def main():
    n20 = {r["file"]: r for r in read_csv("novel_topics_k20.csv")}
    n40 = {r["file"]: r for r in read_csv("novel_topics_k40.csv")}
    novels = [
        {
            "file": f, "author": r["author"], "title": r["title"], "year": r["year"], "period": r["period"],
            "chunks": int(r["n_chunks"]),
            "k20": [round(float(r[f"topic_{t}"]), 4) for t in range(20)],
            "k40": [round(float(n40[f][f"topic_{t}"]), 4) for t in range(40)],
        }
        for f, r in n20.items()
    ]
    with open(ROOT / "modern_meta.csv", encoding="utf-8-sig") as fh:
        issues = [r for r in csv.DictReader(fh) if r["exclude"] or "内容" in r["note"]]
    n_chunks = sum(n["chunks"] for n in novels)
    data = {
        "periods": PERIODS,
        "runs": [build_run("k20"), build_run("k40")],
        "novels": novels,
        "split": split_matrix(),
        "defaultNovel": "老舍_骆驼祥子",
        "facts": [["小说", f"{len(novels)}"], ["段落块", f"{n_chunks:,}"], ["词（过滤后）", "5,719,905"], ["词表", "49,868"]],
        "findings": FINDINGS,
        "issues": [{"file": r["file"], "note": r["note"], "exclude": r["exclude"]} for r in issues],
    }
    template = (ROOT / "modern_summary_template.html").read_text(encoding="utf-8")
    page = template.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out = ROOT / "modern_summary.html"
    out.write_text(STANDALONE.format(page=page), encoding="utf-8")
    print(f"已寫入 {out}（{out.stat().st_size / 1024:.0f} KB）")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(page, encoding="utf-8")
        print(f"已寫入 {sys.argv[1]}")


if __name__ == "__main__":
    main()
