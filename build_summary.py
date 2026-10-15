"""把 results/ 裡的 CSV 彙整成一頁 HTML：summary.html

用法：python build_summary.py [artifact.html]（先跑完 tang_lda.py 的五組實驗）
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RES = ROOT / "results"

# 主題命名來自 分析.md 和 分析_去停用词.md；status：clear 一看就懂 / partial 勉強 / unclear 看不懂
RUNS = [
    {
        "id": "k10",
        "label": "10 个主题",
        "group": "原始",
        "desc": "只去标点，不去任何字。",
        "stop": 0,
        "names": [
            ("春花·艳情", "clear"), ("说理·议论（虚字）", "unclear"), ("离别·羁旅", "clear"),
            ("江行·水路", "clear"), ("虚字·口语议论", "unclear"), ("郊庙·宫廷", "clear"),
            ("边塞·战争", "clear"), ("山林·隐逸·僧寺", "clear"), ("秋夜·月色", "clear"),
            ("近体口语·人生感慨", "unclear"),
        ],
    },
    {
        "id": "k20",
        "label": "20 个主题",
        "group": "原始",
        "desc": "只去标点，不去任何字。",
        "stop": 0,
        "names": [
            ("惜春·伤春", "clear"), ("边塞", "clear"), ("秋夜·闺怨", "clear"),
            ("清幽·玄理", "partial"), ("田园·春景·故园", "clear"), ("叹老·人生", "partial"),
            ("饮酒·宴游", "clear"), ("江行·水路", "clear"), ("骚体·赋体（兮）", "partial"),
            ("闺阁·花间词", "clear"), ("咏史·说理", "unclear"), ("郊庙乐章", "clear"),
            ("游仙·道教", "clear"), ("离别·怀古·乡愁", "clear"), ("秋景·咏物", "clear"),
            ("苦吟·闲居·诗僧", "clear"), ("应制·朝廷", "clear"), ("山林·寺院", "clear"),
            ("咏物（杂）", "unclear"), ("口语·白话感慨", "unclear"),
        ],
    },
    {
        "id": "k20_stopwords",
        "label": "20 个主题",
        "group": "只去虚字",
        "desc": "去掉 33 个最常见的虚字（不 无 有 为 何 自 是 我 之 其 兮 一 得 来 去 时 等）。",
        "stop": 33,
        "names": [
            ("清幽·游览", "partial"), ("田园·山居", "clear"), ("边塞", "clear"),
            ("离别·相思", "clear"), ("春景·花柳", "clear"), ("山水·奇景", "clear"),
            ("闺阁·花间词", "clear"), ("登临·边愁", "clear"), ("寓言·讽喻", "clear"),
            ("议论·说理", "partial"), ("秋夜", "clear"), ("饮酒·宴游", "clear"),
            ("禅寺·僧侣", "clear"), ("郊庙乐章", "clear"), ("归思·归鸟", "clear"),
            ("江行·水路", "clear"), ("叹老·闲游", "clear"), ("泛义字（人 年 日）", "unclear"),
            ("文士·仕途", "clear"), ("游仙·宫廷", "clear"),
        ],
    },
    {
        "id": "k10_clean",
        "label": "10 个主题",
        "group": "去停用字",
        "desc": "用 stopwords.txt 去掉 164 个字：虚词 + 泛义高频字（人 年 日 时 中 上 下 等）。",
        "stop": 164,
        "names": [
            ("秋夜·咏物", "clear"), ("游仙·道教", "clear"), ("送别·羁旅", "clear"),
            ("德·道（郊庙 + 说理）", "partial"), ("江行·水路", "clear"), ("应制·宫廷", "clear"),
            ("春花·艳情", "clear"), ("闲居·叹老·饮酒", "clear"), ("边塞", "clear"),
            ("山林·寺院·隐逸", "clear"),
        ],
    },
    {
        "id": "k20_clean",
        "label": "20 个主题",
        "group": "去停用字",
        "desc": "用 stopwords.txt 去掉 164 个字：虚词 + 泛义高频字（人 年 日 时 中 上 下 等）。",
        "stop": 164,
        "names": [
            ("夜·愁思·梦", "clear"), ("宫宴·应制", "clear"), ("庭园草木·咏物", "clear"),
            ("闺阁·花间词", "clear"), ("离情·相思", "clear"), ("边塞", "clear"),
            ("田园·山居", "clear"), ("怀古·城池", "clear"), ("议论·讽谏", "clear"),
            ("老病·饮酒·闲适", "clear"), ("游仙·修道", "clear"), ("文士·仕途·赠答", "clear"),
            ("春景·花柳", "clear"), ("羁旅·乡愁", "clear"), ("清幽·闲适（韦应物体）", "partial"),
            ("山寺·禅僧", "clear"), ("乐府·民歌·俗语", "partial"), ("郊庙乐章", "clear"),
            ("江行·水路", "clear"), ("神物·奇想", "partial"),
        ],
    },
]


def read_csv(name):
    with open(RES / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def doc_topics(run_id, k):
    rows = read_csv(f"doc_topics_{run_id}.csv")
    theta = np.array([[float(r[f"topic_{t}"]) for t in range(k)] for r in rows])
    length = np.array([int(r["n_chars"]) for r in rows])
    top = np.array([int(r["top_topic"]) for r in rows])
    return theta, length, top


def build_run(run):
    rid, k = run["id"], len(run["names"])
    words = [[] for _ in range(k)]
    for r in read_csv(f"topic_words_{rid}.csv"):
        words[int(r["topic"])].append([r["char"], round(float(r["prob"]), 5)])
    poems = [[] for _ in range(k)]
    for r in read_csv(f"typical_poems_{rid}.csv"):
        poems[int(r["topic"])].append(
            {"title": r["title"], "author": r["author"] or "佚名", "prob": float(r["topic_prob"]), "text": r["text"]}
        )
    theta, length, _ = doc_topics(rid, k)
    # 主題在全部文本中的份量：以每首詩的字數加權
    share = (theta * length[:, None]).sum(0) / length.sum()
    topics = [
        {"id": t, "name": n, "status": s, "share": round(float(share[t]), 4), "words": words[t], "poems": poems[t]}
        for t, (n, s) in enumerate(run["names"])
    ]
    return {**{key: run[key] for key in ("id", "label", "group", "desc", "stop")}, "k": k, "topics": topics}


def split_matrix(k10_id, k20_id):
    """每個 20 主題的詩（按最主要主題歸類），落在各個 10 主題的比例。"""
    _, _, top10 = doc_topics(k10_id, 10)
    _, _, top20 = doc_topics(k20_id, 20)
    m = np.zeros((20, 10))
    np.add.at(m, (top20, top10), 1)
    counts = m.sum(1)
    return {"rows": (m / counts[:, None]).round(3).tolist(), "counts": counts.astype(int).tolist()}


def load_stopword_groups():
    """停用字表按註釋分組：每組的標籤是它上面最近的一行註釋。"""
    groups, label = [], ""
    for line in (ROOT / "stopwords.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            label = line.strip("#─ ").split("（")[0].split("、", 1)[-1] if "──" in line else line.strip("# ")
            continue
        words = line.split("#", 1)[0].split()
        if words:
            groups.append({"label": label, "words": list(dict.fromkeys(words))})
    return groups


def main():
    data = {
        "runs": [build_run(r) for r in RUNS],
        "splits": {
            "原始": {"k10": "k10", "k20": "k20", **split_matrix("k10", "k20")},
            "去停用字": {"k10": "k10_clean", "k20": "k20_clean", **split_matrix("k10_clean", "k20_clean")},
        },
        "stopwords": load_stopword_groups(),
    }
    template = (ROOT / "summary_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = template.replace("__DATA__", payload)

    # 本機直接開啟用的完整文件
    out = ROOT / "summary.html"
    out.write_text(STANDALONE.format(page=page), encoding="utf-8")
    print(f"已寫入 {out}（{out.stat().st_size / 1024:.0f} KB）")
    # 發佈成 Artifact 時只要頁面內容，外殼由平台補上
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(page, encoding="utf-8")
        print(f"已寫入 {sys.argv[1]}")


STANDALONE = """<!doctype html>
<html lang="zh-Hans">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
{page}
</body>
</html>
"""


if __name__ == "__main__":
    main()
