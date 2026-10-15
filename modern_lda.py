"""現代中文小說主題模型（qhchina-data/corpora/modern，172 部）

先跑 modern_segment.py 分詞。每部小說切成約 2000 字的塊，每塊算一篇文件；
只保留兩字以上的名詞、動詞、形容詞、成語，去掉人名（nr*）、停用詞（modern_stopwords.txt）、
只在少於 --min-novels 部小說裡出現的詞（多半是各書的人物、地名），
以及出現在超過 --max-df 比例的塊裡的泛義詞。

用法：
    python modern_lda.py                    # 預設跑 20 和 40 個主題
    python modern_lda.py -k 30 --tag test
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.helpers.texts import load_stopwords

ROOT = Path(__file__).parent
SEG = ROOT / "data" / "modern_seg"
META = ROOT / "modern_meta.csv"
OUT = ROOT / "results" / "modern"

# jieba 詞性：名詞、處所、機構、其他專名、動詞、名動詞、形容詞、名形詞、成語、習用語
KEEP_POS = {"n", "ns", "nt", "nz", "s", "v", "vn", "a", "an", "i", "l"}
MIN_TOKENS = 50  # 過濾後太短的塊（目錄、序言殘段）不算文件
PERIODS = ["1949年前", "1949–1976", "1977–1999", "2000年后"]


def load_meta():
    with open(META, encoding="utf-8-sig") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def load_corpus(args):
    meta = load_meta()
    stop = load_stopwords("zh_sim") | load_stopwords("zh_cl_sim")
    if args.stopwords_file:
        for line in args.stopwords_file.read_text(encoding="utf-8").splitlines():
            stop.update(line.split("#", 1)[0].split())

    raw = {}
    novel_df = Counter()  # 每個詞出現在幾部小說
    for path in sorted(SEG.glob("*.json")):
        if meta[path.stem]["exclude"]:  # 冒名或重複的文件，見 modern_meta.csv
            continue
        chunks = json.loads(path.read_text(encoding="utf-8"))
        kept = [
            [w for w, pos in c["tokens"] if pos in KEEP_POS and w not in stop]
            for c in chunks
        ]
        raw[path.stem] = (chunks, kept)
        novel_df.update({w for ws in kept for w in ws})

    # 塊頻：出現在太多塊裡的詞（起來、知道……）對分主題沒有幫助
    chunk_df = Counter(w for _, kept in raw.values() for ws in kept for w in set(ws))
    n_chunks = sum(len(kept) for _, kept in raw.values())
    vocab_ok = {
        w for w, n in novel_df.items() if n >= args.min_novels and chunk_df[w] / n_chunks <= args.max_df
    }
    docs, info = [], []
    for name, (chunks, kept) in raw.items():
        for i, (c, ws) in enumerate(zip(chunks, kept)):
            ws = [w for w in ws if w in vocab_ok]
            if len(ws) < MIN_TOKENS:
                continue
            docs.append(ws)
            info.append({"novel": name, "chunk": i, "text": c["text"]})
    return docs, info, meta


def best_excerpt(text, top_words, width=220):
    """挑出主題詞最密集的一段，截到 width 字。"""
    paras = [p for p in text.split("\n") if len(p) >= 20] or [text]
    score = lambda p: sum(p.count(w) for w in top_words) / math.sqrt(len(p))
    p = max(paras, key=score)
    if len(p) <= width:
        return p
    hits = [p.find(w) for w in top_words if w in p]
    start = max(0, min(hits) - 40) if hits else 0
    return ("…" if start else "") + p[start : start + width] + "…"


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run(k, docs, info, meta, args):
    lda = LDAGibbsSampler(
        n_topics=k,
        alpha=args.alpha,
        beta=args.beta,
        iterations=args.iterations,
        random_state=args.seed,
        min_word_count=args.min_count,
        log_interval=50,
    )
    lda.fit(docs)
    stem = f"k{k}_{args.tag}" if args.tag else f"k{k}"

    topics = lda.get_topics(n_words=15)
    theta = np.array([[p for _, p in lda.get_document_topics(d)] for d in range(len(docs))])
    length = np.array([len(d) for d in docs])
    weight = theta * length[:, None]

    # 小說 × 主題（以詞數加權）
    novels = sorted({x["novel"] for x in info})
    idx = {n: i for i, n in enumerate(novels)}
    novel_w = np.zeros((len(novels), k))
    for d, x in enumerate(info):
        novel_w[idx[x["novel"]]] += weight[d]
    novel_share = novel_w / novel_w.sum(1, keepdims=True)

    # 時期 × 主題
    period_w = {p: np.zeros(k) for p in PERIODS}
    for n, i in idx.items():
        p = meta[n]["period"]
        if p in period_w:
            period_w[p] += novel_w[i]
    period_share = {p: v / v.sum() for p, v in period_w.items()}

    share = weight.sum(0) / weight.sum()
    lines = [f"# 現代小說 LDA：{k} 個主題", ""]
    lines.append(
        f"{len(novels)} 部小說，{len(docs)} 塊；詞表 {len(lda.vocabulary) if hasattr(lda, 'vocabulary') else '—'}；"
        f"iterations={args.iterations}，min_novels={args.min_novels}，max_df={args.max_df}，seed={args.seed}"
    )
    lines.append("")
    typical_rows, word_rows = [], []
    for t, words in enumerate(topics):
        top = [w for w, _ in words]
        word_rows += [(t, r, w, f"{p:.6f}") for r, (w, p) in enumerate(words, 1)]
        lines.append(f"## 主題 {t}（{share[t]:.1%}）：{' '.join(top)}")
        lines.append("")
        best_novels = np.argsort(-novel_share[:, t])[:5]
        lines.append(
            "最常用的小說："
            + "、".join(f"{novels[i]}（{novel_share[i, t]:.0%}）" for i in best_novels)
        )
        lines.append("")
        seen = set()
        for d, prob in lda.get_top_documents(t, n_docs=300):
            n = info[d]["novel"]
            if n in seen or length[d] < 100:
                continue
            seen.add(n)
            excerpt = best_excerpt(info[d]["text"], top)
            m = meta[n]
            typical_rows.append((t, len(seen), m["author"], m["title"], m["year"], info[d]["chunk"], f"{prob:.4f}", excerpt))
            lines.append(f"- **{m['author']}《{m['title']}》**第 {info[d]['chunk']} 塊（主題比例 {prob:.2f}）")
            lines.append(f"  {excerpt}")
            if len(seen) == 3:
                break
        lines.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"topics_{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(OUT / f"topic_words_{stem}.csv", ["topic", "rank", "word", "prob"], word_rows)
    write_csv(
        OUT / f"typical_passages_{stem}.csv",
        ["topic", "rank", "author", "title", "year", "chunk", "topic_prob", "excerpt"],
        typical_rows,
    )
    write_csv(
        OUT / f"novel_topics_{stem}.csv",
        ["file", "author", "title", "year", "period", "n_chunks", *(f"topic_{t}" for t in range(k))],
        [
            (n, meta[n]["author"], meta[n]["title"], meta[n]["year"], meta[n]["period"],
             sum(1 for x in info if x["novel"] == n), *(f"{v:.4f}" for v in novel_share[i]))
            for n, i in idx.items()
        ],
    )
    write_csv(
        OUT / f"period_topics_{stem}.csv",
        ["topic", "share", *PERIODS],
        [(t, f"{share[t]:.4f}", *(f"{period_share[p][t]:.4f}" for p in PERIODS)) for t in range(k)],
    )
    write_csv(
        OUT / f"doc_topics_{stem}.csv",
        ["novel", "chunk", "n_tokens", "top_topic", *(f"topic_{t}" for t in range(k))],
        [
            (x["novel"], x["chunk"], length[d], int(theta[d].argmax()), *(f"{v:.4f}" for v in theta[d]))
            for d, x in enumerate(info)
        ],
    )
    print("\n".join(l for l in lines if l.startswith("## ")))
    print(f"\n已寫入 {OUT}/*_{stem}.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--topics", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--min-novels", type=int, default=5)
    ap.add_argument("--max-df", type=float, default=0.4, help="出現在超過這個比例的塊裡的詞會被去掉")
    ap.add_argument("--stopwords-file", type=Path, default=ROOT / "modern_stopwords.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if not args.stopwords_file.exists():
        args.stopwords_file = None

    docs, info, meta = load_corpus(args)
    n_novels = len({x["novel"] for x in info})
    print(f"{n_novels} 部小說，{len(docs)} 塊，{sum(map(len, docs))} 個詞")
    for k in args.topics:
        run(k, docs, info, meta, args)


if __name__ == "__main__":
    main()
