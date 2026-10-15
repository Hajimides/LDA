"""《全唐詩》主題模型練習（CHI3242 第7週）

每首詩為一篇文件，以「字」為單位，去掉標點，用 qhchina 的 LDAGibbsSampler 訓練。

用法：
    python tang_lda.py                 # 預設跑 10 和 20 個主題
    python tang_lda.py -k 15 --iterations 300 --stopwords 不 无 有
    python tang_lda.py --stopwords-file stopwords.txt --tag clean
"""

import argparse
import csv
import json
import re
from pathlib import Path

from qhchina.analytics.topicmodels import LDAGibbsSampler

DATA = Path(__file__).parent / "data" / "全唐诗.json"
OUT = Path(__file__).parent / "results"

# 只保留漢字（基本區 + 擴展 A），其餘一律視為標點或雜訊
HANZI = re.compile(r"[㐀-䶿一-鿿]")
# 典型詩過短時，主題比例容易被少數字左右，展示時跳過
MIN_POEM_CHARS = 20


def load_stopwords(path):
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        words.update(line.split("#", 1)[0].split())
    return words


def load_poems():
    poems = json.loads(DATA.read_text(encoding="utf-8"))
    docs, meta = [], []
    for p in poems:
        text = "".join(p["paragraphs"])
        chars = HANZI.findall(text)
        if not chars:
            continue
        docs.append(chars)
        meta.append({"title": p["title"], "author": p["author"], "text": text})
    return docs, meta


def typical_poems(lda, docs, meta, topic_id, n=2):
    picked, seen = [], set()
    for doc_id, prob in lda.get_top_documents(topic_id, n_docs=200):
        # 《全唐詩》有重收的詩（同詩異題），只算一次
        text = "".join(docs[doc_id])
        if len(docs[doc_id]) < MIN_POEM_CHARS or text in seen:
            continue
        seen.add(text)
        picked.append((meta[doc_id], prob))
        if len(picked) == n:
            break
    return picked


def run(k, docs, meta, args):
    lda = LDAGibbsSampler(
        n_topics=k,
        alpha=args.alpha,
        beta=args.beta,
        iterations=args.iterations,
        burnin=args.burnin,
        random_state=args.seed,
        min_word_count=args.min_count,
        stopwords=set(args.stopwords) if args.stopwords else None,
        log_interval=50,
    )
    lda.fit(docs)

    lines = [f"# 《全唐詩》LDA：{k} 個主題", ""]
    lines.append(
        f"文件數 {len(docs)}；iterations={args.iterations}，burnin={args.burnin}，"
        f"alpha={args.alpha or '50/K（並估計）'}，beta={args.beta or '1/K'}，"
        f"停用字={len(args.stopwords) if args.stopwords else 0} 個，seed={args.seed}"
    )
    lines.append("")
    for t, words in enumerate(lda.get_topics(n_words=10)):
        top = " ".join(w for w, _ in words)
        lines.append(f"## 主題 {t}：{top}")
        lines.append("")
        for m, prob in typical_poems(lda, docs, meta, t):
            lines.append(f"- **{m['title']}**（{m['author']}，主題比例 {prob:.2f}）")
            lines.append(f"  {m['text']}")
        lines.append("")

    report = "\n".join(lines)
    OUT.mkdir(exist_ok=True)
    stem = f"k{k}_{args.tag}" if args.tag else f"k{k}"
    path = OUT / f"topics_{stem}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n已寫入 {path}")
    save_csv(lda, docs, meta, k, stem)


def write_csv(path, header, rows):
    # utf-8-sig 讓 Excel 直接開啟也不會亂碼
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"已寫入 {path}")


def save_csv(lda, docs, meta, k, stem):
    write_csv(
        OUT / f"topic_words_{stem}.csv",
        ["topic", "rank", "char", "prob"],
        [
            (t, r, w, f"{p:.6f}")
            for t, words in enumerate(lda.get_topics(n_words=10))
            for r, (w, p) in enumerate(words, 1)
        ],
    )
    write_csv(
        OUT / f"typical_poems_{stem}.csv",
        ["topic", "rank", "title", "author", "topic_prob", "text"],
        [
            (t, r, m["title"], m["author"], f"{p:.4f}", m["text"])
            for t in range(k)
            for r, (m, p) in enumerate(typical_poems(lda, docs, meta, t), 1)
        ],
    )
    rows = []
    for doc_id, m in enumerate(meta):
        probs = [p for _, p in lda.get_document_topics(doc_id)]
        top = max(range(k), key=probs.__getitem__)
        rows.append((doc_id, m["title"], m["author"], len(docs[doc_id]), top, *(f"{p:.4f}" for p in probs)))
    write_csv(
        OUT / f"doc_topics_{stem}.csv",
        ["doc_id", "title", "author", "n_chars", "top_topic", *(f"topic_{t}" for t in range(k))],
        rows,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--topics", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--burnin", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--stopwords", nargs="*", default=None)
    ap.add_argument("--stopwords-file", type=Path, help="停用字表，# 後為註釋，空白分隔")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="輸出檔名後綴，方便比較不同參數")
    args = ap.parse_args()
    if args.stopwords_file:
        args.stopwords = sorted(set(args.stopwords or []) | load_stopwords(args.stopwords_file))

    docs, meta = load_poems()
    print(f"共 {len(docs)} 首詩，{sum(map(len, docs))} 個字")
    for k in args.topics:
        run(k, docs, meta, args)


if __name__ == "__main__":
    main()
