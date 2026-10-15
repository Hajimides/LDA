"""把 data/modern/ 的現代小說切塊並分詞（jieba 詞性標註），結果快取到 data/modern_seg/

每部小說按段落累積成約 CHUNK_CHARS 個漢字的一塊；每塊保留原文和兩字以上的詞（詞, 詞性）。
用法：python modern_segment.py
"""

import json
import re
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "modern"
OUT = ROOT / "data" / "modern_seg"

CHUNK_CHARS = 2000
MIN_TAIL = 500  # 最後一塊太短就併入前一塊
HANZI = re.compile(r"[一-鿿]")
SENT_END = re.compile(r"(?<=[。！？…])")


def units(text):
    """段落；過長的段落再按句號切開，避免一段就超過一塊。"""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= CHUNK_CHARS:
            yield line
            continue
        buf = ""
        for sent in SENT_END.split(line):
            buf += sent
            if len(buf) >= CHUNK_CHARS // 2:
                yield buf
                buf = ""
        if buf:
            yield buf


def chunk(text):
    chunks, buf, n = [], [], 0
    for u in units(text):
        buf.append(u)
        n += len(HANZI.findall(u))
        if n >= CHUNK_CHARS:
            chunks.append("\n".join(buf))
            buf, n = [], 0
    if buf:
        if chunks and n < MIN_TAIL:
            chunks[-1] += "\n" + "\n".join(buf)
        else:
            chunks.append("\n".join(buf))
    return chunks


def segment(path):
    import jieba
    import jieba.posseg as pseg

    jieba.setLogLevel(60)
    out = OUT / (path.stem + ".json")
    if out.exists():
        return path.stem, "cached"
    text = path.read_text(encoding="utf-8")
    result = []
    for c in chunk(text):
        tokens = [[p.word, p.flag] for p in pseg.cut(c) if len(p.word) >= 2 and HANZI.search(p.word)]
        result.append({"text": c, "tokens": tokens})
    out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return path.stem, len(result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.txt"), key=lambda p: -p.stat().st_size)  # 大檔先跑，負載較均勻
    with Pool(8) as pool:
        for i, (name, n) in enumerate(pool.imap_unordered(segment, files), 1):
            print(f"[{i}/{len(files)}] {name}: {n}", flush=True)


if __name__ == "__main__":
    main()
