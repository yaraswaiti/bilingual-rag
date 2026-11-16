from pathlib import Path
import re

import pymupdf as fitz


def _extract_ltr_page(page):
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    x0, y0, x1, _ = s["bbox"]
                    spans.append((y0, x0, x1, s["text"]))
    spans.sort(key=lambda s: s[0])
    lines = []
    for s in spans:
        if lines and abs(s[0] - lines[-1][0][0]) < 3.0:
            lines[-1].append(s)
        else:
            lines.append([s])
    parts = []
    for line in lines:
        line.sort(key=lambda s: s[1])
        prev_x1 = None
        for _, x0, x1, t in line:
            if prev_x1 is not None and (x0 - prev_x1) < 1.0:
                parts.append(t)
            else:
                parts.append(" " + t)
            prev_x1 = x1 if prev_x1 is None else max(prev_x1, x1)
    return " ".join("".join(parts).split())


def load_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        raw = page.get_text("text", sort=True) or ""
        if re.search(r"[\u0600-\u06ff]", raw):
            text = " ".join(raw.split())
        else:
            text = _extract_ltr_page(page)
        if text:
            pages.append({"source": Path(path).name, "page": i, "text": text})
    return pages


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def build_chunks(data_dir="data", chunk_size=1200, overlap=200):
    all_chunks = []
    for f in sorted(Path(data_dir).iterdir()):
        if f.suffix.lower() == ".pdf":
            pages = load_pdf(f)
        elif f.suffix.lower() == ".txt":
            text = " ".join(f.read_text(encoding="utf-8").split())
            pages = [{"source": f.name, "page": "-", "text": text}]
        else:
            continue
        for p in pages:
            for c in chunk_text(p["text"], chunk_size, overlap):
                all_chunks.append(
                    {"source": p["source"], "page": p["page"], "text": c}
                )
    return all_chunks


ARTICLE_SPLIT = re.compile(r"(?=(?:Article|المادة)\s+\d+)")


def split_by_article(text, max_len=1500, overlap=200):
    parts = [p.strip() for p in ARTICLE_SPLIT.split(text) if p.strip()]
    out = []
    for p in parts:
        m = re.match(r"(Article|المادة)\s+(\d+)", p)
        label = f"{m.group(1)} {m.group(2)}" if m else "Preamble"
        if len(p) <= max_len:
            out.append((label, p))
        else:
            for i, c in enumerate(chunk_text(p, max_len, overlap)):
                out.append((label, c if i == 0 else f"{label} (cont.) {c}"))
    return out


def build_chunks_by_article(data_dir="data"):
    all_chunks = []
    for f in sorted(Path(data_dir).iterdir()):
        if f.suffix.lower() == ".pdf":
            text = " ".join(p["text"] for p in load_pdf(f))
        elif f.suffix.lower() == ".txt":
            text = " ".join(f.read_text(encoding="utf-8").split())
        else:
            continue
        for label, chunk in split_by_article(text):
            all_chunks.append({"source": f.name, "page": label, "text": chunk})
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Total chunks: {len(chunks)}")