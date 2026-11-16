from pathlib import Path

import pymupdf as fitz


def load_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True) or ""
        text = " ".join(text.split())
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


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Total chunks: {len(chunks)}")