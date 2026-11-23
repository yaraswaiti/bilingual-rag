import os

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from chunking import build_chunks_by_article

model = SentenceTransformer("intfloat/multilingual-e5-small")


def connect():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load():
    conn = connect()
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        """CREATE TABLE chunks (
            id serial PRIMARY KEY,
            source text,
            page text,
            content text,
            embedding vector(384)
        )"""
    )
    chunks = build_chunks_by_article()
    embs = model.encode(
        ["passage: " + c["text"] for c in chunks],
        normalize_embeddings=True,
        batch_size=32,
    )
    for c, e in zip(chunks, embs):
        conn.execute(
            "INSERT INTO chunks (source, page, content, embedding) VALUES (%s, %s, %s, %s)",
            (c["source"], str(c["page"]), c["text"], e),
        )
    print(f"Loaded {len(chunks)} chunks into PostgreSQL")


def search_pg(query, k=4):
    conn = connect()
    q = model.encode("query: " + query, normalize_embeddings=True)
    rows = conn.execute(
        "SELECT source, page, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (q, k),
    ).fetchall()
    return [{"source": r[0], "page": r[1], "text": r[2]} for r in rows]


if __name__ == "__main__":
    load()
    for r in search_pg("What is this law called?", k=3):
        print(f"\n{r['source']} | {r['page']}\n{r['text'][:150]}")
        