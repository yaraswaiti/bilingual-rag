import os
import time

import numpy as np
from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer

from chunking import build_chunks

LLM_MODEL = "gemini-3.8-flash"

embedder = SentenceTransformer("intfloat/multilingual-e5-small")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

chunks = build_chunks()
emb = embedder.encode(
    ["passage: " + c["text"] for c in chunks],
    normalize_embeddings=True,
    show_progress_bar=True,
)


def search(query, k=4):
    q = embedder.encode("query: " + query, normalize_embeddings=True)
    top = np.argsort(-(emb @ q))[:k]
    return [chunks[i] for i in top]


def answer(question, k=4):
    results = search(question, k)
    context = "\n\n".join(
        f"[{i + 1}] ({c['source']}) {c['text']}" for i, c in enumerate(results)
    )
    prompt = f"""You are an assistant that answers questions about the Jordanian Labor Law.
Answer ONLY using the context below. If the answer is not in the context,
say you could not find it. Answer in the same language as the question.
After each claim, cite the source number like [1].

Context:
{context}

Question: {question}"""
    for attempt in range(4):
        try:
            resp = client.models.generate_content(model=LLM_MODEL, contents=prompt)
            return resp.text, results
        except errors.ServerError:
            wait = 5 * (attempt + 1)
            print(f"الموديل مشغول، بعيد المحاولة بعد {wait} ثانية...")
            time.sleep(wait)
    return "ما قدرت أوصل للموديل هلأ، جربي بعد شوي.", results


if __name__ == "__main__":
    while True:
        q = input("\nاسألي (أو اكتبي exit): ").strip()
        if q.lower() == "exit":
            break
        text, results = answer(q)
        print("\n" + text)
        print("\nSources:")
        for i, c in enumerate(results, 1):
            print(f"  [{i}] {c['source']} | page {c['page']}")