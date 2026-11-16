import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import build_chunks

model = SentenceTransformer("intfloat/multilingual-e5-small")

chunks = build_chunks()
texts = ["passage: " + c["text"] for c in chunks]
emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def search(query, k=3):
    q = model.encode("query: " + query, normalize_embeddings=True)
    scores = emb @ q
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in top]


for question in [
    "كم مدة الإجازة السنوية للعامل؟",
    "What is the notice period for terminating a contract?",
]:
    print("\n" + "=" * 70)
    print("Q:", question)
    for score, c in search(question):
        print(f"\n[{score:.2f}] {c['source']} | page {c['page']}")
        print(c["text"][:300])