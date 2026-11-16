from fastapi import FastAPI
from pydantic import BaseModel

from rag import answer

app = FastAPI(title="Bilingual Labor Law RAG")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Question):
    text, results = answer(body.question)
    return {
        "answer": text,
        "sources": [
            {"source": c["source"], "page": c["page"]} for c in results
        ],
    }