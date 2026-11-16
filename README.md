# Bilingual RAG: Jordanian Labor Law Q&A

A Retrieval-Augmented Generation system that answers questions about the
Jordanian Labor Law in Arabic and English, with source citations.

## How it works
1. Load documents (PDF and UTF-8 text) and split into overlapping chunks (800 chars, 150 overlap).
2. Embed chunks with the multilingual `intfloat/multilingual-e5-small` model.
3. Retrieve the top matches by cosine similarity.
4. Send them to Gemini with a prompt that answers only from the context and cites sources.

## Run
    pip install sentence-transformers pymupdf numpy google-genai
    $env:GEMINI_API_KEY="your_key"   # PowerShell
    python rag.py

## Evaluation
    python eval.py
Evaluated on 30 bilingual questions (hit@3): 87% at 400 chars, 90% at 800, 93% at 1200.
Run the API: uvicorn api:app --reload, then POST /ask with {"question": "..."}
## Problem I solved
The Arabic PDF extracted with corrupted characters (the letters "لا" became "ال"),
so I converted the source text to clean UTF-8 and kept the PDF reader for English.