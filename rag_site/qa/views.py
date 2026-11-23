import sys
from pathlib import Path

from rest_framework.decorators import api_view
from rest_framework.response import Response

# make rag.py (in the repo root) importable
sys.path.append(str(Path(__file__).resolve().parents[2]))
from rag import answer  # noqa: E402


@api_view(["POST"])
def ask(request):
    question = request.data.get("question", "").strip()
    if not question:
        return Response({"error": "question is required"}, status=400)
    text, results = answer(question)
    return Response(
        {
            "answer": text,
            "sources": [
                {"source": c["source"], "page": c["page"]} for c in results
            ],
        }
    )
