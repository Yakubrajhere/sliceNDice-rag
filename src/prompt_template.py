"""Build the grounded RAG prompt."""
from typing import Dict, List

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context.\n"
    "Rules:\n"
    "- If the answer is not in the context, say: \"I don't know based on the provided documents.\"\n"
    "- Do not use outside knowledge or guess.\n"
    "- Be concise and accurate.\n"
    "- Cite sources inline like [source: file.pdf, page 2]."
)


def format_context(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        m = c["metadata"]
        parts.append(f'[{i}] (source: {m["source"]}, page {m.get("page", 1)})\n{c["text"]}')
    return "\n\n".join(parts)


def build_messages(question: str, chunks: List[Dict]) -> List[Dict]:
    context = format_context(chunks) if chunks else "(no relevant context found)"
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
