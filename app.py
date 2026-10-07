"""Main RAG application (interactive CLI).

Usage:
    python app.py
Type 'exit' or 'quit' to leave.
"""
from dotenv import load_dotenv

load_dotenv()

from src.embeddings import OllamaEmbeddings
from src.llm import OllamaLLM
from src.prompt_template import build_messages
from src.retriever import Retriever
from src.vector_db import VectorDB


def main() -> None:
    db = VectorDB(OllamaEmbeddings())
    if db.count() == 0:
        print("Vector store is empty. Run: python ingest.py")
        return

    retriever = Retriever(db)
    llm = OllamaLLM()
    print(f"RAG ready | LLM: {llm.model} | chunks: {db.count()} | type 'exit' to quit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        chunks = retriever.retrieve(question)
        messages = build_messages(question, chunks)

        print("Assistant: ", end="", flush=True)
        for token in llm.stream(messages):
            print(token, end="", flush=True)
        print()

        if chunks:
            sources = sorted({f'{c["metadata"]["source"]} (p.{c["metadata"].get("page", 1)})' for c in chunks})
            print("Sources: " + ", ".join(sources))
        print()


if __name__ == "__main__":
    main()
