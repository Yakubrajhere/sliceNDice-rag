"""Offline ingestion: load -> split -> embed -> store.

Usage:
    python ingest.py            # add/update documents (idempotent)
    python ingest.py --reset    # wipe the vector store first
"""
import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from src.document_loader import load_documents
from src.embeddings import OllamaEmbeddings
from src.text_splitter import split_documents
from src.vector_db import VectorDB


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--reset", action="store_true", help="Delete existing vectors before ingesting")
    args = parser.parse_args()

    data_dir = os.getenv("DATA_DIR", "data")
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

    print(f"Loading documents from '{data_dir}' ...")
    documents = load_documents(data_dir)
    if not documents:
        print("No documents found. Add PDF/TXT/DOCX files to the data/ folder.")
        return

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    db = VectorDB(OllamaEmbeddings())
    if args.reset:
        db.reset()
        print("Vector store reset.")

    print("Embedding and storing ...")
    db.add_chunks(chunks)
    print(f"Done. Vector store now holds {db.count()} chunks.")


if __name__ == "__main__":
    main()
