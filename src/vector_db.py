"""Persistent Chroma vector store."""
import hashlib
import os
from typing import Dict, List

import chromadb

from src.embeddings import OllamaEmbeddings


class VectorDB:
    def __init__(self, embeddings: OllamaEmbeddings, persist_dir: str | None = None, collection: str | None = None):
        self.embeddings = embeddings
        self.persist_dir = persist_dir or os.getenv("VECTORSTORE_DIR", "vectorstore")
        self.collection_name = collection or os.getenv("COLLECTION_NAME", "rag_documents")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _make_id(chunk: Dict) -> str:
        m = chunk["metadata"]
        raw = f'{m["source"]}|{m.get("page", 0)}|{m["chunk"]}|{chunk["text"]}'
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def reset(self) -> None:
        """Delete and recreate the collection (full re-ingest)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict], batch_size: int = 64) -> int:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            self.collection.upsert(
                ids=[self._make_id(c) for c in batch],
                documents=texts,
                embeddings=self.embeddings.embed_documents(texts),
                metadatas=[c["metadata"] for c in batch],
            )
            print(f"  stored {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    # def search(self, query: str, top_k: int = 4) -> List[Dict]:
    #     if self.count() == 0:
    #         return []
    #     result = self.collection.query(
    #         query_embeddings=[self.embeddings.embed_query(query)],
    #         n_results=min(top_k, self.count()),
    #         include=["documents", "metadatas", "distances"],
    #     )
    #     hits = []
    #     for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
    #         hits.append({"text": doc, "metadata": meta, "score": 1.0 - dist})  # cosine similarity
    #     return hits

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        # Result lists can be empty or missing if collection is corrupted
        if not result or not result["documents"] or not result["documents"][0]:
            return []
            
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            # Chromadb HNSW Cosine Distance ranges from 0.0 (identical) to 2.0 (opposite)
            # Correct conversion to Cosine Similarity (1.0 means identical, 0.0 means opposite)
            similarity_score = 1.0 - (dist / 2.0)
            
            hits.append({
                "text": doc, 
                "metadata": meta, 
                "score": round(similarity_score, 4)
            })
        return hits
