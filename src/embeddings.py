"""Embeddings via a local Ollama embedding model."""
import os
from typing import List

import ollama


class OllamaEmbeddings:
    def __init__(self, model: str | None = None, host: str | None = None, batch_size: int = 32):
        self.model = model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.client = ollama.Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embed(model=self.model, input=batch)
            vectors.extend(response["embeddings"])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
