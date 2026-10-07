"""Retrieve the most relevant chunks for a question."""
import os
from typing import Dict, List

from src.vector_db import VectorDB


class Retriever:
    def __init__(self, vector_db: VectorDB, top_k: int | None = None, min_score: float = 0.55):
        self.vector_db = vector_db
        self.top_k = top_k or int(os.getenv("TOP_K", "4"))
        self.min_score = float(os.getenv("MIN_SCORE", min_score))

    def retrieve(self, question: str) -> List[Dict]:
        hits = self.vector_db.search(question, top_k=self.top_k)
        
        # Filter strictly out bad matches below our similarity score floor
        valid_hits = [h for h in hits if h["score"] >= self.min_score]
        
        # Simple non-intrusive debug console tracker
        print(f"\n[RAG Engine] Evaluated {len(hits)} index hits -> Keeping {len(valid_hits)} items above score threshold {self.min_score}")
        for i, h in enumerate(hits):
            status = "PASS" if h["score"] >= self.min_score else "FAIL"
            print(f"   └─ [{status}] Score: {h['score']:.3f} | Source: {h['metadata'].get('source')} (p.{h['metadata'].get('page')})")
            
        return valid_hits

