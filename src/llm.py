"""Local LLM configuration (Ollama)."""
import os
from typing import Dict, Iterator, List

import ollama


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
        self.client = ollama.Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.options = {
            "temperature": float(os.getenv("TEMPERATURE", "0.1")),
            "num_ctx": int(os.getenv("NUM_CTX", "4096")),
        }

    def chat(self, messages: List[Dict]) -> str:
        response = self.client.chat(model=self.model, messages=messages, options=self.options)
        return response["message"]["content"]

    def stream(self, messages: List[Dict]) -> Iterator[str]:
        for part in self.client.chat(model=self.model, messages=messages, options=self.options, stream=True):
            yield part["message"]["content"]
