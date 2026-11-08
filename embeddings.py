"""Lazy embedding clients; local windows prevent silent model truncation."""
from functools import lru_cache
from threading import Lock

import numpy as np

from config import get_settings
from llm.ollama_client import embed_texts


class Embedder:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.lock = Lock()
        if self.settings.embedding_provider == "openai":
            from openai import OpenAI
            key = self.settings.openai_api_key
            if not key or not key.get_secret_value():
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
            self.client = OpenAI(api_key=key.get_secret_value(), timeout=60, max_retries=3)
        elif self.settings.embedding_provider == "local":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            self.model.eval()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.embedding_provider == "openai":
            response = self.client.embeddings.create(model="text-embedding-3-small", input=texts,
                                                       dimensions=1536)
            vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        elif self.settings.embedding_provider == "ollama":
            vectors = embed_texts(str(self.settings.ollama_base_url), self.settings.ollama_embedding_model, texts)
        else:
            # Token IDs are passed directly to avoid decode/re-encode token growth.
            with self.lock:
                tokenizer = self.model.tokenizer
                capacity = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
                windows, owners, weights = [], [], []
                for owner, content in enumerate(texts):
                    ids = tokenizer.encode(content, add_special_tokens=False)
                    for start in range(0, len(ids), capacity):
                        part = ids[start:start + capacity]
                        # MiniLM uses BERT's [CLS] ... [SEP] layout. Construct
                        # features directly; Transformers 5 removed prepare_for_model.
                        input_ids = [tokenizer.cls_token_id, *part, tokenizer.sep_token_id]
                        windows.append({"input_ids": input_ids,
                                        "attention_mask": [1] * len(input_ids),
                                        "token_type_ids": [0] * len(input_ids)})
                        owners.append(owner)
                        weights.append(len(part))
                sums = np.zeros((len(texts), self.settings.dimensions), dtype=np.float64)
                import torch
                with torch.inference_mode():
                    for start in range(0, len(windows), 32):
                        batch = tokenizer.pad(windows[start:start + 32], padding=True, return_tensors="pt")
                        output = self.model(batch)["sentence_embedding"].cpu().numpy()
                        for offset, vector in enumerate(output):
                            index = start + offset
                            sums[owners[index]] += vector * weights[index]
                vectors = sums.tolist()
        array = np.asarray(vectors, dtype=np.float64)
        if array.shape != (len(texts), self.settings.dimensions) or not np.isfinite(array).all():
            raise ValueError("Embedding provider returned invalid vectors")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Embedding provider returned a zero vector")
        return (array / norms).tolist()


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
