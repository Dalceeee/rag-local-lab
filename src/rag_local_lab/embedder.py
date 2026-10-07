import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LocalEmbedder:
    def __init__(self, model_name: str = "google/embeddinggemma-2", dim: int=768):
        self.model_name = model_name
        self.dim = dim
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            self.model_name,
            config_kwargs={"vision_config": None, "audio_config": None},
            device=device
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, prompt_name="Document", truncate_dim=self.dim)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode([query], prompt_name="SearchQuery", truncate_dim=self.dim)
        return embedding