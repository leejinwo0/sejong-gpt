"""질문에 맞는 사료 청크를 찾아 주는 검색기 (앱에서 사용)."""
import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index"
EMBED_CACHE = ROOT / "models" / "embeddings"   # build_index.py와 같은 위치


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        info = json.loads((index_dir / "info.json").read_text(encoding="utf-8"))
        self.query_prefix = info["query"]
        self.model = TextEmbedding(info["name"], cache_dir=str(EMBED_CACHE))
        self.embeddings = np.load(index_dir / "embeddings.npy")
        self.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))

    def search(self, question: str, k: int = 3) -> list[dict]:
        vec = np.array(next(iter(self.model.embed([self.query_prefix + question]))), dtype=np.float32)
        vec /= np.linalg.norm(vec)
        scores = self.embeddings @ vec
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]
