"""RAG 검색 인덱스 만들기 (7단계).

청크 384개를 임베딩해서 index/ 에 저장합니다. 문서가 수백 개 수준이라
FAISS 같은 별도 검색 엔진 없이 numpy 행렬 곱 한 번으로 충분히 빠릅니다.

--compare 를 붙이면 후보 임베딩 모델들을 우리 QA 데이터로 비교합니다.
QA마다 '정답 근거 청크'가 기록되어 있으므로, 질문으로 검색했을 때
그 청크가 상위 k개 안에 들어오는지(Recall@k)를 바로 잴 수 있습니다.

실행 (저장소 루트에서):
    python -m pipeline.build_index --compare     # 후보 모델 비교 (10~20분)
    python -m pipeline.build_index               # 선택한 모델로 인덱스 생성
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
QA_FILE = ROOT / "data" / "qa" / "filtered_qa.jsonl"
INDEX_DIR = ROOT / "index"
# 임베딩 모델 저장 위치. 기본값(임시 폴더)은 Windows가 정리하면서 지울 수 있어서 models/ 아래에 고정
EMBED_CACHE = ROOT / "models" / "embeddings"

# fastembed(ONNX Runtime 기반)로 임베딩. PyTorch 없이 동작해서 설치가 가볍고,
# Windows 스마트 앱 컨트롤이 PyTorch DLL을 막는 환경에서도 쓸 수 있음.
# 모델마다 질문/문서 앞에 붙이는 표시가 다름 (e5 계열은 학습 방식상 필요)
MODELS = {
    "minilm": {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "query": "", "passage": ""},
    "mpnet": {"name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "query": "", "passage": ""},
    "e5-large": {"name": "intfloat/multilingual-e5-large", "query": "query: ", "passage": "passage: "},
}
DEFAULT_MODEL = "e5-large"
N_EVAL = 400   # 비교에 쓸 질문 수 (전체를 쓰면 느려서 고정 표본 사용)


def passage_text(chunk: dict) -> str:
    """위치 머리말을 본문 앞에 붙여 '무엇에 관한 문단인지'를 임베딩에 담음."""
    return f"{chunk['header'].strip('[]')}\n{chunk['text']}"


def embed(model: TextEmbedding, texts: list[str], prefix: str) -> np.ndarray:
    vecs = np.array(list(model.embed([prefix + t for t in texts], batch_size=16)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)   # 코사인 유사도를 위해 길이 1로


def evaluate(key: str, chunks: list[dict], qas: list[dict]) -> dict:
    cfg = MODELS[key]
    print(f"\n[{key}] {cfg['name']} 불러오는 중 (처음엔 내려받기 때문에 오래 걸림)")
    model = TextEmbedding(cfg["name"], cache_dir=str(EMBED_CACHE))
    chunk_ids = [c["chunk_id"] for c in chunks]
    doc = embed(model, [passage_text(c) for c in chunks], cfg["passage"])

    start = time.time()
    q = embed(model, [qa["question"] for qa in qas], cfg["query"])
    per_query_ms = (time.time() - start) / len(qas) * 1000

    ranks = []
    for vec, qa in zip(q, qas):
        order = np.argsort(-doc @ vec)
        ranks.append(next(i for i, idx in enumerate(order) if chunk_ids[idx] == qa["chunk_id"]) + 1)
    ranks = np.array(ranks)
    return {
        "model": key,
        "recall@1": round(float((ranks <= 1).mean()), 3),
        "recall@2": round(float((ranks <= 2).mean()), 3),
        "recall@3": round(float((ranks <= 3).mean()), 3),
        "recall@5": round(float((ranks <= 5).mean()), 3),
        "mrr": round(float((1 / ranks).mean()), 3),
        "ms/질문": round(per_query_ms, 1),
    }


def build(key: str, chunks: list[dict]) -> None:
    cfg = MODELS[key]
    model = TextEmbedding(cfg["name"], cache_dir=str(EMBED_CACHE))
    doc = embed(model, [passage_text(c) for c in chunks], cfg["passage"])
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", doc.astype(np.float32))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "info.json").write_text(json.dumps({"key": key, **cfg, "n_chunks": len(chunks)},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"인덱스 저장: {INDEX_DIR.relative_to(ROOT)} (청크 {len(chunks)}개, 모델 {cfg['name']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="후보 임베딩 모델 비교")
    parser.add_argument("--models", nargs="+", default=list(MODELS), help="비교할 모델 (기본: 전부)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="인덱스를 만들 모델")
    args = parser.parse_args()

    chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.compare:
        qas = [json.loads(l) for l in QA_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        qas = [q for q in qas if q["type"] in ("fact", "casual", "reason")]
        qas = random.Random(42).sample(qas, min(N_EVAL, len(qas)))
        print(f"질문 {len(qas)}개로 비교 (정답 근거 청크가 상위 k개 안에 들어오는 비율)")
        results = [evaluate(k, chunks, qas) for k in args.models]
        print()
        header = list(results[0])
        print(" | ".join(f"{h:>9}" for h in header))
        for r in results:
            print(" | ".join(f"{str(r[h]):>9}" for h in header))
        out = ROOT / "evaluation" / "results" / "retrieval_comparison.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n결과 저장: {out.relative_to(ROOT)}")
    else:
        build(args.model, chunks)


if __name__ == "__main__":
    main()
