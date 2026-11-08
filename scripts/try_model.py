"""학습한 GGUF 모델을 터미널에서 바로 시험해 보는 스크립트.

1) test 세트의 실제 질문 2개 (사료 포함)
2) 사료 없이 묻는 인사·현대 질문 2개
를 차례로 물어보고, 답변을 글자가 나오는 대로 보여 주며 속도(토큰/초)를 잽니다.

실행 (저장소 루트에서):
    python -m scripts.try_model
    python -m scripts.try_model "직접 묻고 싶은 질문"
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from llama_cpp import Llama

from pipeline.persona import load_system_prompt
from pipeline.prompt_format import build_messages

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_PATH = ROOT / os.getenv("MODEL_PATH", "models/sejong-kanana-2.1b-Q4_K_M.gguf")


def ask(llm: Llama, system_prompt: str, question: str, chunks: list[dict]) -> None:
    messages = build_messages(system_prompt, question, chunks)
    print(f"\nQ: {question}")
    if chunks:
        print("   (참고 사료: " + ", ".join(c["header"].strip("[]") for c in chunks) + ")")
    print("A: ", end="", flush=True)

    start, first_token_at, n_tokens = time.time(), None, 0
    for part in llm.create_chat_completion(
        messages=messages, max_tokens=300, temperature=0.3, repeat_penalty=1.1, stream=True,
    ):
        text = part["choices"][0]["delta"].get("content", "")
        if text:
            if first_token_at is None:
                first_token_at = time.time()
            n_tokens += 1
            print(text, end="", flush=True)
    end = time.time()
    if first_token_at:
        speed = n_tokens / max(end - first_token_at, 1e-6)
        print(f"\n   ⏱ 첫 글자까지 {first_token_at - start:.1f}초, 생성 {speed:.1f} 토큰/초")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"모델 파일이 없습니다: {MODEL_PATH}\n.env의 MODEL_PATH를 확인하세요.")

    print(f"모델 불러오는 중: {MODEL_PATH.name}")
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
    system_prompt = load_system_prompt()

    if len(sys.argv) > 1:
        ask(llm, system_prompt, " ".join(sys.argv[1:]), [])
        return

    chunks = {c["chunk_id"]: c for c in map(json.loads, open(ROOT / "data/chunks/chunks.jsonl", encoding="utf-8"))}
    test = [json.loads(l) for l in open(ROOT / "data/dataset/test.jsonl", encoding="utf-8")]
    samples = [r for r in test if r["type"] == "fact"][:1] + [r for r in test if r["type"] == "unanswerable"][:1]
    for r in samples:
        question = r["messages"][1]["content"].split("[질문]")[-1].strip()
        ask(llm, system_prompt, question, [chunks[c] for c in r["context_chunk_ids"]])

    for q in ["안녕하세요, 세종대왕님!", "스마트폰이라는 걸 아세요?"]:
        ask(llm, system_prompt, q, [])


if __name__ == "__main__":
    main()
