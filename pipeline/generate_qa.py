"""Gemini로 청크마다 세종대왕 QA 생성 (4단계).

data/chunks/chunks.jsonl 의 청크마다 질문과 답변을 만들어
data/qa/raw_qa.jsonl 에 이어 붙입니다. 중간에 멈춰도 다시 실행하면
이미 끝난 청크는 건너뛰고 이어서 진행합니다.

실행 (저장소 루트에서):
    python -m pipeline.generate_qa --limit 3     # 먼저 3개 청크로 시험
    python -m pipeline.generate_qa               # 전체 실행
"""
import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from pipeline.persona import load_rules

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
OUTPUT = ROOT / "data" / "qa" / "raw_qa.jsonl"
PROMPT_FILE = ROOT / "prompts" / "qa_generation.md"

# .env를 가장 먼저 읽어야 아래 --model 기본값에 GEMINI_MODEL이 반영됨
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "gemini-3.8-flash"
TEMPERATURE = 0.9         # 질문을 다양하게 뽑기 위해 약간 높게
REQUEST_INTERVAL = 6.0    # 요청 사이 최소 간격(초). 무료 한도에 걸리면 늘리기
MAX_RETRIES = 5

# 질문하는 사람 — 청크마다 하나를 골라 질문 말투를 다양하게 만듦
ASKERS = [
    "역사 숙제를 하는 초등학교 5학년 학생. 쉬운 말과 호기심 가득한 말투.",
    "한국사능력검정시험을 준비하는 대학생. 시험에 나올 만한 사실을 정확히 알고 싶어 함.",
    "조선사를 좋아하는 역사 덕후 직장인. 깊이 있는 이유와 배경을 궁금해함.",
    "한국어를 배우는 외국인 관광객. 조금 서툰 한국어로 짧게 물음.",
    "세종대왕을 존경하는 중년 시민. 공손하고 예의 바른 말투.",
    "장난스럽고 무례한 10대 사용자. 반말과 줄임말을 섞어 툭툭 던지듯 물음.",
]


class QA(BaseModel):
    type: Literal["fact", "reason", "casual", "unanswerable"]
    question: str
    answer: str
    evidence: str


def plan_counts(chunk: dict) -> dict:
    """청크 길이에 따라 질문 수 배분. 짧은 청크는 적게."""
    if chunk["type"] == "overview" or chunk["char_len"] < 250:
        return {"fact": 2, "reason": 0, "casual": 1, "unanswerable": 0}
    return {"fact": 2, "reason": 1, "casual": 1, "unanswerable": 1}


def build_prompt(template: str, rules: str, chunk: dict, asker: str, counts: dict) -> str:
    values = {
        "PERSONA_RULES": rules,
        "ASKER": asker,
        "N_TOTAL": str(sum(counts.values())),
        "N_FACT": str(counts["fact"]),
        "N_REASON": str(counts["reason"]),
        "N_CASUAL": str(counts["casual"]),
        "N_UNANSWERABLE": str(counts["unanswerable"]),
        "HEADER": chunk["header"],
        "CHUNK": chunk["text"],
    }
    prompt = template
    for key, val in values.items():
        prompt = prompt.replace("{{" + key + "}}", val)
    return prompt


def generate(client: genai.Client, model: str, prompt: str) -> list[QA]:
    """429(한도 초과)나 서버 오류면 기다렸다가 재시도."""
    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=list[QA],
    )
    wait = 20.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            if resp.parsed is None:
                raise ValueError(f"JSON 파싱 실패: {resp.text[:200] if resp.text else '(빈 응답)'}")
            return resp.parsed
        except errors.ClientError as e:
            if e.code != 429:
                raise  # 키 오류, 잘못된 모델명 등은 재시도해도 소용없음
            print(f"    한도 초과(429), {wait:.0f}초 대기 후 재시도 ({attempt}/{MAX_RETRIES})")
        except (errors.ServerError, ValueError) as e:
            print(f"    오류: {e} → {wait:.0f}초 대기 후 재시도 ({attempt}/{MAX_RETRIES})")
        time.sleep(wait)
        wait *= 2
    raise RuntimeError("재시도 횟수를 넘었습니다. 잠시 후 다시 실행하면 이어서 진행합니다.")


def load_done_ids() -> set[str]:
    if not OUTPUT.exists():
        return set()
    return {json.loads(l)["chunk_id"] for l in OUTPUT.read_text(encoding="utf-8").splitlines() if l.strip()}


def main(limit: int | None, model: str) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(".env에 GEMINI_API_KEY가 없습니다.")
    client = genai.Client(api_key=api_key)

    template = PROMPT_FILE.read_text(encoding="utf-8")
    rules = load_rules()
    chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = load_done_ids()
    todo = [c for c in chunks if c["chunk_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"전체 청크 {len(chunks)}개, 완료 {len(done)}개, 이번에 처리 {len(todo)}개 (모델: {model})")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    total_qa = 0
    for n, chunk in enumerate(todo, 1):
        rng = random.Random(chunk["chunk_id"])  # 청크마다 고정된 질문자 (재실행해도 같음)
        asker = rng.choice(ASKERS)
        counts = plan_counts(chunk)
        prompt = build_prompt(template, rules, chunk, asker, counts)

        started = time.time()
        qas = generate(client, model, prompt)

        with OUTPUT.open("a", encoding="utf-8") as f:
            for i, qa in enumerate(qas):
                f.write(json.dumps({
                    "qa_id": f"{chunk['chunk_id']}-q{i}",
                    "chunk_id": chunk["chunk_id"],
                    "article_id": chunk["article_id"],
                    "type": qa.type,
                    "asker": asker,
                    "question": qa.question.strip(),
                    "answer": qa.answer.strip(),
                    "evidence": qa.evidence.strip(),
                    "model": model,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }, ensure_ascii=False) + "\n")
        total_qa += len(qas)
        print(f"[{n}/{len(todo)}] {chunk['chunk_id']} {chunk['header']} → QA {len(qas)}개")

        elapsed = time.time() - started
        if n < len(todo) and elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

    print(f"\n완료: 이번에 QA {total_qa}개 생성 → {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="처리할 청크 수 (시험용)")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()
    main(args.limit, args.model)
