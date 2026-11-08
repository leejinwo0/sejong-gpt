"""질문 하나를 받아 사료를 찾고, 세종대왕 답변을 만드는 핵심 흐름.

화면(main.py)과 분리해 두어서, 나중에 음성 입력이나 다른 화면에서도 그대로 씀.
"""
from dataclasses import dataclass
from typing import Iterator

from pipeline.persona import load_system_prompt
from pipeline.prompt_format import build_messages

SYSTEM_PROMPT = load_system_prompt()


@dataclass
class Retrieval:
    chunks: list[dict]       # 모델에 실제로 넘긴 사료
    candidates: list[dict]   # 검색된 사료 전체 (점수 확인용)


def retrieve(retriever, question: str, k: int = 3, min_score: float = 0.0) -> Retrieval:
    """상위 k개를 찾되, 가장 관련 높은 사료의 점수가 기준보다 낮으면 사료 없이 답하게 함.
    인사나 현대 질문처럼 사료가 필요 없는 질문에서 긴 사료를 읽느라 기다리는 시간을 없앰."""
    candidates = retriever.search(question, k=k)
    if not candidates or candidates[0]["score"] < min_score:
        return Retrieval(chunks=[], candidates=candidates)
    return Retrieval(chunks=candidates, candidates=candidates)


def answer_stream(llm, question: str, chunks: list[dict], temperature: float = 0.3) -> Iterator[str]:
    messages = build_messages(SYSTEM_PROMPT, question, chunks)
    return llm.stream(messages, temperature=temperature)
