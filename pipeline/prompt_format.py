"""모델에 넣는 입력 형식 (학습 데이터와 앱이 함께 사용).

학습 때와 실행 때 입력 모양이 조금만 달라도 모델 성능이 떨어지므로,
사료와 질문을 묶는 방법은 반드시 이 함수 하나로만 만듭니다.
"""


def format_context(chunks: list[dict]) -> str:
    """검색된 사료 청크들을 번호 붙인 블록으로."""
    blocks = [f"[{i}] {c['header'].strip('[]')}\n{c['text']}" for i, c in enumerate(chunks, 1)]
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return question
    return f"[참고 사료]\n{format_context(chunks)}\n\n[질문]\n{question}"


def build_messages(system_prompt: str, question: str, chunks: list[dict], answer: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_message(question, chunks)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
