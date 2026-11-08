"""학습한 세종 GGUF 모델 불러오기와 답변 생성."""
import os
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from llama_cpp import Llama

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class SejongLLM:
    def __init__(self, model_path: str | None = None):
        path = ROOT / (model_path or os.getenv("MODEL_PATH", "models/sejong-kanana-2.1b-Q4_K_M.gguf"))
        if not path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {path} (.env의 MODEL_PATH 확인)")
        self.name = path.name
        self.llm = Llama(model_path=str(path), n_ctx=4096, n_threads=os.cpu_count(), verbose=False)

    def stream(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 300) -> Iterator[str]:
        """답변을 글자가 만들어지는 대로 하나씩 내보냄."""
        for part in self.llm.create_chat_completion(
            messages=messages, temperature=temperature, max_tokens=max_tokens,
            repeat_penalty=1.1, stream=True,
        ):
            text = part["choices"][0]["delta"].get("content")
            if text:
                yield text
