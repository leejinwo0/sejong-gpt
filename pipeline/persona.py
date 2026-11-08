"""prompts/persona.md 에서 시스템 프롬프트와 상세 규칙을 읽어 오는 도우미.

앱, QA 생성, 학습 데이터 변환이 모두 이 함수를 써서
같은 페르소나 정의를 공유합니다.
"""
import re
from pathlib import Path

PERSONA_FILE = Path(__file__).resolve().parents[1] / "prompts" / "persona.md"


def _section(text: str, title: str) -> str:
    """'## 제목' 부터 다음 '## ' 직전까지의 내용."""
    m = re.search(rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not m:
        raise ValueError(f"persona.md에서 '## {title}' 섹션을 찾지 못했습니다.")
    return m.group(1).strip()


def load_system_prompt() -> str:
    """학습·실행 공용 짧은 시스템 프롬프트 (코드 블록 안의 문장)."""
    sec = _section(PERSONA_FILE.read_text(encoding="utf-8"), "시스템 프롬프트")
    m = re.search(r"```\s*\n(.*?)\n```", sec, re.S)
    if not m:
        raise ValueError("시스템 프롬프트 섹션에 코드 블록이 없습니다.")
    return m.group(1).strip()


def load_rules() -> str:
    """데이터 생성·검수용 상세 규칙 + 예시."""
    text = PERSONA_FILE.read_text(encoding="utf-8")
    return _section(text, "상세 규칙") + "\n\n" + _section(text, "예시")
