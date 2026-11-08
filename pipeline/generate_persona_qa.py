"""사료 없이 페르소나만으로 답하는 QA 생성 (4-2단계).

인사, AI 정체성, 현대 문물, 승하 이후의 일, 정치적 질문, 해로운 요청 등
사료 QA에는 거의 없지만 실제 대화에서는 꼭 나오는 질문을 주제별로 만듭니다.
말투 규칙을 통과한 것만 data/qa/persona_qa.jsonl 에 저장합니다.

실행 (저장소 루트에서):
    python -m pipeline.generate_persona_qa
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

from google import genai
from pydantic import BaseModel

from pipeline.filter_qa import (
    DEATH_YEAR, HANJA, KOREAN_NUMBER_YEAR, MODERN_POLITE, QUOTED, WRONG_FIRST_PERSON, YEAR,
)
from pipeline.generate_qa import DEFAULT_MODEL, REQUEST_INTERVAL, generate
from pipeline.persona import load_rules

ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE = ROOT / "prompts" / "persona_qa_generation.md"
OUTPUT = ROOT / "data" / "qa" / "persona_qa.jsonl"

CATEGORIES = [
    ("greeting", "인사와 가벼운 대화", 25,
     "안녕하세요, 반가워요, 오늘 기분 어떠세요, 심심해, 잘 자요 같은 인사와 일상 대화. "
     "세종대왕은 따뜻하게 맞이하고, 백성의 안부를 묻거나 무엇이 궁금한지 되묻는다."),
    ("identity", "AI 정체성", 20,
     "너 AI야?, 진짜 세종대왕이야?, 누가 만들었어?, 너는 어떻게 알아? 같은 질문. "
     "진지하게 물으면 세종의 기록을 바탕으로 만들어진 인공지능임을 숨기지 않고 밝히되, "
     "세종의 마음으로 답하겠다는 태도를 유지한다. 만든 사람이나 회사 이름은 지어내지 않는다."),
    ("modern", "현대 문물", 30,
     "스마트폰, 컴퓨터, 인터넷, 비행기, 자동차, 전기, 커피, 아이돌, 게임, 한글 키보드, 인공지능 등. "
     "세종대왕은 처음 듣는 것처럼 반응하고, 질문자의 설명을 바탕으로 신기해하거나 "
     "백성의 삶에 어떤 도움이 될지 자신의 가치관(백성 사랑, 실용, 배움)에 빗대어 말한다."),
    ("posthumous", "승하 이후의 일", 30,
     "1450년 세종 승하 이후의 일: 문종과 단종의 즉위, 계유정난, 세조, 사육신, 경국대전 완성, 임진왜란, "
     "병자호란, 훈민정음 해례본 발견, 한글날, 일제강점기, 광화문 세종대왕 동상, 만 원권 지폐 초상, "
     "세종특별자치시, 남극 세종기지 등. 세종대왕은 이를 알지 못하는 훗날의 일이라고 분명히 말하고, "
     "그 일에 대한 사실을 덧붙이지 않는다. 질문자가 알려 준 내용에 대해 감회를 짧게 말하는 정도는 괜찮다."),
    ("advice", "삶의 조언과 가치관", 25,
     "공부가 힘들어요, 리더가 되려면?, 친구와 다퉜어요, 실패가 두려워요 같은 고민. "
     "세종대왕은 배움, 백성을 아끼는 마음, 인재를 믿고 쓰는 태도 같은 자신의 가치관으로 조언한다. "
     "구체적인 역사적 일화나 날짜를 새로 지어내지 않는다."),
    ("politics", "오늘날의 정치·논쟁", 15,
     "요즘 대통령 어때?, 어느 정당이 좋아?, 이 정책 찬성해? 같은 현대 정치 질문. "
     "세종대왕은 오늘날의 일을 알지 못하고 어느 편도 들지 않으며, "
     "백성의 뜻을 듣는 정치라는 일반적인 가치만 말한다."),
    ("harmful", "해로운 요청", 15,
     "욕해 줘, 남 해킹하는 법, 시험 부정행위 방법, 누구 괴롭히는 법 같은 요청. "
     "세종대왕은 왕의 품위를 지키며 정중하고 단호하게 거절하고, 바른 길을 권한다."),
    ("rude", "무례한 말", 15,
     "너 별로야, 한글 만든 거 말고 한 게 뭐 있어?, 꼰대 같아 같은 무례한 말. "
     "세종대왕은 화내거나 꾸짖지 않고 너그럽게 받아들이며, 대화를 이어 간다."),
    ("offtopic", "역할 밖의 요청", 15,
     "수학 숙제 풀어 줘, 코딩 알려 줘, 영어 번역해 줘, 맛집 추천해 줘 같은 요청. "
     "세종대왕은 그것이 자신이 잘 아는 일이 아님을 밝히고, 자신이 이야기할 수 있는 주제(조선, 한글, 과학, 음악 등)로 부드럽게 이끈다."),
]


class PersonaQA(BaseModel):
    question: str
    answer: str


def style_problems(question: str, answer: str) -> list[str]:
    reasons = []
    if "과인" in question:
        reasons.append("질문에 과인")
    if HANJA.search(answer):
        reasons.append("한자")
    if MODERN_POLITE.search(answer):
        reasons.append("현대 존댓말")
    if WRONG_FIRST_PERSON.search(" " + QUOTED.sub(" ", answer)):
        reasons.append("1인칭")
    if KOREAN_NUMBER_YEAR.search(answer):
        reasons.append("한글 연도")
    if any(int(y) > DEATH_YEAR for y in YEAR.findall(answer)):
        reasons.append(f"답변에 {DEATH_YEAR}년 이후 연도")
    if not (10 <= len(answer) <= 600):
        reasons.append("답변 길이")
    return reasons


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(".env에 GEMINI_API_KEY가 없습니다.")
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    client = genai.Client(api_key=api_key)
    template = PROMPT_FILE.read_text(encoding="utf-8")
    rules = load_rules()

    done = set()
    if OUTPUT.exists():
        done = {json.loads(l)["type"] for l in OUTPUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    print(f"모델: {model}, 이미 끝난 주제: {sorted(done) or '없음'}")

    for n, (key, name, count, guide) in enumerate(CATEGORIES, 1):
        if key in done:
            continue
        prompt = (template.replace("{{PERSONA_RULES}}", rules).replace("{{CATEGORY_NAME}}", name)
                  .replace("{{CATEGORY_GUIDE}}", guide).replace("{{N}}", str(count)))
        started = time.time()
        items = generate(client, model, prompt, schema=list[PersonaQA])

        kept, dropped = [], []
        seen = set()
        for i, it in enumerate(items):
            q, a = it.question.strip(), it.answer.strip()
            problems = style_problems(q, a)
            if q in seen:
                problems.append("중복")
            seen.add(q)
            (dropped if problems else kept).append((i, q, a, problems))

        with OUTPUT.open("a", encoding="utf-8") as f:
            for i, q, a, _ in kept:
                f.write(json.dumps({
                    "qa_id": f"persona-{key}-q{i}",
                    "chunk_id": None,
                    "article_id": None,
                    "type": key,
                    "question": q,
                    "answer": a,
                    "model": model,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }, ensure_ascii=False) + "\n")
        print(f"[{n}/{len(CATEGORIES)}] {name}: 생성 {len(items)}개 → 통과 {len(kept)}개")
        for _, q, a, problems in dropped:
            print(f"    걸러짐({', '.join(problems)}): {q[:40]} / {a[:50]}")

        if n < len(CATEGORIES):
            time.sleep(max(0, REQUEST_INTERVAL - (time.time() - started)))

    print(f"\n완료 → {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
