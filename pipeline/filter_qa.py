"""생성된 QA 자동 검수 (5단계).

data/qa/raw_qa.jsonl 을 규칙으로 검사해서
- 통과한 QA → data/qa/filtered_qa.jsonl
- 걸러진 QA → data/qa/rejected_qa.jsonl (걸린 이유 포함)
으로 나눕니다. 걸러진 항목은 직접 읽어 보고 규칙이 너무 엄격한지 확인하는 데 씁니다.

실행 (저장소 루트에서):
    python -m pipeline.filter_qa
"""
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "qa" / "raw_qa.jsonl"
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
KEPT = ROOT / "data" / "qa" / "filtered_qa.jsonl"
REJECTED = ROOT / "data" / "qa" / "rejected_qa.jsonl"

DEATH_YEAR = 1450
DUP_Q_THRESHOLD = 0.6      # 질문끼리 이 이상 비슷하고
DUP_A_THRESHOLD = 0.5      # 답변끼리도 이 이상 비슷하면 중복으로 봄 (0~1)
EVIDENCE_THRESHOLD = 0.8   # 근거 문장이 사료와 이 비율 이상 겹쳐야 통과

HANJA = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
MODERN_POLITE = re.compile(r"(습니다|입니다|습니까|입니까|에요|예요|해요|세요|어요|아요|죠)\s*[.!?]?(\s|$)")
# '나는/내가'는 "해주에서 나는 기장"처럼 동사로도 쓰이므로 문장 첫머리에 올 때만 1인칭으로 봄
WRONG_FIRST_PERSON = re.compile(r"(^|\s)(짐은|짐이|짐을|짐의|저는|제가|저의)\s|(^|[.!?,])\s*(나는|내가)\s")
QUESTION_FORBIDDEN = re.compile(r"과인|사료|기록에 따르면|기록에 보면|기록상")
KOREAN_NUMBER_YEAR = re.compile(r"(일)?천\s?[이삼사오육칠팔구]?백")
POSTHUMOUS = re.compile(r"시호|묘호|능호|영릉|세종대왕면|계유정난|세조|단종")
YEAR = re.compile(r"(?<!\d)(1[0-9]{3})(?=년)")
# "백성이 글을 알지 못하여"처럼 평범한 문장에 걸리지 않도록, 모른다는 뜻이 분명한 표현만
ABSTAIN = re.compile(r"확인하기 어렵|기억하는 기록|기록에서 찾|기록에 남아 있지 않|알 길이 없|알 수 없소|알기 어렵|꼽기는? 어렵|모르겠|과인[이은] 알지 못|알지 못하는 기록|전하기는 어렵|기억하기 어렵")
# 승하 이후의 일이라 모른다고 한 답 (시간 경계를 지킨 좋은 예시)
TIME_BOUNDARY = re.compile(r"훗날의 일|세상을 떠난 뒤|승하한 뒤|과인이 떠난 뒤")
QUOTED = re.compile(r"[“\"‘'][^”\"’']*[”\"’']")


# 사료의 "세종 2년", "태종 8" 같은 왕력 표기를 서기로 바꾸기 위한 기준 (원년 = 기준 + 1)
REIGN_BASE = {"태조": 1391, "정종": 1398, "태종": 1400, "세종": 1418, "문종": 1450}
REIGN_YEAR = re.compile(r"(" + "|".join(REIGN_BASE) + r")\s?(\d{1,2})(?=\s?년|\)|,|\s|\s?[∼~-])")
REIGN_RANGE = re.compile(r"(" + "|".join(REIGN_BASE) + r")\s?(\d{1,2})\s?[∼~-]\s?(\d{1,2})")


def source_years(text: str) -> set[int]:
    """사료에 나오는 서기 연도 + 왕력 표기를 서기로 바꾼 연도.
    예: "세종 2년 3월" → 1420, "1408년(태종 8)" → 1408"""
    # "1405∼1406년"처럼 '년'이 바로 붙지 않은 연도도 포함
    years = {int(y) for y in re.findall(r"(?<!\d)(1[0-9]{3})(?!\d)", text)}
    years |= {REIGN_BASE[king] + int(n) for king, n in REIGN_YEAR.findall(text)}
    # "태종 5∼6년", "1411∼1413년" 같은 범위는 사이 연도까지 모두 포함
    for king, a, b in REIGN_RANGE.findall(text):
        years |= {REIGN_BASE[king] + n for n in range(int(a), int(b) + 1)}
    for a, b in re.findall(r"(?<!\d)(1[0-9]{3})\s?[∼~-]\s?(1[0-9]{3})(?!\d)", text):
        years |= set(range(int(a), int(b) + 1))
    # "세종 26년에 광평대군, 그 이듬해에 평원대군"처럼 이듬해로 쓴 경우
    if "이듬해" in text:
        years |= {y + 1 for y in years}
    if "즉위" in text:
        years.add(1418)
    return years


def normalize(text: str) -> str:
    """비교용: 공백·문장부호·한자를 지움."""
    text = HANJA.sub("", text)
    return re.sub(r"[\s\W_]+", "", text)


def evidence_coverage(evidence: str, chunk_text: str) -> float:
    """근거 문장 중 사료에 그대로 있는 글자의 비율."""
    ev, src = normalize(evidence), normalize(chunk_text)
    if not ev:
        return 0.0
    if ev in src:
        return 1.0
    matcher = SequenceMatcher(None, ev, src, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(ev)


def check(qa: dict, chunk: dict) -> list[str]:
    reasons = []
    q, a, ev, kind = qa["question"], qa["answer"], qa["evidence"], qa["type"]

    # 형식
    if len(q) < 5:
        reasons.append("질문이 너무 짧음")
    if len(a) < 15:
        reasons.append("답변이 너무 짧음")
    if len(a) > 600:
        reasons.append("답변이 너무 김")

    # 질문 규칙
    if QUESTION_FORBIDDEN.search(q):
        reasons.append("질문에 금지어(과인/사료/기록)")

    # 말투 규칙
    if HANJA.search(a):
        reasons.append("답변에 한자")
    if MODERN_POLITE.search(a):
        reasons.append("답변에 현대 존댓말")
    # 인용문(태종의 말 등) 속 '내가', '나는'은 제외하고 검사
    if WRONG_FIRST_PERSON.search(" " + QUOTED.sub(" ", a)):
        reasons.append("답변 1인칭이 과인이 아님")
    if KOREAN_NUMBER_YEAR.search(a):
        reasons.append("연도를 한글로 씀")

    # 시간 경계: 승하 이후 연도나 사후 내용
    years_q = [int(y) for y in YEAR.findall(q)]
    years_a = [int(y) for y in YEAR.findall(a)]
    # 사후의 일을 물었고 답은 "훗날의 일이라 모른다"고 한 경우는 질문 쪽 사후 언급을 문제 삼지 않음
    is_time_boundary = bool(TIME_BOUNDARY.search(a)) and kind != "unanswerable"
    checked_years = years_a if is_time_boundary else years_q + years_a
    if any(y > DEATH_YEAR for y in checked_years):
        reasons.append(f"{DEATH_YEAR}년 이후 연도 언급")
    if POSTHUMOUS.search(a) or (not is_time_boundary and POSTHUMOUS.search(q)):
        reasons.append("사후 내용 언급")
    if reasons and is_time_boundary:
        return reasons

    # 근거 규칙
    if is_time_boundary:
        # 승하 이후 일이라 모른다고 답한 경우 → 시간 경계 예시로 살림 (사료 대조는 하지 않음)
        qa["type"] = "time_boundary"
        return reasons
    if kind == "unanswerable":
        if not ABSTAIN.search(a):
            reasons.append("답이 없는 질문인데 모른다고 하지 않음")
    else:
        if ABSTAIN.search(a):
            reasons.append("답이 있는 질문인데 모른다고 함")
        if evidence_coverage(ev, chunk["text"]) < EVIDENCE_THRESHOLD:
            reasons.append("근거 문장이 사료에 없음")
        src_years = source_years(chunk["text"])
        if any(y not in src_years for y in years_a):
            reasons.append("사료에 없는 연도")

    return reasons


def char_bigrams(text: str) -> set[str]:
    t = normalize(text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def find_duplicates(qas: list[dict]) -> set[str]:
    """질문과 답변이 둘 다 비슷한 QA 묶음에서 첫 번째만 남기고 나머지 qa_id 반환.
    질문만 보면 "언제 태어났어?"와 "어디서 태어났어?"가 중복으로 잘못 잡히므로 답변도 함께 비교.
    같은 항목(article) 안에서만 비교해서 속도를 지킴."""
    dup_ids = set()
    by_article = defaultdict(list)
    for qa in qas:
        by_article[qa["article_id"]].append(qa)
    for group in by_article.values():
        kept: list[tuple[set[str], set[str]]] = []
        for qa in group:
            qg, ag = char_bigrams(qa["question"]), char_bigrams(qa["answer"])
            if any(jaccard(qg, kq) >= DUP_Q_THRESHOLD and jaccard(ag, ka) >= DUP_A_THRESHOLD for kq, ka in kept):
                dup_ids.add(qa["qa_id"])
            else:
                kept.append((qg, ag))
    return dup_ids


def main() -> None:
    chunks = {c["chunk_id"]: c for c in map(json.loads, CHUNKS.read_text(encoding="utf-8").splitlines()) if c}
    raw = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]

    kept, rejected = [], []
    reason_counter = Counter()
    for qa in raw:
        reasons = check(qa, chunks[qa["chunk_id"]])
        if reasons:
            rejected.append({**qa, "reasons": reasons})
            reason_counter.update(reasons)
        else:
            kept.append(qa)

    dup_ids = find_duplicates(kept)
    for qa in [q for q in kept if q["qa_id"] in dup_ids]:
        rejected.append({**qa, "reasons": ["비슷한 질문 중복"]})
        reason_counter["비슷한 질문 중복"] += 1
    kept = [q for q in kept if q["qa_id"] not in dup_ids]

    for path, rows in ((KEPT, kept), (REJECTED, rejected)):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"전체 {len(raw)}개 → 통과 {len(kept)}개 ({len(kept) / len(raw):.0%}), 걸러짐 {len(rejected)}개")
    print("\n[걸러진 이유] (한 QA가 여러 이유에 걸릴 수 있음)")
    for reason, n in reason_counter.most_common():
        print(f"  {reason}: {n}")

    print("\n[통과한 QA 유형]")
    for kind, n in Counter(q["type"] for q in kept).most_common():
        print(f"  {kind}: {n}")

    print("\n[이유별 예시] 규칙이 너무 엄격하지 않은지 확인용")
    shown = Counter()
    for r in rejected:
        main_reason = r["reasons"][0]
        if shown[main_reason] >= 2:
            continue
        shown[main_reason] += 1
        print(f"\n  ({main_reason}) {r['qa_id']}")
        print(f"   Q: {r['question'][:70]}")
        print(f"   A: {r['answer'][:100]}")


if __name__ == "__main__":
    main()
