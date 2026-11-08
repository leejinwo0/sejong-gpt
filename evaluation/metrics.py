"""세종 GPT 답변 자동 평가 지표.

베이스 모델 비교(학습 전)와 파인튜닝 후 평가에 똑같이 씁니다.
QA 검수(pipeline/filter_qa.py)와 같은 규칙을 재사용해서
"데이터를 거른 기준 = 모델을 평가하는 기준"이 되도록 했습니다.
"""
import re
from collections import defaultdict
from statistics import mean

from pipeline.filter_qa import (
    ABSTAIN, HANJA, MODERN_POLITE, QUOTED, TIME_BOUNDARY, WRONG_FIRST_PERSON, YEAR, normalize, source_years,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# 하오체·합쇼체 아닌 옛 말투 어미 (문장 끝 기준)
HAO_ENDING = re.compile(r"(오|소|라|구려|는가|던가|리까|노라|니라|구나|도다|세|게)[.!?…]*$")

ANSWERABLE = {"fact", "casual", "reason"}
SHOULD_ABSTAIN = {"unanswerable", "time_boundary"}


def rouge_l(pred: str, ref: str) -> float:
    """글자 단위 ROUGE-L F1 (공백·문장부호 제외)."""
    a, b = normalize(pred), normalize(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def score_one(pred: str, ref: str, qa_type: str, context_text: str) -> dict:
    sentences = [s for s in SENTENCE_SPLIT.split(pred.strip()) if s]
    abstained = bool(ABSTAIN.search(pred) or TIME_BOUNDARY.search(pred))
    years = [int(y) for y in YEAR.findall(pred)]
    scores = {
        "rouge_l": rouge_l(pred, ref),
        "uses_gwain": "과인" in pred,
        "no_hanja": not HANJA.search(pred),
        "no_modern_polite": not MODERN_POLITE.search(pred),
        "no_wrong_first_person": not WRONG_FIRST_PERSON.search(" " + QUOTED.sub(" ", pred)),
        "hao_ratio": mean(bool(HAO_ENDING.search(s)) for s in sentences) if sentences else 0.0,
        "length": len(pred),
    }
    if qa_type in ANSWERABLE:
        src = source_years(context_text)
        scores["years_correct"] = all(y in src for y in years) if years else None
        scores["abstain_correct"] = not abstained
    elif qa_type in SHOULD_ABSTAIN:
        scores["abstain_correct"] = abstained
    return scores


def summarize(rows: list[dict]) -> dict:
    """rows: [{"type":..., "scores": {...}}] → 전체·유형별 평균."""
    def agg(items):
        out = {}
        keys = {k for it in items for k in it["scores"]}
        for k in sorted(keys):
            vals = [it["scores"][k] for it in items if it["scores"].get(k) is not None]
            if vals:
                out[k] = round(mean(float(v) for v in vals), 3)
        out["n"] = len(items)
        return out

    by_type = defaultdict(list)
    for r in rows:
        group = "persona" if r["type"].startswith("persona_") else r["type"]
        by_type[group].append(r)
    return {"overall": agg(rows), **{t: agg(items) for t, items in sorted(by_type.items())}}
