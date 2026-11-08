"""검수를 통과한 QA를 학습용 대화 데이터로 변환 (6단계).

QA마다 [시스템 프롬프트 + 참고 사료 + 질문 → 답변] 형태로 만들고
train / val / test 로 나눠 data/dataset/ 에 저장합니다.

참고 사료에는 정답 근거 청크 하나와 '방해 청크' 몇 개를 섞어 넣습니다.
실제 앱의 검색 결과에도 관련 없는 문단이 섞여 오기 때문에,
모델이 여러 사료 중 진짜 근거를 골라 답하는 법을 배우게 하려는 것입니다.

실행 (저장소 루트에서):
    python -m pipeline.build_dataset
"""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from pipeline.persona import load_system_prompt
from pipeline.prompt_format import build_messages

ROOT = Path(__file__).resolve().parents[1]
QA_FILE = ROOT / "data" / "qa" / "filtered_qa.jsonl"
PERSONA_FILE = ROOT / "data" / "qa" / "persona_qa.jsonl"   # 없으면 건너뜀
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
OUT_DIR = ROOT / "data" / "dataset"

SEED = 42
N_DISTRACTORS = 2            # 근거 청크 외에 섞을 방해 청크 수
SPLIT = {"val": 0.05, "test": 0.05}   # 나머지는 train


# '한글'은 20세기 초에 생긴 이름이라 세종의 입에서 나오면 시대에 맞지 않음.
# 받침이 같아(ㄹ/ㅁ 모두 받침 있음) 조사를 고치지 않고 바꿔도 문장이 자연스러움. '한글날'은 제외.
ANACHRONISMS = [(re.compile(r"한글(?!날)"), "훈민정음")]


def fix_anachronisms(answer: str) -> tuple[str, int]:
    count = 0
    for pattern, repl in ANACHRONISMS:
        answer, n = pattern.subn(repl, answer)
        count += n
    return answer, count


def split_of(chunk_id: str) -> str:
    """청크 단위로 고정 분할. 같은 청크에서 나온 QA는 모두 같은 세트에 들어가서
    학습에 쓴 사료로 평가하는 일(정보 누수)을 막음. 해시라 다시 실행해도 결과가 같음."""
    h = int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % 10000 / 10000
    if h < SPLIT["test"]:
        return "test"
    if h < SPLIT["test"] + SPLIT["val"]:
        return "val"
    return "train"


def pick_distractors(qa: dict, gold: dict, chunks_by_article: dict, all_chunks: list, rng: random.Random) -> list[dict]:
    """방해 청크 고르기.
    - 답이 있는 질문: 같은 항목의 다른 청크 1개(주제가 비슷해서 헷갈림) + 다른 항목 청크
    - 답이 없는 질문: 다른 항목 청크만 (같은 항목 다른 청크에 우연히 답이 있을 수 있으므로)"""
    picks = []
    if qa["type"] not in ("unanswerable", "time_boundary"):
        same = [c for c in chunks_by_article[gold["article_id"]] if c["chunk_id"] != gold["chunk_id"]]
        if same:
            picks.append(rng.choice(same))
    others = [c for c in all_chunks if c["article_id"] != gold["article_id"]]
    while len(picks) < N_DISTRACTORS:
        c = rng.choice(others)
        if c not in picks:
            picks.append(c)
    return picks


def load_persona() -> list[dict]:
    if not PERSONA_FILE.exists():
        return []
    return [json.loads(l) for l in PERSONA_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    rng = random.Random(SEED)
    system_prompt = load_system_prompt()
    all_chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunk_map = {c["chunk_id"]: c for c in all_chunks}
    chunks_by_article = defaultdict(list)
    for c in all_chunks:
        chunks_by_article[c["article_id"]].append(c)
    qas = [json.loads(l) for l in QA_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = defaultdict(list)
    fixed = 0
    for qa in qas + (persona if (persona := load_persona()) else []):
        qa["answer"], n = fix_anachronisms(qa["answer"])
        fixed += n
    for qa in qas:
        gold = chunk_map[qa["chunk_id"]]
        context = [gold] + pick_distractors(qa, gold, chunks_by_article, all_chunks, rng)
        rng.shuffle(context)  # 근거가 항상 첫 번째에 오지 않도록
        split = split_of(qa["chunk_id"])
        rows[split].append({
            "id": qa["qa_id"],
            "type": qa["type"],
            "messages": build_messages(system_prompt, qa["question"], context, qa["answer"]),
            "gold_chunk_id": gold["chunk_id"],
            "context_chunk_ids": [c["chunk_id"] for c in context],
        })

    # 페르소나 QA: 근거 사료가 없으므로 무관한 청크 2개만 붙임.
    # 앱은 어떤 질문이든 검색 결과를 붙여 보내므로, 관련 없는 사료가 와도 무시하고 답하는 법을 익히게 함.
    if persona:
        for qa in persona:
            context = rng.sample(all_chunks, N_DISTRACTORS)
            rows[split_of(qa["qa_id"])].append({
                "id": qa["qa_id"],
                "type": f"persona_{qa['type']}",
                "messages": build_messages(system_prompt, qa["question"], context, qa["answer"]),
                "gold_chunk_id": None,
                "context_chunk_ids": [c["chunk_id"] for c in context],
            })
        print(f"페르소나 QA {len(persona)}개 포함")
    else:
        print("페르소나 QA 파일이 없어 사료 QA만 사용")

    print(f"답변 속 '한글' → '훈민정음' 교체: {fixed}곳")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        path = OUT_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        types = Counter("persona" if r["type"].startswith("persona_") else r["type"] for r in rows[split])
        lengths = [sum(len(m["content"]) for m in r["messages"]) for r in rows[split]]
        avg = f"평균 {mean(lengths):.0f}자, 최대 {max(lengths)}자" if lengths else "-"
        print(f"{split:5s}: {len(rows[split]):4d}개 ({avg}) {dict(types)}")

    sample = rows["train"][0] if rows["train"] else None
    if sample:
        print("\n===== 예시 =====")
        for m in sample["messages"]:
            print(f"--- {m['role']} ---")
            print(m["content"][:600] + ("..." if len(m["content"]) > 600 else ""))


if __name__ == "__main__":
    main()
