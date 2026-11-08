"""문서를 검색·QA 생성용 청크로 나누기 (3단계).

data/raw/encykorea/articles.jsonl 의 문단들을 같은 섹션·소제목 안에서
적당한 길이로 묶어 data/chunks/chunks.jsonl 에 저장합니다.

- 항목마다 '개요 청크' 1개 (제목·정의·속성·요약)를 따로 만듭니다.
- 본문은 섹션 경계를 넘지 않도록 묶고, 너무 긴 문단은 문장 단위로 자릅니다.
- 각 청크 앞에 "[세종 > 세종의 업적 > 훈민정음의 창제]" 같은 위치 머리말을 붙여
  검색할 때 문맥을 잃지 않게 합니다.

실행 (저장소 루트에서):
    python -m pipeline.chunk
"""
import json
import re
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "encykorea" / "articles.jsonl"
OUTPUT = ROOT / "data" / "chunks" / "chunks.jsonl"

TARGET_CHARS = 600   # 이 길이를 넘으면 청크를 마감
MAX_CHARS = 900      # 한 문단이 이보다 길면 문장 단위로 쪼갬
MIN_CHARS = 150      # 이보다 짧은 마지막 조각은 앞 청크에 붙임
DROP_UNDER = 20      # 합칠 곳도 없는 이보다 짧은 본문 조각(예: "1책.")은 버림

# 세종과 무관한 동명 항목 (1차 파싱 결과 확인)
EXCLUDE_IDS = {
    "E0020922",  # 박연 (조선시대사) - 세종대 음악가 박연과 동명이인
    "E0037364",  # 영릉 (조선시대사) - 세종 영릉이 아님
}

SENTENCE_END = re.compile(r"(?<=[다요음함됨임]\.)\s+|(?<=[.?!])\s+(?=[가-힣A-Z“\"(])")


def split_long(text: str) -> list[str]:
    """MAX_CHARS보다 긴 문단을 문장 경계에서 TARGET_CHARS 안팎으로 자름."""
    if len(text) <= MAX_CHARS:
        return [text]
    pieces, buf = [], ""
    for sent in SENTENCE_END.split(text):
        if buf and len(buf) + len(sent) + 1 > TARGET_CHARS:
            pieces.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def header(title: str, section: str = "", subsection: str = "") -> str:
    return "[" + " > ".join(p for p in (title, section, subsection) if p) + "]"


def overview_chunk(art: dict) -> str:
    lines = [f"{art['title']}({art['hanja']})" if art.get("hanja") else art["title"]]
    if art.get("definition"):
        lines.append(f"정의: {art['definition']}")
    for group, items in art.get("attributes", {}).items():
        # "이칭: 이칭 OOO"처럼 그룹명과 항목명이 같으면 항목명 생략
        kv = ", ".join(v if k == group else f"{k} {v}" for k, v in items.items())
        lines.append(f"{group}: {kv}")
    if art.get("summary"):
        lines.append(f"요약: {art['summary']}")
    if art.get("keywords"):
        lines.append("키워드: " + ", ".join(art["keywords"]))
    return "\n".join(lines)


def group_paragraphs(paragraphs: list[dict]) -> list[tuple[str, str, str]]:
    """같은 (섹션, 소제목) 안에서 문단을 묶어 (섹션, 소제목, 본문) 목록 반환."""
    groups = []
    for para in paragraphs:
        key = (para["section"], para["subsection"])
        for piece in split_long(para["text"]):
            if groups and groups[-1][0] == key and len(groups[-1][1]) + len(piece) <= TARGET_CHARS:
                groups[-1][1] += "\n" + piece
            else:
                groups.append([key, piece])

    # 너무 짧은 조각은 같은 섹션의 앞 청크에 합침 (소제목이 달라도 섹션이 같으면 허용)
    merged = []
    for key, text in groups:
        if merged and merged[-1][0][0] == key[0] and len(text) < MIN_CHARS:
            merged[-1][1] += "\n" + text
        else:
            merged.append([key, text])
    return [(sec, sub, text) for (sec, sub), text in merged]


def build_chunks(art: dict) -> list[dict]:
    base = {
        "article_id": art["id"],
        "title": art["title"],
        "source": art["source"],
        "url": art["url"],
    }
    chunks = [{
        **base,
        "chunk_id": f"{art['id']}-00",
        "type": "overview",
        "section": "",
        "subsection": "",
        "header": header(art["title"], "개요"),
        "text": overview_chunk(art),
    }]
    body_groups = [g for g in group_paragraphs(art["paragraphs"]) if len(g[2]) >= DROP_UNDER]
    for n, (sec, sub, text) in enumerate(body_groups, 1):
        chunks.append({
            **base,
            "chunk_id": f"{art['id']}-{n:02d}",
            "type": "body",
            "section": sec,
            "subsection": sub,
            "header": header(art["title"], sec, sub),
            "text": text,
        })
    for c in chunks:
        c["char_len"] = len(c["text"])
        c["mentions_sejong"] = c["text"].count("세종")
    return chunks


def main() -> None:
    articles = [json.loads(l) for l in INPUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [a for a in articles if a["id"] not in EXCLUDE_IDS]
    print(f"항목 {len(articles)}개 중 {len(articles) - len(kept)}개 제외 → {len(kept)}개 사용")

    all_chunks = []
    for art in kept:
        all_chunks.extend(build_chunks(art))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    body = [c["char_len"] for c in all_chunks if c["type"] == "body"]
    print(f"청크 {len(all_chunks)}개 (개요 {len(all_chunks) - len(body)}개, 본문 {len(body)}개) → {OUTPUT.relative_to(ROOT)}")
    if body:
        print(f"본문 청크 길이: 평균 {mean(body):.0f}자, 중앙값 {median(body):.0f}자, 최소 {min(body)}자, 최대 {max(body)}자")
        print(f"'세종'이 들어간 본문 청크: {sum(1 for c in all_chunks if c['type'] == 'body' and c['mentions_sejong'])}개")

    sample = next((c for c in all_chunks if c["type"] == "body"), None)
    if sample:
        print("\n예시 청크:")
        print(sample["header"])
        print(sample["text"][:300] + ("..." if len(sample["text"]) > 300 else ""))


if __name__ == "__main__":
    main()
