"""한국민족문화대백과사전 HTML → 구조화된 문서(JSONL) 변환 (2단계).

encykorea.py로 받아 둔 HTML에서 제목·분류·요약·본문 문단·관련 항목을 뽑아
data/raw/encykorea/articles.jsonl 에 한 줄에 한 항목씩 저장합니다.

실행 (저장소 루트에서):
    python -m pipeline.collect.encykorea_parse
"""
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "encykorea"
MANIFEST = RAW_DIR / "manifest.jsonl"
OUTPUT = RAW_DIR / "articles.jsonl"

ARTICLE_LINK = re.compile(r"^/Article/(E\d{7})")
H2_MARK = "§§H2§§"
ROW_SEP = "§§ROW§§"  # 표의 행 구분 (표 하나를 문단 하나로 유지하기 위한 임시 표시)
LICENSE_NOTE = "한국민족문화대백과사전은 공공저작물로서 공공누리 제도에 따라 이용 가능"


def clean(text: str) -> str:
    """연속 공백을 하나로 줄이고 앞뒤 공백 제거."""
    return re.sub(r"\s+", " ", text or "").strip()


def text_of(node) -> str:
    return clean(node.get_text(" ")) if node else ""


# ---------- 메타 정보 ----------

def parse_meta(soup: BeautifulSoup) -> dict:
    meta = soup.select_one("#meta")
    title = text_of(meta.select_one(".meta_head .subject h3")) if meta else ""

    # 한자 표기: PC용 버튼의 직접 텍스트만 (툴팁 내부 텍스트 제외)
    hanja = ""
    btn = meta.select_one(".origin.pc button.origin_tooltip") if meta else None
    if btn:
        hanja = clean("".join(s for s in btn.strings if s.parent is btn))

    categories = [text_of(li) for li in soup.select(".category ul li")]
    definition = text_of(soup.select_one(".category .head_cont")) or text_of(soup.select_one("#defi .detail"))

    attributes = {}
    for block in soup.select(".property .info_block"):
        if "author" in (block.get("class") or []):
            continue
        group = text_of(block.select_one(".tit"))
        items = {}
        for li in block.select("ul.list li"):
            key, val = text_of(li.select_one("strong")), text_of(li.select_one("span"))
            if key:
                items[key] = val
        if group and items:
            attributes[group] = items

    author = {}
    info = soup.select_one(".author_info")
    if info:
        author = {
            "name": text_of(info.select_one("li.author a")),
            "affiliation": text_of(info.select_one("li.author span")).strip("()"),
            "written": text_of(info.select_one("li.date_first")).replace("집필", "").strip(),
            "updated": text_of(info.select_one("li.date_last")).replace("최종수정", "").strip(),
        }

    return {
        "title": title,
        "hanja": hanja,
        "categories": categories,
        "definition": definition,
        "attributes": attributes,
        "author": author,
    }


def parse_summary(soup: BeautifulSoup) -> tuple[str, list[str]]:
    box = soup.select_one("#summary .detail")
    if not box:
        return "", []
    keywords = [text_of(a) for a in box.select(".detail_keyword a")]
    paragraphs = [text_of(p) for p in box.find_all("p", recursive=False)]
    return " ".join(p for p in paragraphs if p), keywords


def parse_related(soup: BeautifulSoup) -> list[dict]:
    related = []
    for a in soup.select(".related_li a[href]"):
        m = ARTICLE_LINK.match(a["href"])
        if m:
            related.append({
                "id": m.group(1),
                "title": text_of(a.select_one("strong")),
                "desc": text_of(a.select_one(".desc")),
            })
    return related


# ---------- 본문 ----------

def table_to_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [text_of(c) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n" + ROW_SEP.join(rows) + "\n"


def detail_to_paragraphs(detail: Tag) -> list[tuple[str, str]]:
    """섹션의 .detail 영역을 (소제목, 문단) 목록으로 변환."""
    # 본문이 아닌 요소 제거: 주석 번호 버튼, 접기 버튼, 스크립트
    for el in detail.select("button, script, style"):
        el.decompose()
    # 이미지·PDF 등 미디어는 문단 경계 역할을 하므로 줄바꿈으로 대체
    for el in detail.select(".detail_cont_media_wrap, figure, iframe"):
        el.replace_with(NavigableString("\n"))
    for table in detail.find_all("table"):
        table.replace_with(NavigableString(table_to_text(table)))
    for h2 in detail.find_all("h2"):
        h2.replace_with(NavigableString(f"\n{H2_MARK}{text_of(h2)}\n"))
    # 블록 요소 경계에 줄바꿈을 넣어 문단이 붙지 않게 함 (링크 같은 인라인 요소는 그대로)
    for el in detail.find_all(["p", "br", "li", "div", "ul", "ol"]):
        el.insert_before(NavigableString("\n"))
        el.insert_after(NavigableString("\n"))

    result, subsection = [], ""
    for line in detail.get_text().split("\n"):
        line = clean(line)
        if not line:
            continue
        if line.startswith(H2_MARK):
            subsection = line[len(H2_MARK):]
        else:
            result.append((subsection, line.replace(ROW_SEP, "\n")))
    return result


def parse_body(soup: BeautifulSoup) -> tuple[list[dict], list[str]]:
    body = soup.select_one("#body_content")
    if not body:
        return [], []

    linked = []
    for a in body.select("a[href]"):
        m = ARTICLE_LINK.match(a["href"])
        if m:
            linked.append(m.group(1))

    paragraphs = []
    for section in body.select("section.content_section"):
        heading = text_of(section.select_one(".section_tit h3"))
        detail = section.select_one(".detail")
        if not detail:
            continue
        for subsection, text in detail_to_paragraphs(detail):
            paragraphs.append({"section": heading, "subsection": subsection, "text": text})

    return paragraphs, list(dict.fromkeys(linked))


def parse_article(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    meta = parse_meta(soup)
    summary, keywords = parse_summary(soup)
    paragraphs, linked_ids = parse_body(soup)
    all_text = summary + " ".join(p["text"] for p in paragraphs)
    return {
        **meta,
        "summary": summary,
        "keywords": keywords,
        "paragraphs": paragraphs,
        "linked_ids": linked_ids,
        "related": parse_related(soup),
        "mentions_sejong": all_text.count("세종"),
    }


def main() -> None:
    entries = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    seed_ids = {e["id"] for e in entries}

    articles, link_counter = [], Counter()
    for e in entries:
        if not e.get("html_path"):
            print(f"[건너뜀] {e['id']} {e['title']}: HTML 없음")
            continue
        html = (ROOT / e["html_path"]).read_text(encoding="utf-8")
        art = parse_article(html)
        record = {
            "id": e["id"],
            "url": e["url"],
            "field": e["field"],
            **art,
            "source": e["source"],
            "license_note": LICENSE_NOTE,
        }
        articles.append(record)
        link_counter.update(i for i in art["linked_ids"] if i not in seed_ids)
        print(f"{e['id']} {art['title'] or e['title']}: 문단 {len(art['paragraphs'])}개, '세종' {art['mentions_sejong']}회")

    with OUTPUT.open("w", encoding="utf-8") as f:
        for r in articles:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_paras = sum(len(r["paragraphs"]) for r in articles)
    no_body = [f"{r['id']} {r['title']}" for r in articles if not r["paragraphs"]]
    no_sejong = [f"{r['id']} {r['title']}" for r in articles if r["mentions_sejong"] == 0]

    print("\n===== 요약 =====")
    print(f"항목 {len(articles)}개, 문단 {total_paras}개 → {OUTPUT.relative_to(ROOT)}")
    if no_body:
        print(f"본문을 못 찾은 항목 ({len(no_body)}): {no_body}")
    if no_sejong:
        print(f"'세종'이 한 번도 안 나오는 항목 ({len(no_sejong)}): {no_sejong}")
    print("시드에 없지만 본문에서 자주 링크된 항목 (확장 후보, 상위 15개):")
    for item_id, n in link_counter.most_common(15):
        print(f"  {item_id}  {n}회")


if __name__ == "__main__":
    main()
