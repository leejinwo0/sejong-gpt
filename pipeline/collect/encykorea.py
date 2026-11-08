"""한국민족문화대백과사전 항목 HTML 수집 (1단계: 다운로드만).

- 공공데이터포털 색인 CSV에서 항목명·분야를 찾고
- encykorea_seeds.txt에 적힌 항목만 웹페이지 HTML을 내려받아 캐시합니다.
- 본문 추출(파싱)은 HTML 구조를 확인한 뒤 다음 단계에서 작성합니다.

실행 (저장소 루트에서):
    python -m pipeline.collect.encykorea --dry-run   # 대상 목록만 확인
    python -m pipeline.collect.encykorea             # 실제 다운로드
"""
import argparse
import json
import re
import time
import urllib.robotparser
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "encykorea"
INDEX_CSV = RAW_DIR / "index_20240130.csv"
SEEDS_FILE = Path(__file__).with_name("encykorea_seeds.txt")
HTML_DIR = RAW_DIR / "html"
MANIFEST = RAW_DIR / "manifest.jsonl"

BASE_URL = "https://encykorea.aks.ac.kr"
HEADERS = {"User-Agent": "sejong-gpt-portfolio/0.1 (personal non-commercial research)"}
DELAY_SEC = 1.5  # 서버 부담을 줄이기 위한 요청 간격


def load_seeds() -> list[str]:
    ids = []
    for line in SEEDS_FILE.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            ids.append(token)
    return list(dict.fromkeys(ids))  # 중복 제거, 순서 유지


def load_index() -> pd.DataFrame:
    df = pd.read_csv(INDEX_CSV, encoding="utf-8-sig")
    df["id"] = df["웹사이트 주소"].str.extract(r"(E\d+)")[0]
    return df.set_index("id")


def article_url(item_id: str) -> str:
    # 색인 CSV의 옛 주소(/Contents/Item/...) 대신 현재 주소 형식 사용
    return f"{BASE_URL}/Article/{item_id}"


def robots_allows(url: str) -> bool:
    rp = urllib.robotparser.RobotFileParser(f"{BASE_URL}/robots.txt")
    try:
        rp.read()
    except Exception as e:  # robots.txt를 못 읽으면 보수적으로 사람이 확인하도록 중단
        raise SystemExit(f"robots.txt를 읽지 못했습니다: {e}")
    return rp.can_fetch(HEADERS["User-Agent"], url)


def main(dry_run: bool) -> None:
    index = load_index()
    seeds = load_seeds()

    missing = [i for i in seeds if i not in index.index]
    if missing:
        print(f"[경고] 색인에 없는 ID: {missing}")
    targets = [i for i in seeds if i in index.index]

    print(f"수집 대상 {len(targets)}개")
    for i in targets:
        row = index.loc[i]
        print(f"  {i}  {row['항목명']}  ({row['분야']})")
    if dry_run:
        return

    if not robots_allows(article_url(targets[0])):
        raise SystemExit("robots.txt가 이 경로의 수집을 허용하지 않습니다. 중단합니다.")

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    with MANIFEST.open("w", encoding="utf-8") as mf:
        for n, item_id in enumerate(targets, 1):
            row = index.loc[item_id]
            url = article_url(item_id)
            path = HTML_DIR / f"{item_id}.html"
            status = "cached"

            if not path.exists():
                try:
                    res = session.get(url, timeout=20)
                    res.raise_for_status()
                    path.write_text(res.text, encoding="utf-8")
                    status = "ok"
                except requests.RequestException as e:
                    status = f"error: {e}"
                time.sleep(DELAY_SEC)

            print(f"[{n}/{len(targets)}] {item_id} {row['항목명']} → {status}")
            mf.write(json.dumps({
                "id": item_id,
                "title": row["항목명"],
                "field": row["분야"],
                "url": url,
                "html_path": str(path.relative_to(ROOT)) if path.exists() else None,
                "status": status,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "source": f"[출처 : {row['항목명']} - 한국민족문화대백과사전]",
            }, ensure_ascii=False) + "\n")

    print(f"완료: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="다운로드 없이 대상 목록만 출력")
    main(parser.parse_args().dry_run)
