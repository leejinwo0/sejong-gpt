"""세종 GPT 로컬 앱 (Streamlit).

실행 (저장소 루트에서):
    streamlit run app/main.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # streamlit은 app/ 폴더를 기준으로 실행하므로 저장소 루트를 경로에 추가

import streamlit as st

from app.chat import answer_stream, retrieve
from app.llm import SejongLLM
from app.retriever import Retriever

st.set_page_config(page_title="세종 GPT", page_icon="👑", layout="centered")


@st.cache_resource(show_spinner="세종대왕을 모셔 오는 중... (처음 한 번만 걸립니다)")
def load_models():
    return SejongLLM(), Retriever()


def render_sources(chunks: list[dict], show_text: bool) -> None:
    with st.expander(f"📜 참고한 사료 {len(chunks)}건"):
        for c in chunks:
            title = c["header"].strip("[]")
            st.markdown(f"**{title}** · 관련도 {c['score']:.2f}  \n[{c['source']}]({c['url']})")
            if show_text:
                st.caption(c["text"])


# ---------- 사이드바 ----------
with st.sidebar:
    st.header("설정")
    k = st.slider("참고할 사료 수", 1, 5, 3, help="많을수록 근거를 놓칠 일이 줄지만 답이 늦게 시작됩니다.")
    min_score = st.slider("사료 사용 기준 점수", 0.70, 0.95, 0.80, 0.01,
                          help="가장 관련 높은 사료의 점수가 이보다 낮으면 사료 없이 답합니다. (인사·잡담용)")
    temperature = st.slider("답변 다양성", 0.0, 1.0, 0.3, 0.05)
    show_text = st.toggle("사료 본문까지 보기", value=False)
    if st.button("대화 지우기", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption("사료 출처: 한국민족문화대백과사전 (한국학중앙연구원, 공공누리)\n\n"
               "모델: Kanana 1.5 2.1B를 세종대왕 페르소나로 LoRA 파인튜닝 (Apache 2.0)")

# ---------- 본문 ----------
st.title("👑 세종 GPT")
st.caption("사료를 근거로 답하는 세종대왕 AI · 1450년 승하 이후의 일은 알지 못합니다")

llm, retriever = load_models()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar="👑"):
        st.markdown(turn["answer"])
        if turn["chunks"]:
            render_sources(turn["chunks"], show_text)

question = st.chat_input("세종대왕께 여쭙고 싶은 것을 적어 보시오")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="👑"):
        result = retrieve(retriever, question, k=k, min_score=min_score)
        if result.chunks:
            titles = ", ".join(c["header"].strip("[]").split(" > ")[0] for c in result.chunks)
            status = st.status(f"사료를 살피는 중... ({titles})", expanded=False)
        else:
            status = None

        start = time.time()
        answer = st.write_stream(answer_stream(llm, question, result.chunks, temperature))
        if status:
            status.update(label=f"사료 {len(result.chunks)}건을 살펴 답함 · {time.time() - start:.1f}초", state="complete")
            render_sources(result.chunks, show_text)

    # 매 질문을 독립적으로 처리 (모델을 한 번에 한 질문씩 답하도록 학습했기 때문)
    st.session_state.history.append({"question": question, "answer": answer, "chunks": result.chunks})
