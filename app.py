from __future__ import annotations

import os
from dataclasses import replace

try:
    import streamlit as st
except ImportError as exc:
    raise RuntimeError("Install the app dependencies with: pip install -e '.[app]'") from exc

from deepread.engine import EvidenceGraphEngine
from deepread.config import Settings

st.set_page_config(page_title="DeepRead", page_icon="🔎", layout="wide")
st.title("DeepRead · EvidenceGraph")
st.caption("Hierarchical retrieval with visible evidence budgets and stop decisions")

corpus = st.sidebar.text_input("Corpus", os.getenv("DEEPREAD_CORPUS", "data/sample_corpus"))
provider = st.sidebar.selectbox("Provider", ["auto", "offline", "openai"], index=["auto", "offline", "openai"].index(os.getenv("DEEPREAD_PROVIDER", "auto")))
model = st.sidebar.text_input("OpenAI model", os.getenv("DEEPREAD_OPENAI_MODEL", "gpt-5.6-terra"))
question = st.text_area("Question", "Why are wetlands useful for both flood control and climate mitigation?")

if st.button("Build evidence graph", type="primary"):
    with st.spinner("Planning, retrieving, and reading…"):
        settings = replace(Settings.from_env(), provider=provider, openai_model=model)
        answer = EvidenceGraphEngine(corpus, settings).ask(question, "traces/latest.json")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Supported answer")
        st.markdown(answer.text)
        st.subheader("Citations")
        for index, item in enumerate(answer.citations, 1):
            with st.expander(f"[{index}] {item.title} · {item.section}"):
                st.write(item.text)
                st.caption(f"Level: {item.read_level.value} · Cost: {item.token_cost} tokens · Score: {item.score:.4f}")
    with right:
        st.metric("Requirement coverage", f"{answer.coverage:.0%}")
        st.metric("Read tokens", answer.trace.read_tokens)
        st.metric("Citation tokens", answer.trace.citation_tokens)
        st.metric("API tokens", answer.trace.api_total_tokens)
        st.metric("Provider", answer.trace.provider)
        if answer.trace.estimated_api_cost_usd is not None:
            st.metric("Estimated API cost", f"${answer.trace.estimated_api_cost_usd:.6f}")
        st.info(f"Stop reason: `{answer.stop_reason}`")
        with st.expander("API usage"):
            st.json(answer.trace.api_calls)
        with st.expander("Task graph", expanded=True):
            st.json([{"id": t.id, "question": t.question, "requirements": [r.description for r in t.requirements]} for t in answer.trace.tasks])
        with st.expander("Read decisions"):
            st.dataframe([{"passage": r.passage_id, "level": r.level.value, "utility": r.utility, "selected": r.selected, "reason": r.reason} for r in answer.trace.reads], use_container_width=True)
        with st.expander("Full trace"):
            st.json(answer.trace.to_dict())
