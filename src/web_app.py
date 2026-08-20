"""
Checkpoint 4 - Application Interface (Streamlit)

Thin web front end over the Checkpoint 3 RAG chain (rag_app.build_chain).
All retrieval/grounding/memory logic stays in rag_app.py -- this file is
presentation only.

Run:
    streamlit run src/web_app.py
"""
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_app import build_chain, extract_text  # noqa: E402

st.set_page_config(page_title="SyllaBot", page_icon="\U0001F4DA")
st.title("\U0001F4DA SyllaBot")
st.caption("Ask questions about your course syllabi. Answers are grounded "
           "only in the syllabi that have been ingested.")

if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "GEMINI_API_KEY is not set. Set it before launching Streamlit, e.g.\n\n"
        '`$env:GEMINI_API_KEY = "your-key-here"` (PowerShell), then rerun '
        "`streamlit run src/web_app.py`."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading SyllaBot (embedding model + vector index)...")
def get_chain():
    return build_chain()


rag_chain = get_chain()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about a course syllabus...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                config = {"configurable": {"session_id": st.session_state.session_id}}
                response = rag_chain.invoke({"question": question}, config=config)
                answer = extract_text(response.content)
            st.markdown(answer)
        except Exception as exc:
            answer = "Sorry, I couldn't get an answer right now. Please try again in a moment."
            st.error(answer)
            with st.expander("Error details"):
                st.code(str(exc))

    st.session_state.messages.append({"role": "assistant", "content": answer})
