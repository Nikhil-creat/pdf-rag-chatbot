"""
AI PDF Chatbot (RAG) — Streamlit UI

Run with:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import RAGPipeline
from src.config import settings

st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄", layout="wide")


def get_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    return st.session_state.pipeline


def main():
    st.title("📄 AI PDF Chatbot (RAG)")
    st.caption(
        "Upload a PDF, then ask questions about it. Answers are grounded in "
        "the document via retrieval-augmented generation — the model only "
        "sees the chunks retrieved for your question, not the whole PDF."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not settings.anthropic_api_key:
        st.warning(
            "No ANTHROPIC_API_KEY found. Copy `.env.example` to `.env` and "
            "add your key before asking questions (uploading/indexing still "
            "works without it)."
        )

    with st.sidebar:
        st.header("Upload a PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_file is not None:
            already_ingested = uploaded_file.name in get_pipeline().ingested_files
            if already_ingested:
                st.info(f"'{uploaded_file.name}' is already indexed.")
            elif st.button("Process PDF", type="primary"):
                with st.spinner("Extracting text, chunking, and embedding..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    try:
                        pipeline = get_pipeline()
                        n_chunks = pipeline.ingest_pdf(tmp_path, uploaded_file.name)
                    finally:
                        os.unlink(tmp_path)

                if n_chunks == 0:
                    st.error(
                        "No extractable text found. This PDF may be scanned "
                        "images — OCR isn't included in this version."
                    )
                else:
                    st.success(f"Indexed {n_chunks} chunks from '{uploaded_file.name}'.")

        pipeline = get_pipeline()
        if pipeline.ingested_files:
            st.divider()
            st.subheader("Indexed documents")
            for name in pipeline.ingested_files:
                st.write(f"• {name}")

        st.divider()
        with st.expander("Pipeline settings"):
            st.write(f"**Embedding model:** `{settings.embedding_model}`")
            st.write(f"**LLM:** `{settings.llm_model}`")
            st.write(f"**Chunk size / overlap:** {settings.chunk_size} / {settings.chunk_overlap}")
            st.write(f"**Top-k retrieved chunks:** {settings.top_k}")

    # --- Chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**p. {s.page}** (score {s.score:.2f}) — {s.text[:200]}...")

    # --- Chat input ---
    question = st.chat_input("Ask a question about the uploaded PDF...")
    if question:
        pipeline = get_pipeline()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if pipeline.store.is_empty:
                answer_text = "Upload and process a PDF first — I don't have any document indexed yet."
                sources = []
                st.markdown(answer_text)
            else:
                with st.spinner("Retrieving relevant passages and generating an answer..."):
                    try:
                        result = pipeline.answer_query(question)
                        answer_text, sources = result.answer, result.sources
                    except RuntimeError as e:
                        answer_text, sources = str(e), []
                st.markdown(answer_text)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"**p. {s.page}** (score {s.score:.2f}) — {s.text[:200]}...")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "sources": sources}
        )


if __name__ == "__main__":
    main()
