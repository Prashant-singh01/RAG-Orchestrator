"""Streamlit user interface for the RAG application."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from config import CHROMA_DB_PATH, DATA_DIR
from evaluation import evaluate_response, evaluate_retrieval
from src.document_loader import DocumentLoader
from src.logger import get_logger
from src.rag_pipeline import RAGPipeline
from src.text_splitter import TextChunker
from src.vector_store import VectorStore

LOGGER = get_logger("streamlit_app")


def build_ui() -> None:
    """Render the complete Streamlit interface."""
    st.set_page_config(page_title="RAG Project", page_icon="📚", layout="wide")
    st.title("RAG Project")
    st.caption(
        "A beginner-to-intermediate Retrieval Augmented Generation application "
        "built with Python, Hugging Face, LangChain, ChromaDB, and Streamlit."
    )

    with st.sidebar:
        st.header("Project Controls")
        st.markdown("Upload a PDF to build or refresh the vector store.")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        process_button = st.button("Process PDF")
        st.markdown("---")
        st.subheader("Status")
        status_placeholder = st.empty()

    if uploaded_file is not None and process_button:
        try:
            file_path = DATA_DIR / uploaded_file.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(uploaded_file.getvalue())
            LOGGER.info("PDF uploaded: %s", file_path)

            status_placeholder.info("Processing PDF and building vector store...")
            start = time.time()

            documents = DocumentLoader(str(file_path.relative_to(Path.cwd()))).load_documents()
            chunks = TextChunker().split_documents(documents)
            LOGGER.info("Chunk count generated: %s", len(chunks))

            vector_store = VectorStore()
            vector_store.create_vector_store(chunks)
            LOGGER.info("Vector store created successfully at %s", CHROMA_DB_PATH)

            elapsed = round(time.time() - start, 4)
            status_placeholder.success(f"PDF processed successfully in {elapsed} seconds.")
            st.success("Vector store created successfully.")
        except Exception as exc:
            LOGGER.exception("Error during PDF processing: %s", exc)
            status_placeholder.error(f"Error while processing PDF: {exc}")
            st.error(f"Error while processing PDF: {exc}")

    question = st.text_input("Ask a question about the uploaded document")
    ask_button = st.button("Ask")

    if ask_button and question.strip():
        try:
            LOGGER.info("User question: %s", question)
            start_time = time.time()
            pipeline = RAGPipeline()
            prepared = pipeline.prepare_question(question)
            documents = prepared["documents"]
            context = prepared["context"]

            if not documents:
                st.warning("I couldn't find this information in the provided document.")
                return

            answer_placeholder = st.empty()
            answer_tokens: list[str] = []
            for token in pipeline.stream_answer(prepared):
                answer_tokens.append(token)
                answer_placeholder.markdown("".join(answer_tokens))

            answer = "".join(answer_tokens).strip()
            eval_metrics = evaluate_response(answer, context, start_time)
            retrieval_metrics = evaluate_retrieval(documents)
            eval_metrics.update(retrieval_metrics)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Chunks")
            for index, document in enumerate(documents, start=1):
                with st.expander(f"Chunk {index}"):
                    st.write(document.page_content)
                    st.json(document.metadata)

            st.subheader("Metadata")
            st.json(eval_metrics)

            LOGGER.info("Evaluation metrics: %s", eval_metrics)
        except Exception as exc:
            LOGGER.exception("Error while answering question: %s", exc)
            st.error(f"Error while answering question: {exc}")


if __name__ == "__main__":
    build_ui()
