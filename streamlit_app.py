import time

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="MultiRAG",
    page_icon="📚",
    layout="wide",
)

st.title("MultiRAG")
st.caption("Content-Aware Multimodal RAG")


def get_health():
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def upload_pdf(file):
    try:
        response = requests.post(
            f"{API_URL}/api/documents/upload",
            files={
                "file": (
                    file.name,
                    file.getvalue(),
                    "application/pdf",
                )
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")
        return None


def get_document(document_id):
    try:
        response = requests.get(
            f"{API_URL}/api/documents/{document_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def ask(question, document_ids=None):
    try:
        response = requests.post(
            f"{API_URL}/api/query",
            json={
                "question": question,
                "document_ids": document_ids,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        try:
            detail = response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"Question failed: {detail}")
        return None


health = get_health()

with st.sidebar:
    st.header("System")

    if health:
        st.success("FastAPI connected")
        st.write(f"Qdrant: {health.get('qdrant', 'unknown')}")
        st.write(f"Gemini: {health.get('gemini', 'unknown')}")
    else:
        st.error("FastAPI is not running")

    st.divider()
    st.write("**Pipeline**")
    st.write("📄 PDF ingestion")
    st.write("🧩 Text + table + visual evidence")
    st.write("🔎 Retrieve top 15")
    st.write("↕️ Rerank top 5")
    st.write("🧠 Gemini summary")


st.header("1. Upload a PDF")

file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
)

if file:
    st.write(f"Selected: **{file.name}**")

    if st.button("Process PDF", type="primary"):
        with st.spinner("Uploading PDF..."):
            result = upload_pdf(file)

        if result:
            document_id = result["document_id"]
            st.info("PDF uploaded. Processing started.")

            progress = st.empty()

            for _ in range(60):
                status = get_document(document_id)

                if status and status.get("status") == "completed":
                    progress.success(
                        f"Ready: {status.get('evidence_items', 0)} evidence items indexed."
                    )
                    st.session_state["document_id"] = document_id
                    break

                if status and status.get("status") == "failed":
                    progress.error(
                        f"Processing failed: {status.get('error', 'Unknown error')}"
                    )
                    break

                progress.info("Processing PDF... extracting and indexing evidence.")
                time.sleep(1)
            else:
                progress.warning("Processing is taking longer than expected. You can wait and ask again.")


st.header("2. Ask a Question")

question = st.text_area(
    "Ask something about your document",
    placeholder="Example: What is the candidate's highest qualification?",
)

if st.button("Ask MultiRAG", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        document_ids = None
        if st.session_state.get("document_id"):
            document_ids = [st.session_state["document_id"]]

        with st.spinner("Retrieving, reranking and summarizing..."):
            result = ask(question.strip(), document_ids)

        if result:
            st.divider()

            st.subheader("Answer")
            st.write(result.get("answer", "No answer returned."))

            st.subheader("Best Match")
            st.info(result.get("best_match_type", "unknown"))

            sources = result.get("sources", [])
            if sources:
                st.subheader("Retrieved Evidence")

                for number, source in enumerate(sources, start=1):
                    title = (
                        f"Source {number} · Page {source.get('page', '?')} · "
                        f"{source.get('content_type', 'unknown')}"
                    )

                    with st.expander(title):
                        st.write(f"**Vector score:** {source.get('vector_score', 0):.4f}")
                        st.write(f"**Rerank score:** {source.get('rerank_score', 0):.4f}")
                        st.write(source.get("text", ""))

st.divider()
st.caption("Streamlit · FastAPI · LangGraph · Qdrant · BGE Reranker · Gemini")