# MultiRAG

A simple content-aware multimodal RAG system for PDFs.

## Architecture

```text
                    USER
                      |
                  Streamlit
                      |
                   FastAPI
                      |
                 LangGraph
                      |
                 Retrieve Top 15
                      |
                 BGE Reranker
                      |
                  Top 5 evidence
                      |
                Gemini Summarizer
                      |
               Answer + sources

PDF ingestion:

PDF -> text/table/visual extraction
                  |
        visual -> Gemini Vision
                  |
              embeddings
                  |
                Qdrant
```

The important idea is simple: retrieval finds the best candidate evidence, the reranker chooses the strongest evidence, and one Gemini summarizer turns that evidence into the final answer.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your `GEMINI_API_KEY`.

Start Qdrant:

```bash
docker start multirag-qdrant
```

If the container does not exist yet:

```bash
docker run -d --name multirag-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

In a second terminal:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

## Components

- FastAPI: backend API
- Streamlit: simple demo interface
- PyMuPDF/pdfplumber: PDF extraction
- Sentence Transformers: local embeddings
- Qdrant: vector database
- BGE CrossEncoder: reranking
- Gemini: visual understanding and final answer summarization
- LangGraph: simple retrieve -> rerank -> summarize workflow