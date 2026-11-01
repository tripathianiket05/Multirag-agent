from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_env: str = "development"

    # Gemini
    gemini_api_key: str | None = None
    text_model: str = "gemini-3.8-flash"
    vision_model: str = "gemini-3.8-flash"

    # Local retrieval models
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "multirag"

    # Chunking and retrieval
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k_retrieval: int = 15
    top_k_rerank: int = 5

    # Upload and features
    max_file_size_mb: int = 50
    enable_reranker: bool = True

    # Data paths
    documents_dir: Path = ROOT_DIR / "data" / "documents"
    images_dir: Path = ROOT_DIR / "data" / "images"
    metadata_file: Path = ROOT_DIR / "data" / "documents.json"

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        extra="ignore",
    )


settings = Settings()
settings.documents_dir.mkdir(parents=True, exist_ok=True)
settings.images_dir.mkdir(parents=True, exist_ok=True)
settings.metadata_file.parent.mkdir(parents=True, exist_ok=True)