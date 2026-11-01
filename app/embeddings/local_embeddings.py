from sentence_transformers import SentenceTransformer
from app.core.config import settings

class LocalEmbeddings:
    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(settings.embedding_model, device="cpu")
        return cls._model

    @classmethod
    def embed_text(cls, text: str) -> list[float]:
        return cls._get_model().encode(text, normalize_embeddings=True).tolist()

    @classmethod
    def embed_documents(cls, texts: list[str]) -> list[list[float]]:
        return cls._get_model().encode(texts, normalize_embeddings=True).tolist()