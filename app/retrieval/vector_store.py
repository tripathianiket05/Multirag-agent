import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams
from app.core.config import settings
from app.embeddings.local_embeddings import LocalEmbeddings

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url) if settings.qdrant_url else QdrantClient(path=settings.qdrant_path)
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        if any(item.name == self.collection for item in self.client.get_collections().collections):
            return
        size = len(LocalEmbeddings.embed_text("multirag"))
        self.client.create_collection(collection_name=self.collection, vectors_config=VectorParams(size=size, distance=Distance.COSINE))

    def upsert(self, chunks: list[dict]):
        if not chunks:
            return
        vectors = LocalEmbeddings.embed_documents([chunk["text"] for chunk in chunks])
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])),
                vector=vector,
                payload={
                    "original_id": chunk["id"], "document_id": chunk["document_id"],
                    "page": chunk["page"], "content_type": chunk["content_type"],
                    "text": chunk["text"], "source_path": chunk.get("source_path"),
                },
            ))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, question: str, document_ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
        query_filter = None
        if document_ids:
            query_filter = Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=document_ids))])
        points = self.client.query_points(
            collection_name=self.collection,
            query=LocalEmbeddings.embed_text(question),
            query_filter=query_filter,
            limit=limit or settings.top_k_retrieval,
        ).points
        return [{"score": point.score, "payload": point.payload} for point in points]