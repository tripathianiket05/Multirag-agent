from fastapi import APIRouter
from qdrant_client import QdrantClient

from app.core.config import settings

router = APIRouter()


@router.get("")
async def health():
    result = {
        "status": "ok",
        "qdrant": "unknown",
        "gemini": "configured" if settings.gemini_api_key else "missing",
    }

    try:
        client = (
            QdrantClient(url=settings.qdrant_url)
            if settings.qdrant_url
            else QdrantClient(path=settings.qdrant_path)
        )
        client.get_collections()
        client.close()
        result["qdrant"] = "healthy"
    except Exception:
        result["qdrant"] = "unhealthy"
        result["status"] = "degraded"

    if result["gemini"] == "missing":
        result["status"] = "degraded"

    return result