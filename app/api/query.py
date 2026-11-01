from fastapi import APIRouter, Request

from app.core.exceptions import MultiRAGException
from app.models.schemas import QueryRequest, QueryResponse, SourceCitation
from app.workflows.query_graph import QueryGraph

router = APIRouter()


@router.post("", response_model=QueryResponse)
async def ask_question(request: Request, body: QueryRequest):
    result = QueryGraph().run(body.question, body.document_ids)
    ranked = result.get("reranked", [])

    if not ranked:
        raise MultiRAGException("No relevant evidence was found.", 404)

    sources = []
    for item in ranked:
        payload = item["payload"]
        sources.append(
            SourceCitation(
                document_id=payload.get("document_id", "unknown"),
                page=int(payload.get("page", 0)),
                content_type=payload.get("content_type", "unknown"),
                vector_score=float(item.get("score", 0.0)),
                rerank_score=float(item.get("rerank_score", 0.0)),
                text=payload.get("text", "")[:1000],
                source_path=payload.get("source_path"),
            )
        )

    return QueryResponse(
        answer=result["answer"],
        best_match_type=ranked[0]["payload"].get("content_type", "unknown"),
        sources=sources,
        request_id=getattr(request.state, "request_id", None),
    )