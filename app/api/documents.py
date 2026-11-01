import json
import logging

from fastapi import APIRouter, BackgroundTasks, File, UploadFile

from app.core.config import settings
from app.core.exceptions import MultiRAGException
from app.ingestion.pdf_loader import PDFLoader
from app.models.schemas import DocumentInfo, DocumentResponse
from app.workflows.ingestion import IngestionWorkflow

router = APIRouter()
logger = logging.getLogger(__name__)
loader = PDFLoader()


def _read() -> list[dict]:
    if not settings.metadata_file.exists():
        return []

    try:
        return json.loads(settings.metadata_file.read_text())
    except json.JSONDecodeError:
        return []


def _write(items: list[dict]):
    settings.metadata_file.write_text(json.dumps(items, indent=2))


def _update(document_id: str, **changes):
    items = _read()

    for item in items:
        if item["document_id"] == document_id:
            item.update(changes)
            break

    _write(items)


def process_document(document_id: str):
    try:
        pages, count = IngestionWorkflow().run(document_id)
        _update(
            document_id,
            status="completed",
            pages=pages,
            evidence_items=count,
            error=None,
        )
    except Exception as exc:
        logger.exception("Ingestion failed for %s", document_id)
        _update(document_id, status="failed", error=str(exc))


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    document_id, filename = await loader.save_upload(file)

    items = _read()
    items.append(
        {
            "document_id": document_id,
            "filename": filename,
            "status": "processing",
            "pages": None,
            "evidence_items": 0,
            "error": None,
        }
    )
    _write(items)

    background_tasks.add_task(process_document, document_id)

    return DocumentResponse(
        document_id=document_id,
        filename=filename,
        status="processing",
        message="PDF uploaded. Processing has started.",
    )


@router.get("", response_model=list[DocumentInfo])
async def list_documents():
    return _read()


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    for item in _read():
        if item["document_id"] == document_id:
            return item

    raise MultiRAGException("Document not found.", 404)