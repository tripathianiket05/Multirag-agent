import logging

from app.core.exceptions import MultiRAGException
from app.generation.vlm import GeminiVLM
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_processor import TextProcessor
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionWorkflow:
    """Extract a PDF and turn its content into searchable evidence."""

    def __init__(self):
        self.loader = PDFLoader()
        self.processor = TextProcessor()
        self.vlm = GeminiVLM()
        self.store = VectorStore()

    def run(self, document_id: str) -> tuple[int, int]:
        pages, records = self.loader.extract(document_id)
        usable_records = []

        for record in records:
            if record["content_type"] == "visual":
                try:
                    record["text"] = self.vlm.describe(record["source_path"])
                    usable_records.append(record)
                except MultiRAGException as exc:
                    logger.warning(
                        "Skipping visual evidence on page %s: %s",
                        record["page"],
                        exc.message,
                    )
            else:
                usable_records.append(record)

        chunks = self.processor.process(usable_records)
        self.store.upsert(chunks)

        return pages, len(chunks)