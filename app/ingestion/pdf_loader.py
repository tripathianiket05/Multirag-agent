import logging
import uuid
from pathlib import Path
from typing import Any

import fitz
import pdfplumber
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import MultiRAGException

logger = logging.getLogger(__name__)

class PDFLoader:
    async def save_upload(self, file: UploadFile) -> tuple[str, str]:
        filename = file.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise MultiRAGException("Only PDF files are supported.", 400)
        content = await file.read()
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise MultiRAGException(f"File is too large. Maximum size is {settings.max_file_size_mb} MB.", 400)
        document_id = str(uuid.uuid4())
        (settings.documents_dir / f"{document_id}.pdf").write_bytes(content)
        return document_id, filename

    def extract(self, document_id: str) -> tuple[int, list[dict[str, Any]]]:
        pdf_path = settings.documents_dir / f"{document_id}.pdf"
        if not pdf_path.exists():
            raise MultiRAGException("Document not found.", 404)

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            raise MultiRAGException("The PDF is invalid or corrupted.", 400) from exc

        records: list[dict[str, Any]] = []
        try:
            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page_number, page in enumerate(doc, start=1):
                    plumber_page = plumber_pdf.pages[page_number - 1]
                    table_boxes = []

                    for table_index, table in enumerate(plumber_page.find_tables()):
                        table_boxes.append(table.bbox)
                        table_text = self._table_to_markdown(table.extract() or [])
                        if table_text:
                            records.append({
                                "id": f"{document_id}_p{page_number}_table_{table_index}",
                                "document_id": document_id, "page": page_number,
                                "content_type": "table", "text": table_text,
                            })

                    text_parts = []
                    for block in page.get_text("blocks"):
                        x0, y0, x1, y1, text, *_ = block
                        text = text.strip()
                        if not text:
                            continue
                        if any(self._overlaps((x0, y0, x1, y1), box) for box in table_boxes):
                            continue
                        text_parts.append(text)
                    if text_parts:
                        records.append({
                            "id": f"{document_id}_p{page_number}_text",
                            "document_id": document_id, "page": page_number,
                            "content_type": "text", "text": "\n\n".join(text_parts),
                        })

                    # Rendering the whole page catches vector charts/diagrams too.
                    if page.get_images(full=True) or page.get_drawings():
                        image_path = self._save_page_image(page, document_id, page_number)
                        records.append({
                            "id": f"{document_id}_p{page_number}_visual",
                            "document_id": document_id, "page": page_number,
                            "content_type": "visual", "source_path": str(image_path), "text": "",
                        })
            return len(doc), records
        except MultiRAGException:
            raise
        except Exception as exc:
            logger.exception("PDF extraction failed for %s", document_id)
            raise MultiRAGException("Could not extract the PDF content.") from exc
        finally:
            doc.close()

    @staticmethod
    def _table_to_markdown(rows: list[list[Any]]) -> str:
        rows = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
        rows = [row for row in rows if any(row)]
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
        return "\n".join(lines)

    @staticmethod
    def _overlaps(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1)

    @staticmethod
    def _save_page_image(page: fitz.Page, document_id: str, page_number: int) -> Path:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        path = settings.images_dir / f"{document_id}_page_{page_number}.png"
        pixmap.save(path)
        return path