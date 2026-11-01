import mimetypes
from pathlib import Path

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.exceptions import MultiRAGException


class GeminiVLM:
    """Turn an extracted PDF page image into searchable visual evidence."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise MultiRAGException("GEMINI_API_KEY is missing from .env.", 500)

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.vision_model

    def describe(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise MultiRAGException(f"Visual file not found: {image_path}", 400)

        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"

        prompt = """Analyze this PDF page image for question answering.
Describe useful visible information such as diagrams, charts, labels,
relationships, numbers, and important visual elements.
Ignore decoration and do not guess details that are not visible.
Return a concise factual description."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=path.read_bytes(),
                        mime_type=mime_type,
                    ),
                    prompt,
                ],
            )
        except Exception as exc:
            raise MultiRAGException(
                f"Gemini visual processing failed: {exc}", 502
            ) from exc

        description = (response.text or "").strip()
        if not description:
            raise MultiRAGException(
                "Gemini returned an empty visual description.", 502
            )

        return description