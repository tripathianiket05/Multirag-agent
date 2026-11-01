from google import genai

from app.core.config import settings
from app.core.exceptions import MultiRAGException


class GeminiLLM:
    """Generate the final answer from the evidence retrieved by MultiRAG."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise MultiRAGException("GEMINI_API_KEY is missing from .env.", 500)

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.text_model

    def summarize(self, question: str, evidence: list[dict]) -> str:
        context_parts = []

        for number, item in enumerate(evidence, start=1):
            context_parts.append(
                f"SOURCE {number}\n"
                f"Document: {item.get('document_id', 'unknown')}\n"
                f"Page: {item.get('page', '?')}\n"
                f"Content type: {item.get('content_type', 'unknown')}\n"
                f"Evidence: {item.get('text', '')}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are the final answer generator for MultiRAG.

Answer the question using ONLY the evidence below.

Rules:
- Do not invent information.
- Use the retrieved evidence as the source of truth.
- You may combine information from multiple sources.
- If the evidence is not enough, say that clearly.
- Keep the answer simple and direct.
- Mention page numbers when useful.

Question:
{question}

Retrieved evidence:
{context}

Answer:"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
        except Exception as exc:
            raise MultiRAGException(
                f"Gemini answer generation failed: {exc}", 502
            ) from exc

        answer = (response.text or "").strip()
        if not answer:
            raise MultiRAGException("Gemini returned an empty answer.", 502)

        return answer