from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

class TextProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, records: list[dict]) -> list[dict]:
        chunks = []
        for record in records:
            if record["content_type"] == "text":
                for index, piece in enumerate(self.splitter.split_text(record["text"])):
                    item = dict(record)
                    item["id"] = f"{record['id']}_{index}"
                    item["text"] = piece
                    chunks.append(item)
            else:
                chunks.append(dict(record))
        return chunks