from app.ingestion.text_processor import TextProcessor

def test_text_is_split():
    processor = TextProcessor()
    processor.splitter._chunk_size = 50
    processor.splitter._chunk_overlap = 10
    chunks = processor.process([{"id": "d_p1_text", "document_id": "d", "page": 1, "content_type": "text", "text": "This is a long piece of text that should be split into smaller pieces for retrieval."}])
    assert len(chunks) > 1
    assert chunks[0]["content_type"] == "text"

def test_non_text_is_kept_whole():
    processor = TextProcessor()
    chunks = processor.process([{"id": "t1", "document_id": "d", "page": 1, "content_type": "table", "text": "| A | B |"}, {"id": "v1", "document_id": "d", "page": 2, "content_type": "visual", "text": "diagram"}])
    assert [item["id"] for item in chunks] == ["t1", "v1"]