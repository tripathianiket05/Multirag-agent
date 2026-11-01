from unittest.mock import MagicMock
from app.retrieval.reranker import Reranker

def test_reranker_sorts_by_score(monkeypatch):
    monkeypatch.setattr("app.retrieval.reranker.settings.enable_reranker", True)
    fake = MagicMock()
    fake.predict.return_value = [0.2, 0.9, 0.4]
    Reranker._model = fake
    results = [{"score": 0.9, "payload": {"text": "old", "content_type": "text"}}, {"score": 0.8, "payload": {"text": "best", "content_type": "visual"}}, {"score": 0.7, "payload": {"text": "middle", "content_type": "table"}}]
    ranked = Reranker().rerank("question", results)
    assert ranked[0]["payload"]["text"] == "best"
    assert ranked[0]["rerank_score"] == 0.9