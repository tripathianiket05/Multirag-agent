from app.core.config import settings

class Reranker:
    _model = None

    def _get_model(self):
        if not settings.enable_reranker:
            return None
        if self.__class__._model is None:
            from sentence_transformers import CrossEncoder
            self.__class__._model = CrossEncoder(settings.reranker_model, device="cpu")
        return self.__class__._model

    def rerank(self, question: str, results: list[dict]) -> list[dict]:
        if not results:
            return []
        if not settings.enable_reranker:
            return results[:settings.top_k_rerank]
        scores = self._get_model().predict([(question, item["payload"].get("text", "")) for item in results])
        ranked = []
        for item, score in zip(results, scores):
            copy = dict(item)
            copy["rerank_score"] = float(score)
            ranked.append(copy)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:settings.top_k_rerank]