from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.generation.llm import GeminiLLM
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore


class QueryState(TypedDict, total=False):
    question: str
    document_ids: list[str] | None
    retrieved: list[dict]
    reranked: list[dict]
    answer: str


class QueryGraph:
    """Simple MultiRAG workflow: retrieve -> rerank -> summarize."""

    def __init__(self):
        self.store = VectorStore()
        self.reranker = Reranker()
        self.llm = GeminiLLM()

        graph = StateGraph(QueryState)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("rerank", self.rerank)
        graph.add_node("summarize", self.summarize)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "summarize")
        graph.add_edge("summarize", END)

        self.graph = graph.compile()

    def retrieve(self, state: QueryState):
        results = self.store.search(
            state["question"],
            state.get("document_ids"),
        )
        return {"retrieved": results}

    def rerank(self, state: QueryState):
        results = self.reranker.rerank(
            state["question"],
            state.get("retrieved", []),
        )
        return {"reranked": results}

    def summarize(self, state: QueryState):
        evidence = [item["payload"] for item in state.get("reranked", [])]
        answer = self.llm.summarize(state["question"], evidence)
        return {"answer": answer}

    def run(self, question: str, document_ids: list[str] | None = None):
        return self.graph.invoke(
            {
                "question": question.strip(),
                "document_ids": document_ids,
            }
        )