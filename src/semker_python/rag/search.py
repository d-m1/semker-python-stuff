from __future__ import annotations

from typing import Any

from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
from semantic_kernel.functions import kernel_function

from semker_python.vectordb.qdrant_store import QdrantStore

TOP_K = 5


class SearchPlugin:
  """Kernel plugin for semantic search over the dinosaur knowledge base."""

  def __init__(self, store: QdrantStore, embedding_service: AzureTextEmbedding) -> None:
    self._store = store
    self._embedding_svc = embedding_service

  @kernel_function(name="retrieve", description="Retrieve relevant dinosaur info for a query")
  async def retrieve(self, query: str) -> str:
    embeddings = await self._embedding_svc.generate_raw_embeddings([query])
    hits = self._store.search(query_vector=embeddings[0], top_k=TOP_K)
    total = self._store.count()
    return f"[Database contains {total} dinosaurs total]\n\n{_format_hits(hits)}"


def _format_hits(hits: list[dict[str, Any]]) -> str:
  parts: list[str] = []
  for i, hit in enumerate(hits, 1):
    name = hit.get("name", "Unknown")
    desc = hit.get("description", "")
    score = hit.get("score", 0.0)
    parts.append(f"[{i}] {name} (similarity: {score:.3f})\n{desc}")
  return "\n\n".join(parts)
