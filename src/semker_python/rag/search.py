from __future__ import annotations

from typing import Any

from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

from semker_python.vectordb.qdrant_store import QdrantStore


async def retrieve_context(
  query: str,
  embedding_service: AzureTextEmbedding,
  store: QdrantStore,
  top_k: int = 5,
) -> list[dict[str, Any]]:
  embeddings = await embedding_service.generate_raw_embeddings([query])
  query_vector: list[float] = embeddings[0]
  return store.search(query_vector=query_vector, top_k=top_k)


def build_context_string(hits: list[dict[str, Any]]) -> str:
  parts: list[str] = []
  for i, hit in enumerate(hits, 1):
    name = hit.get("name", "Unknown")
    desc = hit.get("description", "")
    score = hit.get("score", 0.0)
    parts.append(f"[{i}] {name} (similarity: {score:.3f})\n{desc}")
  return "\n\n".join(parts)
