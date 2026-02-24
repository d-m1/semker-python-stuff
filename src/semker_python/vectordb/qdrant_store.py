from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "dinosaurs"


class QdrantStore:
  def __init__(self, vector_size: int = 1536) -> None:
    self.client = QdrantClient(":memory:")
    self.vector_size = vector_size
    self._ensure_collection()

  def _ensure_collection(self) -> None:
    names = [c.name for c in self.client.get_collections().collections]
    if COLLECTION_NAME not in names:
      self.client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
          size=self.vector_size,
          distance=Distance.COSINE,
        ),
      )

  def upsert(
    self,
    records: list[dict[str, Any]],
    embeddings: list[list[float]],
  ) -> int:
    points = [
      PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, rec["name"].lower())),
        vector=emb,
        payload={
          "name": rec.get("name"),
          "diet": rec.get("diet"),
          "period": rec.get("period"),
          "continent": rec.get("continent"),
          "description": rec.get("description"),
        },
      )
      for rec, emb in zip(records, embeddings, strict=True)
    ]
    self.client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Upserted {len(points)} points into '{COLLECTION_NAME}'")
    return len(points)

  def search(
    self,
    query_vector: list[float],
    top_k: int = 5,
  ) -> list[dict[str, Any]]:
    results = self.client.query_points(
      collection_name=COLLECTION_NAME,
      query=query_vector,
      limit=top_k,
    )
    return [
      {**(dict(pt.payload) if pt.payload else {}), "score": pt.score} for pt in results.points
    ]

  def count(self) -> int:
    info = self.client.get_collection(COLLECTION_NAME)
    return info.points_count or 0
