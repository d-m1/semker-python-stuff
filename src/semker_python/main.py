from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding

from semker_python.ingestion.loader import load_dinosaurs
from semker_python.rag.answer import answer_function
from semker_python.rag.search import SearchPlugin
from semker_python.vectordb.qdrant_store import QdrantStore

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "")
VECTOR_SIZE = 1536
EMBED_BATCH_SIZE = 64


def _check_env() -> None:
  if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    print("ERROR: set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env")
    sys.exit(1)


def build_kernel() -> tuple[Kernel, AzureTextEmbedding]:
  """Create a kernel with Azure OpenAI chat and embedding services."""
  _check_env()
  embedding_svc = AzureTextEmbedding(
    deployment_name=EMBEDDING_DEPLOYMENT,
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
  )
  kernel = Kernel()
  kernel.add_service(
    AzureChatCompletion(
      deployment_name=CHAT_DEPLOYMENT,
      endpoint=AZURE_OPENAI_ENDPOINT,
      api_key=AZURE_OPENAI_API_KEY,
    )
  )
  kernel.add_service(embedding_svc)
  return kernel, embedding_svc


async def ingest(embedding_svc: AzureTextEmbedding) -> QdrantStore:
  print("Loading dataset...")
  records = load_dinosaurs()

  descriptions = [r["description"] for r in records]
  all_embeddings: list[list[float]] = []
  print(f"Embedding {len(descriptions)} records with {EMBEDDING_DEPLOYMENT}...")
  for i in range(0, len(descriptions), EMBED_BATCH_SIZE):
    batch = descriptions[i : i + EMBED_BATCH_SIZE]
    batch_embeddings = await embedding_svc.generate_raw_embeddings(batch)
    all_embeddings.extend(batch_embeddings)

  store = QdrantStore(vector_size=VECTOR_SIZE)
  store.upsert(records, all_embeddings)
  print(f"{store.count()} dinosaurs indexed.\n")
  return store


async def ask(question: str, kernel: Kernel | None = None) -> None:
  if kernel is None:
    kernel, embedding_svc = build_kernel()
    store = await ingest(embedding_svc)
    plugin = SearchPlugin(store, embedding_svc)
    kernel.add_plugin(plugin, plugin_name="search")

  search_fn = kernel.plugins["search"]["retrieve"]
  context_result = await kernel.invoke(search_fn, query=question)

  result = await kernel.invoke(
    answer_function, retrieved_context=str(context_result), user_question=question
  )

  print(f"\n{question}\n{'─' * len(question)}")
  print(result)
  print()


async def interactive() -> None:
  kernel, embedding_svc = build_kernel()
  store = await ingest(embedding_svc)
  plugin = SearchPlugin(store, embedding_svc)
  kernel.add_plugin(plugin, plugin_name="search")

  print("Type a question, or 'quit' to exit.\n")
  while True:
    try:
      question = input("Ask > ").strip()
    except (EOFError, KeyboardInterrupt):
      print()
      break
    if not question or question.lower() in ("quit", "exit", "q"):
      break
    await ask(question, kernel=kernel)


def main() -> None:
  if len(sys.argv) < 2:
    print('usage: semker-python <ingest | ask "question" | interactive>')
    sys.exit(0)

  command = sys.argv[1].lower()

  if command == "ingest":
    _kernel, embedding_svc = build_kernel()
    asyncio.run(ingest(embedding_svc))
  elif command == "ask":
    if len(sys.argv) < 3:
      print('usage: semker-python ask "your question"')
      sys.exit(1)
    asyncio.run(ask(" ".join(sys.argv[2:])))
  elif command == "interactive":
    asyncio.run(interactive())
  else:
    print(f"unknown command: {command}")
    sys.exit(1)


if __name__ == "__main__":
  main()
