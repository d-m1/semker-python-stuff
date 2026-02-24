from __future__ import annotations

from semantic_kernel.connectors.ai.open_ai import (
  AzureChatCompletion,
  AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory

from semker_python.rag.prompts import RAG_SYSTEM_PROMPT


async def generate_answer(
  user_question: str,
  retrieved_context: str,
  chat_service: AzureChatCompletion,
) -> str:
  system_message = RAG_SYSTEM_PROMPT.replace("{{$retrieved_context}}", retrieved_context).replace(
    "{{$user_question}}", user_question
  )

  history = ChatHistory()
  history.add_system_message(system_message)
  history.add_user_message(user_question)

  settings = AzureChatPromptExecutionSettings(
    temperature=0.0,
  )

  response = await chat_service.get_chat_message_content(
    chat_history=history,
    settings=settings,
  )

  return str(response)
