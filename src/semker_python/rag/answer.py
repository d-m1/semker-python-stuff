from __future__ import annotations

from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.functions import KernelFunction

from semker_python.rag.prompts import RAG_PROMPT

# SK renders the template variables and routes to the chat service automatically.
answer_function = KernelFunction.from_prompt(
  function_name="answer",
  plugin_name="rag",
  description="Answer dinosaur questions using retrieved context",
  prompt=RAG_PROMPT,
  prompt_execution_settings=AzureChatPromptExecutionSettings(temperature=0.0),
)
