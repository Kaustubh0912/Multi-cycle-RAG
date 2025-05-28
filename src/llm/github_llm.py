from typing import Any, Dict, List

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from typing_extensions import AsyncIterator

from ..config.settings import settings
from ..core.exceptions import LLMException
from ..core.interfaces import LLMInterface, StreamingChunk


class GitHubLLM(LLMInterface):
    """GitHub Models LLM implementation"""

    def __init__(self) -> None:
        super().__init__()
        try:
            self.client = ChatCompletionsClient(
                endpoint="https://models.github.ai/inference",
                credential=AzureKeyCredential(settings.github_token),
            )
            self.model = settings.llm_model
            self.temperature = settings.llm_temperature
            self.max_tokens = settings.llm_max_tokens
        except Exception as e:
            raise LLMException(f"Failed to initialize GitHub LLM: {e}")

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a simple prompt"""
        try:
            response = self.client.complete(
                messages=[UserMessage(prompt)],
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMException(f"Generation failed: {e}")

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Chat with structured messages"""
        try:
            # Convert our message format to Azure format
            azure_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    azure_messages.append(SystemMessage(msg["content"]))
                elif msg["role"] == "user":
                    azure_messages.append(UserMessage(msg["content"]))
                elif msg["role"] == "assistant":
                    azure_messages.append(AssistantMessage(msg["content"]))

            response = self.client.complete(
                messages=azure_messages,
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMException(f"Chat failed: {e}")

    async def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Generate text from prompt with streaming"""
        try:
            response = self.client.complete(
                stream=True,
                messages=[UserMessage(prompt)],
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                model_extras={"stream_options": {"include_usage": True}},
            )
            for update in response:
                chunk_content = ""
                usage_info = None
                is_complete = False

                # Extract content from the update
                if update.choices and update.choices[0].delta:
                    chunk_content = update.choices[0].delta.content or ""

                # Extract usage information (if available)
                if update.usage:
                    usage_info = {
                        "completion_tokens": getattr(
                            update.usage, "completion_tokens", 0
                        ),
                        "prompt_tokens": getattr(
                            update.usage, "prompt_tokens", 0
                        ),
                        "total_tokens": getattr(
                            update.usage, "total_tokens", 0
                        ),
                    }
                    is_complete = True  # Usage info usually comes at the end

                yield StreamingChunk(
                    content=chunk_content,
                    is_complete=is_complete,
                    usage_info=usage_info,
                )
        except Exception as e:
            raise LLMException(f"Streaming generation failed: {e}")

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Chat with streaming responses"""
        try:
            # Convert our messages to Azure Format
            azure_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    azure_messages.append(SystemMessage(msg["content"]))
                elif msg["role"] == "user":
                    azure_messages.append(UserMessage(msg["content"]))
                elif msg["role"] == "assistant":
                    azure_messages.append(AssistantMessage(msg["content"]))

            response = self.client.complete(
                stream=True,
                messages=azure_messages,
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                model_extras={"stream_options": {"include_usage": True}},
            )

            for update in response:
                chunk_content = ""
                usage_info = None
                is_complete = False

                # Extract content from the update
                if update.choices and update.choices[0].delta:
                    chunk_content = update.choices[0].delta.content or ""

                # Extract usage information (if available)
                if update.usage:
                    usage_info = {
                        "completion_tokens": getattr(
                            update.usage, "completion_tokens", 0
                        ),
                        "prompt_tokens": getattr(
                            update.usage, "prompt_tokens", 0
                        ),
                        "total_tokens": getattr(
                            update.usage, "total_tokens", 0
                        ),
                    }
                    is_complete = True

                yield StreamingChunk(
                    content=chunk_content,
                    is_complete=is_complete,
                    usage_info=usage_info,
                )

        except Exception as e:
            raise LLMException(f"Streaming chat failed: {e}")
