from typing import Any, Dict, List

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential

from ..config.settings import settings
from ..core.exceptions import LLMException
from ..core.interfaces import LLMInterface


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
