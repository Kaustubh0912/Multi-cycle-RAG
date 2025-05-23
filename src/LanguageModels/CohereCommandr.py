# pyrefly: ignore-all-errors
import os

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()
# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
)
response = client.complete(
    messages=[
        SystemMessage("""You are a helpful assistant."""),
        UserMessage("Can you explain the basics of machine learning?"),
    ],
    model="cohere/Cohere-command-r",
    temperature=1.0,
    max_tokens=1000,
    top_p=1.0,
)
print(response.choices[0].message.content)
