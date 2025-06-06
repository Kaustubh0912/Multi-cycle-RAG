from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """

    # SurrealDB Configuration
    surrealdb_url: str = Field(
        default="wss://cloakystores-06a9f7u3jlrsf43q77o8ttu1kk.aws-euw1.surreal.cloud"
    )
    surrealdb_user: str = Field(default="your_username")
    surrealdb_pass: str = Field(default="your_password")
    surrealdb_ns: str = Field(default="rag")
    surrealdb_db: str = Field(default="rag")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Authentication
    github_token: str = Field(default="")

    # Generation model (primary)
    llm_model: str = Field(default="meta/Meta-Llama-3.1-405B-Instruct")
    llm_temperature: float = Field(default=0.7)
    llm_max_tokens: int = Field(default=3000)

    # Evaluation model (for self-assessment)
    evaluation_model: str = Field(default="cohere/Cohere-command-r")
    evaluation_temperature: float = Field(default=0.3)
    evaluation_max_tokens: int = Field(default=1000)

    # Summary model (for final synthesis)
    summary_model: str = Field(default="meta/Meta-Llama-3.1-70B-Instruct")
    summary_temperature: float = Field(default=0.5)
    summary_max_tokens: int = Field(default=4000)

    # Reflexion configuration
    max_reflexion_cycles: int = Field(default=3)
    confidence_threshold: float = Field(default=0.85)
    initial_retrieval_k: int = Field(default=3)
    reflexion_retrieval_k: int = Field(default=5)

    # Memory cache
    enable_memory_cache: bool = Field(default=True)
    max_cache_size: int = Field(default=100)

    # Vector store
    chroma_persist_directory: str = Field(default="./chroma_db")
    chroma_collection_name: str = Field(default="rag_collection")

    # Embedding
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_device: str = Field(default="cpu")

    # Document processing
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)


# Global settings instance
settings = Settings()
