from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # automatically reads from env as we're using pydantic v2

    # llm configuration
    github_token: str = Field(default="")
    llm_model: str = Field(default="cohere/Cohere-command-r")
    llm_temperature: float = Field(default=0.7)
    llm_max_tokens: int = Field(default=1000)

    # Query Decomposition Configuration
    enable_query_decomposition: bool = Field(default=True)
    use_context_aware_decomposer: bool = Field(default=True)
    decomposition_temperature: float = Field(default=0.3)
    max_sub_queries: int = Field(default=5)
    min_query_length_for_decomposition: int = Field(default=15)

    # Streaming configuration
    enable_streaming: bool = Field(default=True)
    steam_include_usage: bool = Field(default=True)

    # database configuration
    vector_store_type: str = Field(default="chroma")
    chroma_persist_directory: str = Field(default="./chroma_db")
    chroma_collection_name: str = Field(default="rag_collection")

    # embedding configuration
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_device: str = Field(default="cpu")

    # rag configuration
    retrieval_k: int = Field(default=5)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # logging
    log_level: str = Field(default="INFO")


# global settings instance
settings = Settings()
