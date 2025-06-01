from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # LLM configuration - Multiple models for reflexion
    github_token: str = Field(default="")

    # Generation model (primary)
    llm_model: str = Field(default="meta/Meta-Llama-3.1-405B-Instruct")
    llm_temperature: float = Field(default=0.7)
    llm_max_tokens: int = Field(default=1000)

    # Evaluation model (for self-assessment)
    evaluation_model: str = Field(default="cohere/Cohere-command-r")
    evaluation_temperature: float = Field(default=0.3)
    evaluation_max_tokens: int = Field(default=500)

    # Summary model (for final synthesis)
    summary_model: str = Field(default="meta/Meta-Llama-3.1-70B-Instruct")
    summary_temperature: float = Field(default=0.5)
    summary_max_tokens: int = Field(default=2000)

    # Reflexion Loop Configuration
    enable_reflexion_loop: bool = Field(default=True)
    max_reflexion_cycles: int = Field(default=5)
    confidence_threshold: float = Field(default=0.8)
    initial_retrieval_k: int = Field(default=3)
    reflexion_retrieval_k: int = Field(default=5)

    # Memory cache settings
    enable_memory_cache: bool = Field(default=True)
    max_cache_size: int = Field(default=100)

    # Query Decomposition Configuration (keeping existing)
    enable_query_decomposition: bool = Field(
        default=False
    )  # Disabled for reflexion mode
    use_context_aware_decomposer: bool = Field(default=True)
    decomposition_temperature: float = Field(default=0.3)
    max_sub_queries: int = Field(default=5)
    min_query_length_for_decomposition: int = Field(default=15)

    # Streaming configuration
    enable_streaming: bool = Field(default=True)
    steam_include_usage: bool = Field(default=True)

    # Database configuration
    vector_store_type: str = Field(default="chroma")
    chroma_persist_directory: str = Field(default="./chroma_db")
    chroma_collection_name: str = Field(default="rag_collection")

    # Embedding configuration
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_device: str = Field(default="cpu")

    # RAG configuration
    retrieval_k: int = Field(default=5)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Logging
    log_level: str = Field(default="INFO")


# Global settings instance
settings = Settings()
