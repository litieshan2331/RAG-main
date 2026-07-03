from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "KnowEngine Python RAG"
    app_env: str = "local"
    debug: bool = True

    mysql_url: str = "mysql+pymysql://root:123456@localhost:3306/know_engine?charset=utf8mb4"
    redis_url: str = "redis://:123456@localhost:6379/0"

    openai_api_key: str = Field(default="", repr=False)
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen-max-latest"
    streaming_chat_model: str = "qwen-max-latest"
    vision_model: str = "qwen3-vl-plus"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 9

    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "know-engine-vector"
    elasticsearch_request_timeout_seconds: float = 60

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = Field(default="neo4j666", repr=False)
    neo4j_database: str = "neo4j"

    minio_endpoint: str = "localhost:9000"
    minio_public_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = Field(default="minioadmin", repr=False)
    minio_bucket: str = "know-engine"
    minio_secure: bool = False

    mineru_parse_api_url: str = "http://47.104.64.223:8000"
    mineru_connect_timeout_seconds: float = 30
    mineru_response_timeout_seconds: float = 300
    manage_mineru_process: bool = False
    mineru_api_command: str = "mineru-api"
    mineru_api_host: str = "127.0.0.1"
    mineru_api_port: int = 8000
    mineru_startup_timeout_seconds: float = 120
    mineru_shutdown_timeout_seconds: float = 15
    mineru_log_file: str = ".runtime/mineru-api.log"
    mineru_fail_fast_on_startup: bool = False
    mineru_generate_image_descriptions: bool = True
    mineru_image_description_max_bytes: int = 5_000_000

    default_chunk_size: int = 1000
    default_chunk_overlap: int = 80
    hybrid_top_k: int = 5
    hybrid_min_score: float = 0.5
    hybrid_candidate_k: int = 30
    hybrid_rrf_k: int = 60

    reranker_enabled: bool = True
    reranker_model: str = "qwen3-rerank"
    reranker_api_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    reranker_timeout_seconds: float = 30
    reranker_instruct: str = "根据用户问题判断文档片段相关性，用于 RAG 问答召回重排。"
    reranker_max_input_chars: int = 900

    web_search_enabled: bool = True
    tavily_api_key: str = Field(default="", repr=False)
    tavily_api_url: str = "https://api.tavily.com/search"
    tavily_search_depth: str = "basic"
    tavily_max_results: int = 5
    tavily_timeout_seconds: float = 20
    tavily_include_answer: bool = True

    react_agent_enabled: bool = True
    react_agent_max_steps: int = 5
    react_agent_confidence_threshold: float = 0.55
    react_context_max_chars: int = 12_000
    react_observation_max_chars: int = 2_500
    react_context_max_tokens: int = 6_000
    react_context_trigger_ratio: float = 0.8
    react_context_reserve_ratio: float = 0.35
    react_tool_result_max_tokens: int = 1_200
    react_summary_max_tokens: int = 1_000
    react_context_llm_summary_enabled: bool = True

    text2sql_max_rows: int = 100
    text2sql_empty_result_triggers_fallback: bool = True
    text2cypher_max_rows: int = 100
    text2cypher_empty_result_triggers_fallback: bool = True
    text2cypher_schema_sample_limit: int = 5

    conversation_history_max_messages: int = 10
    conversation_history_max_chars: int = 8_000
    context_dependency_threshold: float = 0.6

    ragas_judge_model: str = ""
    ragas_timeout_seconds: float = 120
    ragas_max_samples: int = 50

    ingestion_worker_count: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
