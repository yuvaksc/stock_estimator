"""Central configuration loaded from environment / .env (see .env.example)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ── Groq / LLM ──
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # ── Database ──
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/stocks"

    # ── Supabase REST (keepalive only) ──
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # ── Redis ──
    redis_url: str = "redis://localhost:6379/0"

    # ── Kafka / Redpanda ──
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_security_protocol: str = "PLAINTEXT"  # or SASL_SSL
    kafka_sasl_mechanism: str = "SCRAM-SHA-256"
    kafka_sasl_username: str = ""
    kafka_sasl_password: str = ""
    kafka_topic_social: str = "raw.social"
    kafka_topic_ticks: str = "raw.ticks"
    kafka_consumer_group: str = "stock-sentiment-worker"

    # ── Finnhub ──
    finnhub_api_key: str = ""

    # ── Reddit ──
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "stock-sentiment-app/0.1 by u/your_reddit_username"

    # ── Ingestion targets (comma-separated in env) ──
    tickers: str = "AAPL,JPM,ORCL,MSFT,AMZN,GOOG"
    subreddits: str = "wallstreetbets,stocks,investing"

    # ── Semantic cache ──
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 21600
    cache_max_vectors: int = 5000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── FinBERT ──
    finbert_model: str = "ProsusAI/finbert"

    # ── API / auth ──
    api_auth_token: str = ""  # empty disables auth (dev)
    cors_origins: str = "http://localhost:3000"

    # ── ML artifacts ──
    results_dir: str = "Results"
    merged_tensors_dir: str = "test_merged_tensors"
    raw_price_dir: str = "stock_data/price/raw"
    model_version: str = "multimodal-regressor-v1"

    # ── MLOps (optional) ──
    mlflow_tracking_uri: str = ""
    wandb_api_key: str = ""

    # ── Worker ──
    run_worker_inproc: bool = False

    # ── derived helpers ──
    @property
    def tickers_list(self) -> list[str]:
        return [t.strip().upper() for t in self.tickers.split(",") if t.strip()]

    @property
    def subreddits_list(self) -> list[str]:
        return [s.strip() for s in self.subreddits.split(",") if s.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [c.strip() for c in self.cors_origins.split(",") if c.strip()]

    @property
    def kafka_uses_sasl(self) -> bool:
        return self.kafka_security_protocol.upper().startswith("SASL")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
