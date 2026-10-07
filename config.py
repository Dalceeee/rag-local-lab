import os

# PostgreSQL configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5433"))  # 5433 to avoid conflicts with local Postgres
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# LM Studio Configurazione / Generation
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "liquid/lfm2.5-1.2b")