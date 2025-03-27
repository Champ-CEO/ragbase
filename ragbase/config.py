import os
from pathlib import Path


class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "images"

    class Database:
        DOCUMENTS_COLLECTION = "documents"

    class Model:
        EMBEDDINGS = "BAAI/bge-base-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        GENERAL_LLM = "llama-3.3-70b-versatile"
        COMPLEX_LLM = "deepseek-r1-distill-llama-70b"
        TEMPERATURE = 0.0
        MAX_TOKENS = 8000

    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = False

    # Optimization settings
    AUTO_COMPLEXITY_ROUTING = True  # Automatically route queries to appropriate model based on complexity
    USE_TOKEN_OPTIMIZATION = True   # Apply token optimization techniques to reduce token usage

    DEBUG = False
    CONVERSATION_MESSAGES_LIMIT = 6
