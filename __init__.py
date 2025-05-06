# ---- LLM ----
API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1/"
MODEL_NAME = "granite3.3:8b"
SUMMARIZATION_MODEL = MODEL_NAME

# ---- TIKA ----
TIKA_SERVER_URL = "http://127.0.0.1:9998/tika"

# ---- QDRANT ----
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120
DISTANCE_METRIC = "MANHATTAN"

# ---- EMBEDDING ----
EMBEDDING_MODEL = "bge-m3:latest"
EMBEDDING_DIMENSION = 1024
SPARSE_EMBEDDING_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
SENTENCE_TRANSFORMER_MODEL = "aari1995/German_Semantic_V3"

# ---- SPACY ----
SPACY_MODEL = "de_core_news_lg"


# ---- NEO4J ----
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = None

# ---- OTHER ----

RERANKER_MODEL = "bge-m3:latest"

SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"

MODEL_CONTEXT_LIMIT_TOKENS = 127000
