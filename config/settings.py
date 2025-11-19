import os
from dotenv import load_dotenv
load_dotenv()

# Base URL for the API
API_BASE_URL = "https://connect.adopt.ai"

# Endpoint paths
AUTH_ENDPOINT = "/v1/auth/token"
ACTIONS_ENDPOINT = "/v1/actions/list"
CONVERSATIONS_ENDPOINT = "/v1/conversations/{conversation_id}/messages"

# Credentials
CLIENT_ID = os.getenv("CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")
CONVERSATION_ID = os.getenv("CONVERSATION_ID")

# ============================================================================
# Embedding Configuration
# ============================================================================

# Choose your embedding provider: "local", "openai", "voyage", or "titan"
EMBEDDING_PROVIDER = "local"  # Using OpenAI (local has issues on macOS)

# Model name depends on your provider:
# - local: "all-mpnet-base-v2" (default, free, runs on your computer)
# - openai: "text-embedding-3-small" or "text-embedding-3-large"
# - voyage: "voyage-2"
# - titan: "amazon.titan-embed-text-v1"
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Provider-specific API keys (loaded from .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Default to us-east-1

# Dimension mapping for different models
# This tells us how many numbers are in each embedding vector
EMBEDDING_DIMENSIONS = {
    "all-mpnet-base-v2": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "voyage-2": 1024,
    "amazon.titan-embed-text-v1": 1536,
}

# Get the dimension for the current model
# This dynamically adjusts based on which model you're using
def get_embedding_dimension() -> int:
    """
    Returns the embedding dimension for the currently configured model.
    
    Raises:
        ValueError: If the model is not recognized
    """
    if EMBEDDING_MODEL not in EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Unknown embedding model: {EMBEDDING_MODEL}. "
            f"Supported models: {list(EMBEDDING_DIMENSIONS.keys())}"
        )
    return EMBEDDING_DIMENSIONS[EMBEDDING_MODEL]

# For backward compatibility with code that uses EMBEDDING_DIMENSION
EMBEDDING_DIMENSION = get_embedding_dimension()

# ============================================================================
# Matching Thresholds
# ============================================================================
NO_MATCH_THRESHOLD = 0.7
CONFUSION_MARGIN = 0.1 # Top 2 results are within this distance of each other
TOP_K_RESULTS = 3

# Log Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"