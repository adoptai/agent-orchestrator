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

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

# Matching Thresholds
NO_MATCH_THRESHOLD = 0.7
CONFUSION_MARGIN = 0.1 # Top 2 results are within this distance of each other
TOP_K_RESULTS = 3

# Log Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"