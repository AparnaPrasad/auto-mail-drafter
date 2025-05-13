import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
MAX_RETRIEVAL_DOCS = 3

# Model settings
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7

# Vector database settings
INDEX_NAME = "index-name"
DIMENSION = 1024  # OpenAI embedding dimension
METRIC = "cosine"

# Email settings
EMAIL_CONFIG = {
    "address": os.getenv("EMAIL_ADDRESS", "test@email.com"),
    "password": os.getenv("EMAIL_PASSWORD", "app password"),
    "imap_server": os.getenv("IMAP_SERVER", "imap.gmail.com"),
    "imap_port": int(os.getenv("IMAP_PORT", "993")),
    "use_ssl": os.getenv("USE_SSL", "True").lower() == "true"
}

# Database settings
DATABASE_CONFIG = {
    "path": os.getenv("DB_PATH", "emails.db")
}

# Fetch settings
FETCH_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", "100"))
}

# API Keys (loaded from environment variables)
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "pinecone": os.getenv("PINECONE_API_KEY"),
    "pinecone_env": os.getenv("PINECONE_ENVIRONMENT")
}

def validate_config():
    """Validate the configuration and required environment variables."""
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate email configuration
    if not all([EMAIL_CONFIG["address"], EMAIL_CONFIG["password"]]):
        raise ValueError("Email address and password must be provided")
    
    # Validate database path
    if not DATABASE_CONFIG["path"]:
        raise ValueError("Database path must be provided")

# Validate configuration on import
validate_config()

# Email response settings
MAX_RESPONSE_LENGTH = 1000
RESPONSE_TONE = "professional" 