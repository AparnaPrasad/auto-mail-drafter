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
INDEX_NAME = os.getenv("INDEX_NAME")  # Default value if not set
DIMENSION = 1024  # OpenAI embedding dimension
METRIC = "cosine"

# Email settings
EMAIL_CONFIG = {
    "address": os.getenv("EMAIL_ADDRESS"),
    "password": os.getenv("EMAIL_PASSWORD"),
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
        "PINECONE_ENVIRONMENT",
        "INDEX_NAME"  # Adding INDEX_NAME to required variables
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
RULES  = {
    "emailResponseRules": [
        "1. Keep the tone friendly and professional.",
        "2. Keep the response concise and to the point.",
        "3. Your name is Aparna Prasad, CEO of the company.",
        "4. The company name is Acme Corp.",
        "5. The company address is 123 Main St, Anytown, USA 12345.",
        "6. The company phone number is 123-456-7890.",
        "7. The company email is support@acme.com.",
        "8. The company website is https://www.acme.com."
    ],
    "reactRules": [
        "1. Do not respond to emails that are not relevant to the company.",
        "2. If the email is not relevant, do not respond.",
        "3. Do not summarize or describe your response. Just **call the function**.",
        "4. Do not respond to no-reply or alert emails.",
        "5. If its a thank you email, respond."
    ]
}