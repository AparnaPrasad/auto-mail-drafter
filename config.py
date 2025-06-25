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

# Evaluation configuration
EVALUATION_CONFIG = {
    # LangSmith Configuration
    "langsmith": {
        "api_key": os.getenv("LANGSMITH_API_KEY"),
        "tracing_v2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "project": os.getenv("LANGCHAIN_PROJECT", "email-agent-evaluation")
    },
    
    # Evaluation Settings
    "timeout": int(os.getenv("EVALUATION_TIMEOUT", "300")),  # Timeout in seconds for each test case
    "max_retries": int(os.getenv("MAX_RETRIES", "3")),  # Maximum retries for failed test cases
    "save_detailed_logs": os.getenv("SAVE_DETAILED_LOGS", "true").lower() == "true",
    "output_dir": os.getenv("OUTPUT_DIR", "./evaluation/reports"),
    
    # Test Configuration
    "run_all_tests": os.getenv("RUN_ALL_TESTS", "true").lower() == "true",
    "run_categories": os.getenv("RUN_CATEGORIES", "").split(",") if os.getenv("RUN_CATEGORIES") else [],
    
    # Scoring Weights
    "scoring_weights": {
        "tool_usage": float(os.getenv("TOOL_USAGE_WEIGHT", "0.4")),
        "response_quality": float(os.getenv("RESPONSE_QUALITY_WEIGHT", "0.3")),
        "processing_time": float(os.getenv("PROCESSING_TIME_WEIGHT", "0.1")),
        "similarity": float(os.getenv("SIMILARITY_WEIGHT", "0.2"))
    },
    
    # Test Case Categories
    "test_categories": [
        "pricing_inquiry",
        "refund_request", 
        "technical_support",
        "thank_you",
        "feature_inquiry",
        "account_management",
        "spam"
    ]
}