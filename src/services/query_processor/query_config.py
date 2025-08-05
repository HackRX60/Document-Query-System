# config.py
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class QueryProcessorConfig:
    """Configuration for the Query Processor"""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    
    # Pinecone Settings
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "document-search")
    
    # Model Settings
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    GEMINI_MODEL: str = "gemini-2.5-flash" 
    
    # Search Settings
    DEFAULT_TOP_K: int = 10
    MAX_CONTEXT_LENGTH: int = 4000
    CONFIDENCE_THRESHOLD: float = 0.3
    
    # Domain Settings
    SUPPORTED_DOMAINS: list = None
    
    def __post_init__(self):
        if self.SUPPORTED_DOMAINS is None:
            self.SUPPORTED_DOMAINS = ["insurance", "legal", "hr", "compliance"]
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            self.GEMINI_API_KEY,
            self.PINECONE_API_KEY,
            self.PINECONE_ENVIRONMENT,
            self.PINECONE_INDEX_NAME
        ]
        
        missing_keys = [key for key in required_keys if not key]
        if missing_keys:
            raise ValueError(f"Missing required configuration: {missing_keys}")
        
        return True

# requirements.txt content
REQUIREMENTS = """
google-generativeai>=0.3.0
pinecone-client>=2.2.4
sentence-transformers>=2.2.2
numpy>=1.21.0
python-dotenv>=0.19.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6
aiofiles>=23.0.0
pandas>=1.5.0
"""

# .env template
ENV_TEMPLATE = """
# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp-free
PINECONE_INDEX_NAME=document-search

# Optional Settings
LOG_LEVEL=INFO
MAX_WORKERS=4
CACHE_ENABLED=true
"""

def setup_environment():
    """Setup environment and dependencies"""
    import subprocess
    import sys
    
    print("Setting up Query Processor environment...")
    
    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS)
    
    # Create .env template
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(ENV_TEMPLATE)
        print("Created .env template - please fill in your API keys")
    
    # Install dependencies
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
    
    print("Environment setup complete!")

if __name__ == "__main__":
    setup_environment()