from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-large-en-v1.5"
    dimensions: int = 1024
    max_length: int = 512
    batch_size: int = 32
    device: str = "auto"
    
@dataclass
class PineconeConfig:
    region: str = "us-east-1"  # Changed from environment to region
    api_key: str = os.getenv("PINECONE_API_KEY")
    index_name: str = "hackrx-insurace-docs"
    dimension: int = 1024
    metric: str = "cosine"
    cloud: str = "aws"  

    




