from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INDEX_NAME,
    API_KEYS
)

# Ingests company documentation to pinecone db
class DocumentationIngestor:
    def __init__(self, index_name: str = INDEX_NAME):
        self.embeddings = OpenAIEmbeddings(
            api_key=API_KEYS["openai"]
        )
        
        # Initialize Pinecone
        self.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            pinecone_api_key=API_KEYS["pinecone"]
        )
    
    def add_documents(self, documents: List[str]):
        """Add documents to Pinecone.
        
        Args:
            documents: List of document strings to be added to the vector store
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_text("\n".join(documents))
        
        # Add documents to Pinecone
        self.vector_store.add_texts(texts)

# Example usage
if __name__ == "__main__":
    # Initialize the ingestor
    ingestor = DocumentationIngestor()
    
    # Example documents
    documents = [
        "Our company policy states that refunds must be requested within 30 days of purchase.",
        "Technical support is available 24/7 through our help desk portal.",
        "All premium features are included in the enterprise subscription plan.",
        "To access premium features, users need to log in to their account and navigate to the 'Premium' section.",
        "Refund requests should be submitted through the customer portal with the order number and reason for refund."
   
    ]
    
    ingestor.add_documents(documents)
