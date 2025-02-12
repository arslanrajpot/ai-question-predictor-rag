import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class VectorStore:
    def __init__(self, index_name="cv-question-index", dimension=768):
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            if index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            self.index = self.pc.Index(index_name)
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise ValueError(f"Error initializing Pinecone: {e}")

    def upsert(self, vectors, batch_size=100, namespace=None):
        """Upsert vectors into Pinecone index in batches."""
        try:
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                logger.info(f"Upserting batch of {len(batch)} vectors to namespace {namespace}")
                self.index.upsert(vectors=batch, namespace=namespace)
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise ValueError(f"Error upserting vectors: {e}")

    def query(self, vector, top_k=5, namespace=None):
        """Query Pinecone for similar vectors."""
        try:
            if not vector:
                logger.warning("Empty vector provided for query")
                return []
            logger.info(f"Querying Pinecone with vector (length {len(vector)}): {vector[:5]}... for namespace {namespace}, top_k={top_k}")
            result = self.index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=namespace)
            matches = result["matches"]
            match_details = [f"{match['id']}: {match['score']}" for match in matches]
            logger.info(f"Pinecone query returned {len(matches)} matches: {match_details}")
            return matches
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return []

    def clear_namespace(self, namespace):
        """Clear vectors in a specific namespace."""
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            if "Namespace not found" in str(e):
                logger.info(f"Namespace {namespace} does not exist, no vectors to clear.")
            else:
                logger.error(f"Error clearing namespace {namespace}: {e}")
                raise ValueError(f"Error clearing namespace: {e}")