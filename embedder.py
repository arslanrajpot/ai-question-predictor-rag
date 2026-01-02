import os
from dotenv import load_dotenv
import logging
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

class Embedder:
    def __init__(self, model='models/gemini-embedding-001', output_dimensionality=768):
        self.model = model
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.output_dimensionality = output_dimensionality

    def generate_embedding(self, text, task_type="retrieval_query"):
        """Generate embedding for the given text using Gemini."""
        try:
            if not text.strip():
                logger.warning("Empty text provided for embedding")
                return None
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={"output_dimensionality": self.output_dimensionality},
            )
            embedding = response.embeddings[0].values
            logger.info(f"Generated embedding for text (length {len(text)}) with task_type {task_type}: {embedding[:5]}... (length {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise ValueError(f"Error generating embedding: {e}")