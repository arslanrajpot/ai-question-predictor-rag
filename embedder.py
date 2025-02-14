import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class Embedder:
    def __init__(self, model='models/text-embedding-004'):
        self.model = model

    def generate_embedding(self, text, task_type="retrieval_query"):
        """Generate embedding for the given text using Gemini."""
        try:
            if not text.strip():
                logger.warning("Empty text provided for embedding")
                return None
            result = genai.embed_content(model=self.model, content=text, task_type=task_type)
            embedding = result['embedding']
            logger.info(f"Generated embedding for text (length {len(text)}) with task_type {task_type}: {embedding[:5]}... (length {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise ValueError(f"Error generating embedding: {e}")