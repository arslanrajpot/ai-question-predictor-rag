import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class QuestionMatcher:
    def __init__(self, vector_store, embedder, generative_model='models/gemini-1.5-flash'):
        self.vector_store = vector_store
        self.embedder = embedder
        self.generative_model = genai.GenerativeModel(generative_model)
        self.current_type_role = None

    def chunk_cv(self, cv_text, max_chunk_size=500):
        """Split CV text into chunks of max_chunk_size characters."""
        try:
            chunks = []
            words = cv_text.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chunk_size:
                    current_chunk += word + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
            logger.info(f"Split CV into {len(chunks)} chunks: {[chunk[:50] + '...' for chunk in chunks]}")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking CV: {e}")
            return [cv_text[:max_chunk_size]]  # Fallback to single chunk

    def load_questions(self, candidate_type, job_role):
        """Load questions and store embeddings in Pinecone."""
        try:
            filename = f"questions/{candidate_type.lower()}_{job_role.lower()}.txt"
            if not os.path.exists(filename):
                logger.error(f"Question file {filename} not found")
                raise ValueError(f"Question file {filename} not found")
            with open(filename, 'r') as f:
                questions = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(questions)} questions from {filename}: {questions}")

            if not questions:
                logger.warning(f"No questions found in {filename}")
                return questions

            # Generate embeddings and upsert to Pinecone
            namespace = f"{candidate_type}_{job_role}"
            self.vector_store.clear_namespace(namespace)
            question_embeddings = [self.embedder.generate_embedding(q, task_type="retrieval_query") for q in questions]
            logger.info(f"Generated {len(question_embeddings)} embeddings for questions")
            vectors = [(f"{candidate_type}_{job_role}_q{i}", emb, {"text": q, "type": candidate_type, "role": job_role})
                       for i, (q, emb) in enumerate(zip(questions, question_embeddings))]
            self.vector_store.upsert(vectors, namespace=namespace)
            logger.info(f"Upserted {len(vectors)} vectors to Pinecone namespace {namespace}")
            return questions
        except Exception as e:
            logger.error(f"Error loading questions from {filename}: {e}")
            raise ValueError(f"Error loading questions: {e}")

    def predict_questions(self, cv_text, job_description, candidate_type, job_role, top_k=5):
        """Predict relevant questions using Pinecone and Gemini."""
        try:
            if not cv_text.strip():
                logger.warning("CV text is empty")
                return []
            if not job_description.strip():
                logger.warning("Job description is empty")
                return []

            # Combine CV and job description, then chunk
            combined_text = f"CV: {cv_text}\nJob Description: {job_description}"
            chunks = self.chunk_cv(combined_text)
            logger.info(f"Processing {len(chunks)} CV chunks")

            # Load questions if not already loaded
            if self.current_type_role != (candidate_type, job_role):
                questions = self.load_questions(candidate_type, job_role)
                self.current_type_role = (candidate_type, job_role)
            else:
                questions = self.load_questions(candidate_type, job_role)

            if not questions:
                logger.warning("No questions available")
                return []

            # Query Pinecone for each chunk
            namespace = f"{candidate_type}_{job_role}"
            all_matches = []
            for i, chunk in enumerate(chunks):
                cv_embedding = self.embedder.generate_embedding(chunk, task_type="retrieval_query")
                logger.info(f"Generated embedding for chunk {i+1} (length {len(chunk)}): {cv_embedding[:5]}...")
                matches = self.vector_store.query(cv_embedding, top_k=top_k, namespace=namespace)
                all_matches.extend(matches)
                logger.info(f"Chunk {i+1} returned {len(matches)} matches")

            # Aggregate unique questions, sorted by score
            unique_questions = []
            seen_ids = set()
            for match in sorted(all_matches, key=lambda x: x['score'], reverse=True):
                if "metadata" in match and "text" in match["metadata"] and match["id"] not in seen_ids:
                    unique_questions.append(match["metadata"]["text"])
                    seen_ids.add(match["id"])
            unique_questions = unique_questions[:top_k]
            logger.info(f"Aggregated {len(unique_questions)} unique questions: {unique_questions}")

            if not unique_questions:
                logger.warning("No matches found, falling back to first questions")
                unique_questions = questions[:top_k]

            # Refine questions using Gemini
            refined_questions = self.refine_questions(cv_text, job_description, unique_questions)
            logger.info(f"Returning {len(refined_questions)} refined questions: {refined_questions}")
            return refined_questions
        except Exception as e:
            logger.error(f"Error predicting questions: {e}")
            return []  # Return empty list on error

    def refine_questions(self, cv_text, job_description, questions):
        """Refine questions using Gemini."""
        try:
            if not questions:
                logger.warning("No questions to refine")
                return questions
            prompt = (
                f"Given the CV: '{cv_text[:1000]}' and Job Description: '{job_description[:500]}', "
                f"refine these questions to be more specific and relevant to the candidate’s experience and job requirements: {', '.join(questions)}. "
                f"Return exactly {len(questions)} refined questions as a numbered list (e.g., '1. Question text')."
            )
            logger.info(f"Refining questions with prompt: {prompt[:200]}...")
            response = self.generative_model.generate_content(prompt)
            logger.info(f"Gemini raw response for refinement: {response.text[:200]}...")
            refined = []
            for line in response.text.split('\n'):
                line = line.strip()
                if line and re.match(r'^\d+\.\s+', line):
                    question = re.sub(r'^\d+\.\s+', '', line)
                    refined.append(question)
            refined = refined[:len(questions)]
            logger.info(f"Refined {len(refined)} questions with Gemini: {refined}")
            if not refined:
                logger.warning("Gemini returned no valid refined questions, returning unrefined")
                return questions
            return refined
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                logger.warning(f"Gemini API quota exceeded, returning unrefined questions: {e}")
                return questions
            logger.error(f"Error refining questions: {e}")
            return questions