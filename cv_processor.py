import PyPDF2
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVProcessor:
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file."""
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise ValueError(f"Error extracting text from PDF: {e}")

    def preprocess_text(self, text):
        """Clean and preprocess text."""
        try:
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)
            text = re.sub(r'\b\d{10}\b', '', text)
            return text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            raise ValueError(f"Error preprocessing text: {e}")