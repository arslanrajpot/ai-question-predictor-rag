from flask import Flask, request, jsonify, render_template
from cv_processor import CVProcessor
from question_matcher import QuestionMatcher
from vector_store import VectorStore
from embedder import Embedder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
cv_processor = CVProcessor()
vector_store = VectorStore()
embedder = Embedder()
question_matcher = QuestionMatcher(vector_store, embedder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_cv', methods=['POST'])
def upload_cv():
    try:
        cv_file = request.files['cv']
        candidate_type = request.form['candidate_type']
        job_role = request.form['job_role']
        job_description = request.form['job_description']

        cv_text = cv_processor.extract_text_from_pdf(cv_file)
        logger.info(f"Extracted CV text length: {len(cv_text)}")

        questions = question_matcher.predict_questions(cv_text, job_description, candidate_type, job_role)
        return jsonify({"questions": questions})
    except Exception as e:
        logger.error(f"Error in upload_cv: {e}")
        return jsonify({"questions": []}), 500

if __name__ == '__main__':
    app.run(debug=True)