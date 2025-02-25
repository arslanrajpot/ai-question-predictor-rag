from flask import Flask, request, jsonify, render_template
from cv_processor import CVProcessor
from embedder import Embedder
from vector_store import VectorStore
from question_matcher import QuestionMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
processor = CVProcessor()
embedder = Embedder()
vector_store = VectorStore()
matcher = QuestionMatcher(vector_store, embedder)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_cv', methods=['POST'])
def upload_cv():
    try:
        if 'cv' not in request.files:
            return jsonify({"error": "No CV file provided"}), 400
        if not request.form.get('candidate_type') or not request.form.get('job_role'):
            return jsonify({"error": "Candidate type and job role are required"}), 400
        if not request.form.get('job_description'):
            return jsonify({"error": "Job description is required"}), 400

        file = request.files['cv']
        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        cv_text = processor.extract_text_from_pdf(file)
        cv_text = processor.preprocess_text(cv_text)
        candidate_type = request.form['candidate_type'].lower()
        job_role = request.form['job_role'].lower()
        job_description = processor.preprocess_text(request.form['job_description'])

        # Validate candidate type and job role
        valid_types = ['experienced', 'ase', 'intern']
        valid_roles = ['mern', 'python']
        if candidate_type not in valid_types or job_role not in valid_roles:
            return jsonify({"error": "Invalid candidate type or job role"}), 400

        questions = matcher.predict_questions(cv_text, job_description, candidate_type, job_role)
        return jsonify({"questions": questions})
    except Exception as e:
        logger.error(f"Error in upload_cv: {e}")
        return jsonify({"error": str(e)}), 500