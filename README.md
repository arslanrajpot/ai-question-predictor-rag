·# AI Interview Question Predictor (RAG)

An AI-powered tool that predicts the interview questions a candidate is most likely to face, based on their actual CV, the target job role, and the job description. It's built for recruiters, hiring teams, and candidates who want targeted interview preparation instead of generic question lists.

## Overview

The candidate uploads their CV (PDF) and specifies a candidate type (e.g. Intern, Associate Software Engineer, Experienced) and job role (e.g. MERN, Python). The system extracts the CV text, generates embeddings with Google Gemini, and retrieves the most relevant questions from a curated, role- and level-specific question bank using vector similarity search. The result is a shortlist of interview questions genuinely relevant to that candidate's background and the specific job description, rather than a generic, one-size-fits-all list.

## Key Features

CV parsing directly from PDF uploads. Gemini-powered semantic embeddings for accurate matching between a candidate's experience and relevant questions. A curated question bank segmented by role (MERN, Python) and experience level (Intern, Associate Software Engineer, Experienced). A simple web UI for uploading a CV and job description and instantly viewing predicted questions. A modular, RAG-based architecture that is easy to extend with new roles, levels, or question sets.

## Tech Stack

Flask (Python) for the web application and API. Google Gemini (`gemini-embedding-001`) for text embeddings. A custom vector store for similarity search over the question bank. PDF text extraction for CV parsing.

## Project Structure

```
api.py               API route handlers
main.py              Flask app entrypoint
cv_processor.py       extracts text from an uploaded CV (PDF)
embedder.py           generates Gemini embeddings for text
vector_store.py        stores and searches question embeddings
question_matcher.py    matches CV + job description to relevant questions
questions/             curated question bank (by role and experience level)
templates/             front-end HTML templates
```

## Getting Started

Clone the repository, then install dependencies with `pip install -r requirements.txt`. Create a `.env` file with your `GEMINI_API_KEY`. Run the app with `python main.py`, then open the app in your browser, upload a candidate's CV, select the candidate type and job role, add the job description, and get an instant list of predicted interview questions.

```bash
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
python main.py
```

## Roadmap

Planned improvements include support for additional roles beyond MERN and Python, richer scoring/explanations for why each question was selected, and exporting the predicted question set as a shareable interview prep sheet.

## Author

Built by **Arslan Arshad**, Full-Stack & AI Engineer.
Portfolio: https://arslan-arshad.netlify.app/ · Email: arslanarshad1018@gmail.com
