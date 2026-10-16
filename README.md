
# AI Recruitment & Candidate Matching Platform

## Overview
The AI Recruitment & Candidate Matching Platform is a Python-based web application that helps recruiters analyse resumes and compare candidates against job requirements.

The application combines traditional skill matching with semantic similarity using Google Gemini to support candidate review.

## Features
- Upload multiple PDF resumes
- Extract text from resumes
- Identify relevant skills automatically
- Compare candidate skills with job requirements
- Calculate semantic similarity using Gemini embeddings
- Rank candidates using a hybrid matching score
- Generate AI-assisted candidate insights
- Generate job-specific interview questions
- Download candidate results as a CSV file
- View candidate summaries and detected skill gaps

## Technology Stack
- Python
- Streamlit
- Pandas
- PyPDF
- Google Gemini API
- Gemini Embeddings

## Project Structure
```text
ai-recruitment-platform/
├── app.py
├── README.md
└── .streamlit/
    └── secrets.toml
```

## Installation

1. Clone or download this repository.
2. Open the project folder in a terminal.
3. Install the dependencies:

```bash
python -m pip install streamlit pandas pypdf google-genai
```

4. Create a `.streamlit/secrets.toml` file and add your own Gemini API key:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

5. Start the application:

```bash
python -m streamlit run app.py
```

6. Open the local URL displayed in the terminal.

## How to Use
1. Enter a job title and department.
2. Paste the job description and required skills.
3. Upload candidate resumes in PDF format.
4. Click **Analyse and Rank Candidates**.
5. Review candidate rankings and skill matches.
6. Generate AI candidate insights and interview questions.
7. Download the results as a CSV file.

## Responsible Use
Matching scores are experimental decision-support indicators, not probabilities of job success. Recruiters should verify candidate information, evaluate applications fairly, protect personal data, and never rely solely on automated scores for hiring decisions.

## Author
Developed as an AI recruitment and candidate matching project.
