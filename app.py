
import os
import re
import math
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from google import genai




st.set_page_config(
    page_title="AI Recruitment Platform",
    page_icon="👥",
    layout="wide"
)

MODEL_NAME = "gemini-flash-latest"
EMBEDDING_MODEL = "gemini-embedding-001"

SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#",
    "html", "css", "react", "angular", "vue", "node.js",
    "express", "django", "flask", "streamlit", "sql", "mysql",
    "postgresql", "mongodb", "machine learning", "deep learning",
    "artificial intelligence", "data analysis", "data science",
    "pandas", "numpy", "power bi", "tableau", "excel", "statistics",
    "nlp", "natural language processing", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "generative ai",
    "llm", "langchain", "langgraph", "prompt engineering", "rag",
    "faiss", "git", "github", "docker", "kubernetes", "aws", "azure",
    "google cloud", "api", "rest api", "fastapi", "linux",
    "figma", "ui/ux", "wordpress", "communication", "leadership",
    "teamwork", "problem solving", "project management",
    "recruitment", "human resources", "hr", "talent acquisition",
    "interviewing"
]




st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: #f7f8fc;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17172f, #302b60);
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #242047;
        letter-spacing: -0.4px;
    }

    .hero-panel {
        background: linear-gradient(120deg, #242047 0%, #635bdb 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(70, 60, 150, 0.12);
    }

    .hero-eyebrow {
        color: #dcd9ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .hero-title {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin-top: 10px;
    }

    .hero-description {
        color: #eeecff;
        font-size: 15px;
        margin-top: 8px;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #ececf5;
        border-radius: 16px;
        padding: 20px 18px;
        box-shadow: 0 5px 18px rgba(36, 35, 75, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #777a91;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #635bdb;
        font-weight: 800;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: #635bdb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #5048c6;
        color: white;
        border: none;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #b7b3ef;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stDataFrame"] {
        background: white;
        border-radius: 14px;
        border: 1px solid #ececf5;
        overflow: hidden;
    }

    [data-testid="stExpander"] {
        background: white;
        border: 1px solid #ececf5;
        border-radius: 12px;
    }

    hr {
        border-color: #e9e9f3;
    }
    </style>
    """,
    unsafe_allow_html=True
)




def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Check your existing "
            ".streamlit/secrets.toml file."
        )

    return genai.Client(api_key=api_key)


def generate_ai_response(prompt):
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text
    finally:
        client.close()




def extract_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)

    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    return text.strip()


def find_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = (
            r"(?<![a-z0-9+#])"
            + re.escape(skill)
            + r"(?![a-z0-9+#])"
        )

        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


def extract_required_skills(job_description, manual_skills):
    detected = find_skills(job_description)

    manual = [
        skill.strip().lower()
        for skill in manual_skills.split(",")
        if skill.strip()
    ]

    return sorted(set(detected + manual))


def calculate_match(candidate_skills, required_skills):
    if not required_skills:
        return 0.0, []

    matched = [
        skill for skill in required_skills
        if skill in candidate_skills
    ]

    score = len(matched) / len(required_skills) * 100

    return round(score, 1), matched




def get_text_embedding(client, text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text[:8000]
    )

    if not response.embeddings:
        raise ValueError("No embedding was returned.")

    return response.embeddings[0].values


def cosine_similarity(vector_a, vector_b):
    dot = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot / (magnitude_a * magnitude_b)


def calculate_semantic_similarity(client, job_vector, resume_text):
    resume_vector = get_text_embedding(
        client,
        "Candidate resume: " + resume_text
    )

    similarity = cosine_similarity(job_vector, resume_vector)

    return round(max(0.0, min(1.0, similarity)) * 100, 1)




def generate_candidate_insights(
    job_title,
    job_description,
    candidate_name,
    resume_text,
    candidate_skills,
    required_skills,
    matched_skills
):
    missing = [
        skill for skill in required_skills
        if skill not in candidate_skills
    ]

    prompt = f"""
You are an assistant helping a recruiter review a resume.

Job title: {job_title}

Job description:
{job_description[:5000]}

Candidate: {candidate_name}

Skills detected:
{", ".join(candidate_skills) or "None detected"}

Required skills:
{", ".join(required_skills) or "None provided"}

Matched skills:
{", ".join(matched_skills) or "None detected"}

Skills not detected:
{", ".join(missing) or "None"}

Resume text:
{resume_text[:7000]}

Provide:
1. A brief candidate summary.
2. Evidence of relevant skills from the resume.
3. Potential skill gaps, using "not detected" where appropriate.
4. Three relevant interview questions.
5. Follow-up questions to verify experience.

Do not invent qualifications or experience.
Do not infer protected personal characteristics.
Do not make a final hiring decision.
"""

    return generate_ai_response(prompt)


def generate_interview_questions(job_title, job_description, skills):
    prompt = f"""
Create 8 interview questions for this role.

Job title: {job_title}
Job description: {job_description[:5000]}
Relevant skills: {", ".join(skills)}

Include 3 technical questions, 2 practical scenarios,
2 problem-solving questions and 1 teamwork question.
Do not provide answers.
"""

    return generate_ai_response(prompt)




st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-eyebrow">SMART HIRING WORKSPACE</div>
        <div class="hero-title">AI Recruitment Platform</div>
        <div class="hero-description">
            Discover relevant skills, compare resumes, and support
            recruitment with AI-powered insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)




with st.sidebar:
    st.markdown("## 👥 RecruitAI")
    st.caption("Recruitment workspace")

    st.divider()

    st.markdown("### Job Settings")

    job_title = st.text_input(
        "Job title",
        value="AI Trainee"
    )

    department = st.selectbox(
        "Department",
        [
            "Artificial Intelligence",
            "Software Development",
            "Data Science",
            "Human Resources",
            "Marketing",
            "Other"
        ]
    )

    st.divider()

    st.caption("Powered by Python, Streamlit and Google Gemini")




st.header("1. Job Details")

job_description = st.text_area(
    "Job description",
    height=150,
    placeholder=(
        "Paste the job description here, including responsibilities "
        "and qualifications..."
    )
)

manual_skills = st.text_input(
    "Additional required skills (comma-separated)",
    placeholder="Python, SQL, machine learning"
)

required_skills = extract_required_skills(
    job_description,
    manual_skills
)

if required_skills:
    st.markdown("**Required skills identified**")
    st.write(", ".join(required_skills))
else:
    st.info("Add a job description or enter required skills.")




st.header("2. Upload Resumes")

uploaded_files = st.file_uploader(
    "Choose candidate resumes in PDF format",
    type=["pdf"],
    accept_multiple_files=True
)

analyse_button = st.button(
    "🔍 Analyse and Rank Candidates",
    type="primary",
    use_container_width=True
)


# ==================================================
# ANALYSE CANDIDATES
# ==================================================

if analyse_button:
    if not job_description.strip() and not required_skills:
        st.error("Please provide a job description or required skills.")

    elif not uploaded_files:
        st.error("Please upload at least one PDF resume.")

    else:
        results = []
        resume_texts = {}
        semantic_client = None
        job_vector = None
        semantic_error = None

        progress = st.progress(0)
        status = st.empty()

        try:
            semantic_client = get_gemini_client()

            if job_description.strip():
                job_vector = get_text_embedding(
                    semantic_client,
                    "Job requirements: " + job_description
                )

        except Exception as error:
            semantic_error = str(error)
            semantic_client = None
            job_vector = None

        for index, uploaded_file in enumerate(uploaded_files):
            status.write(
                f"Analysing {index + 1}/{len(uploaded_files)}: "
                f"{uploaded_file.name}"
            )

            try:
                resume_text = extract_pdf_text(uploaded_file)

                if not resume_text:
                    st.warning(
                        f"No readable text found in {uploaded_file.name}. "
                        "Scanned PDFs may require OCR."
                    )
                    progress.progress(
                        (index + 1) / len(uploaded_files)
                    )
                    continue

                candidate_name = os.path.splitext(
                    uploaded_file.name
                )[0]

                candidate_skills = find_skills(resume_text)

                skill_score, matched_skills = calculate_match(
                    candidate_skills,
                    required_skills
                )

                missing_skills = [
                    skill for skill in required_skills
                    if skill not in candidate_skills
                ]

                semantic_score = None
                semantic_status = "Unavailable"

                if semantic_client is not None and job_vector is not None:
                    try:
                        semantic_score = calculate_semantic_similarity(
                            semantic_client,
                            job_vector,
                            resume_text
                        )
                        semantic_status = "Available"

                    except Exception as error:
                        print(
                            f"Semantic matching error for "
                            f"{candidate_name}: {error}"
                        )

                if semantic_score is not None:
                    hybrid_score = round(
                        0.5 * skill_score + 0.5 * semantic_score,
                        1
                    )
                else:
                    hybrid_score = skill_score

                results.append({
                    "Candidate": candidate_name,
                    "Job Title": job_title,
                    "Department": department,
                    "Match Score (%)": skill_score,
                    "Semantic Similarity (%)": semantic_score,
                    "Hybrid Match (%)": hybrid_score,
                    "Semantic Status": semantic_status,
                    "Matched Skills": ", ".join(matched_skills),
                    "Skills Not Detected": ", ".join(missing_skills),
                    "Resume File": uploaded_file.name
                })

                resume_texts[candidate_name] = {
                    "text": resume_text,
                    "skills": candidate_skills,
                    "matched": matched_skills,
                    "missing": missing_skills
                }

            except Exception as error:
                st.error(
                    f"Could not analyse {uploaded_file.name}: {error}"
                )

            progress.progress((index + 1) / len(uploaded_files))

        if semantic_client is not None:
            semantic_client.close()

        progress.empty()
        status.empty()

        if results:
            results.sort(
                key=lambda item: item["Hybrid Match (%)"],
                reverse=True
            )

            for rank, item in enumerate(results, start=1):
                item["Rank"] = rank

            st.session_state["recruitment_results"] = pd.DataFrame(results)
            st.session_state["resume_texts"] = resume_texts
            st.session_state["required_skills"] = required_skills
            st.session_state["job_title"] = job_title
            st.session_state["job_description"] = job_description
            st.session_state["department"] = department

            st.success(
                f"Successfully analysed {len(results)} candidate(s)."
            )

            if semantic_error:
                st.warning(
                    "Semantic matching was unavailable. Skill-based "
                    "matching is still available. Check Gemini API access "
                    "if you want semantic scores."
                )

        else:
            st.error("No readable resumes could be analysed.")




if "recruitment_results" in st.session_state:
    df = st.session_state["recruitment_results"]

    st.divider()
    st.header("3. Recruitment Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Candidates", len(df))
    col2.metric(
        "Average Skill Match",
        f"{df['Match Score (%)'].mean():.1f}%"
    )
    col3.metric(
        "Average Hybrid Match",
        f"{df['Hybrid Match (%)'].mean():.1f}%"
    )
    col4.metric(
        "Highest Hybrid Match",
        f"{df['Hybrid Match (%)'].max():.1f}%"
    )

    st.caption(
        "Hybrid match is an experimental ranking indicator, not a "
        "probability of job success or a validated hiring score."
    )

    st.markdown(
        """
        <div style="margin: 25px 0 12px 0;">
            <div style="font-size: 25px; font-weight: 800; color: #242047;">
                Candidate Rankings
            </div>
            <div style="font-size: 14px; color: #777a91; margin-top: 5px;">
                Compare skills and semantic similarity across applicants.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    display_columns = [
        "Rank",
        "Candidate",
        "Match Score (%)",
        "Semantic Similarity (%)",
        "Hybrid Match (%)",
        "Semantic Status",
        "Matched Skills",
        "Skills Not Detected"
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Candidate Results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="candidate_matching_results.csv",
        mime="text/csv",
        use_container_width=True
    )


if "recruitment_results" in st.session_state:
    st.divider()
    st.header("4. Candidate Details")

    df = st.session_state["recruitment_results"]

    selected_candidate = st.selectbox(
        "Select a candidate",
        df["Candidate"].tolist()
    )

    candidate_row = df[
        df["Candidate"] == selected_candidate
    ].iloc[0]

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Skill Match",
        f"{candidate_row['Match Score (%)']:.1f}%"
    )

    semantic_value = candidate_row["Semantic Similarity (%)"]

    if pd.notna(semantic_value):
        metric2.metric(
            "Semantic Similarity",
            f"{semantic_value:.1f}%"
        )
    else:
        metric2.metric("Semantic Similarity", "Unavailable")

    metric3.metric(
        "Hybrid Match",
        f"{candidate_row['Hybrid Match (%)']:.1f}%"
    )

    candidate_data = st.session_state["resume_texts"][
        selected_candidate
    ]

    st.markdown("**Skills detected in resume**")

    if candidate_data["skills"]:
        st.write(", ".join(candidate_data["skills"]))
    else:
        st.write("No recognised skills detected.")

    st.markdown("**Required skills not detected**")

    if candidate_data["missing"]:
        st.write(", ".join(candidate_data["missing"]))
    else:
        st.write("No required skills are missing from the detected checklist.")

    with st.expander("View extracted resume text"):
        st.text(candidate_data["text"][:12000])

    if st.button(
        "✨ Generate AI Candidate Insights",
        use_container_width=True
    ):
        try:
            with st.spinner("Generating candidate insights..."):
                insights = generate_candidate_insights(
                    st.session_state["job_title"],
                    st.session_state["job_description"],
                    selected_candidate,
                    candidate_data["text"],
                    candidate_data["skills"],
                    st.session_state["required_skills"],
                    candidate_data["matched"]
                )

            st.session_state["current_candidate_insights"] = {
                "candidate": selected_candidate,
                "text": insights
            }

        except Exception as error:
            st.error(f"Could not generate insights: {error}")

    saved_insights = st.session_state.get(
        "current_candidate_insights"
    )

    if (
        saved_insights
        and saved_insights["candidate"] == selected_candidate
    ):
        st.subheader("AI Candidate Insights")
        st.markdown(saved_insights["text"])



if "recruitment_results" in st.session_state:
    st.divider()
    st.header("5. Interview Question Generator")

    if st.button(
        "📝 Generate Interview Questions",
        use_container_width=True
    ):
        try:
            with st.spinner("Generating interview questions..."):
                questions = generate_interview_questions(
                    st.session_state["job_title"],
                    st.session_state["job_description"],
                    st.session_state["required_skills"]
                )

            st.session_state["interview_questions"] = questions

        except Exception as error:
            st.error(f"Could not generate questions: {error}")

    if "interview_questions" in st.session_state:
        st.markdown(st.session_state["interview_questions"])




st.divider()

st.caption(
    "AI Recruitment & Candidate Matching Platform | "
    "Python • Streamlit • Google Gemini"
)

st.caption(
    "For decision support only. Verify candidate information and "
    "review applications fairly. Do not use automated scores as the "
    "sole basis for hiring decisions."
)
