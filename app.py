"""Resume Job Match Predictor - a small, explainable NLP project."""

from __future__ import annotations

import io
import re
from typing import Iterable

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILLS = {
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "SQL",
    "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Django", "Flask",
    "FastAPI", "Spring Boot", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
    "PyTorch", "Machine Learning", "Deep Learning", "Data Analysis", "Power BI",
    "Tableau", "Excel", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git",
    "Linux", "MongoDB", "PostgreSQL", "MySQL", "REST API", "GraphQL", "Spark",
    "Hadoop", "Airflow", "CI/CD", "Selenium", "Figma", "Agile", "Jira",
}


def normalize(text: str) -> str:
    """Normalize text while preserving meaningful technical tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\-/\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(uploaded_file: io.BytesIO) -> str:
    reader = PdfReader(uploaded_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def find_skills(text: str, skills: Iterable[str] = SKILLS) -> list[str]:
    normalized = normalize(text)
    found = []
    for skill in skills:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(found, key=str.lower)


def calculate_match(resume: str, job_description: str) -> dict:
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([normalize(resume), normalize(job_description)])
    similarity = float(cosine_similarity(matrix[0], matrix[1])[0][0])

    resume_skills = find_skills(resume)
    job_skills = find_skills(job_description)
    matched = sorted(set(resume_skills) & set(job_skills), key=str.lower)
    missing = sorted(set(job_skills) - set(resume_skills), key=str.lower)
    skill_coverage = len(matched) / len(job_skills) if job_skills else similarity

    # Blend lexical similarity with explicit required-skill coverage.
    weighted_score = similarity * 45 + skill_coverage * 55
    # When requirements list recognised skills, the score is their exact coverage.
    # For example, 1 matched skill out of 4 is always 25%.
    score = round(skill_coverage * 100) if job_skills else round(weighted_score * 100)
    category = "High Match" if score >= 70 else "Medium Match" if score >= 40 else "Low Match"
    return {
        "score": score,
        "category": category,
        "matched": matched,
        "missing": missing,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
    }


st.set_page_config(page_title="Resume Job Match Predictor", page_icon="R", layout="wide")
st.title("Resume Job Match Predictor")
st.caption("Upload a resume, add a job description, and get an explainable skill-based match score.")

left, right = st.columns(2)
with left:
    st.subheader("Resume")
    resume_pdf = st.file_uploader("Upload resume PDF", type=["pdf"])
    resume_text = st.text_area("Or paste resume text", height=260, placeholder="Paste your resume here...")
with right:
    st.subheader("Job Description")
    job_file = st.file_uploader("Upload job description text file", type=["txt"])
    job_description = st.text_area(
        "Paste the job description", height=350,
        placeholder="Example: Python, Machine Learning, SQL, Pandas, and Scikit-learn required.",
    )

if st.button("Analyze Match", type="primary", use_container_width=True):
    if job_file:
        try:
            job_description = f"{job_description}\n{job_file.getvalue().decode('utf-8')}".strip()
        except UnicodeDecodeError:
            st.error("The job description file must be UTF-8 encoded text.")
    if resume_pdf:
        try:
            extracted = extract_pdf_text(resume_pdf)
            resume_text = f"{resume_text}\n{extracted}".strip()
        except Exception as error:
            st.error(f"The PDF could not be read: {error}")
    if not resume_text.strip() or not job_description.strip():
        st.warning("Please provide both a resume and a job description.")
    else:
        result = calculate_match(resume_text, job_description)
        st.divider()
        score, category, skills = st.columns([1, 1, 2])
        score.metric("Match Score", f"{result['score']}%")
        category.metric("Prediction", result["category"])
        skills.metric("Required Skills Covered", f"{len(result['matched'])}/{len(result['job_skills'])}")

        matched_col, missing_col = st.columns(2)
        with matched_col:
            st.subheader("Matched Skills")
            if result["matched"]:
                st.success("\n".join(f"[Matched] {skill}" for skill in result["matched"]))
            else:
                st.info("No catalogued job skills were found in the resume.")
        with missing_col:
            st.subheader("Missing Skills")
            if result["missing"]:
                st.warning("\n".join(f"[Missing] {skill}" for skill in result["missing"]))
                st.info("Recommendation: focus on " + ", ".join(result["missing"][:3]) + ".")
            else:
                st.success("No listed skills are missing. Nice work.")

        with st.expander("Analysis details"):
            st.write("Resume skills detected:", ", ".join(result["resume_skills"]) or "None")
            st.write("Job skills detected:", ", ".join(result["job_skills"]) or "None")
            st.caption("Score equals required-skill coverage when job skills are detected; otherwise it uses TF-IDF text similarity.")
