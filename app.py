"""Resume Job Match Predictor - a small, explainable NLP project."""

from __future__ import annotations

import io
import re
from typing import Iterable

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


SKILLS = {
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "SQL",
    "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Django", "Flask",
    "FastAPI", "Spring Boot", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
    "PyTorch", "Machine Learning", "Deep Learning", "Data Analysis", "Power BI",
    "Tableau", "Excel", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git",
    "Linux", "MongoDB", "PostgreSQL", "MySQL", "REST API", "GraphQL", "Spark",
    "Hadoop", "Airflow", "CI/CD", "Selenium", "Figma", "Agile", "Jira",
    "Artificial Intelligence", "Agentic AI", "NLP", "Natural Language Processing",
    "Matplotlib", "Seaborn", "OpenCV", "JDBC", "GitHub", "Jenkins", "API",
    "Speech Recognition", "Data Preprocessing",
}

SKILL_ALIASES = {
    "Power BI": ("PowerBI",),
}

ROLE_PROFILES = {
    "software developer": (
        ("Programming languages", ("Python", "Java", "C++", "C", "JavaScript", "TypeScript", "C#")),
        ("Web development", ("React", "HTML", "CSS", "JavaScript", "Node.js", "Angular", "Vue")),
        ("Database", ("SQL", "MySQL", "PostgreSQL", "MongoDB", "JDBC")),
        ("Version control", ("Git", "GitHub")),
        ("Build and delivery tools", ("Jenkins", "Docker", "CI/CD")),
    ),
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
        variants = (skill, *SKILL_ALIASES.get(skill, ()))
        if any(
            re.search(
                r"(?<![a-z0-9])" + re.escape(variant.lower()) + r"(?![a-z0-9])",
                normalized,
            )
            for variant in variants
        ):
            found.append(skill)
    return sorted(found, key=str.lower)


def find_keywords(text: str) -> list[str]:
    """Extract meaningful words for matching non-catalogued job titles and terms."""
    words = re.findall(r"[a-z]+", normalize(text))
    return sorted(
        {word for word in words if len(word) > 1 and word not in ENGLISH_STOP_WORDS}
    )


def find_role_profile(job_description: str) -> tuple[str, ...] | None:
    normalized = normalize(job_description)
    for title, requirements in ROLE_PROFILES.items():
        if re.search(r"\b" + re.escape(title) + r"\b", normalized):
            return requirements
    return None


def calculate_match(resume: str, job_description: str) -> dict:
    resume_skills = find_skills(resume)
    job_skills = find_skills(job_description)
    matched = sorted(set(resume_skills) & set(job_skills), key=str.lower)
    missing = sorted(set(job_skills) - set(resume_skills), key=str.lower)
    job_terms = find_keywords(job_description)
    resume_terms = set(find_keywords(resume))
    matched_terms = sorted(set(job_terms) & resume_terms)
    missing_terms = sorted(set(job_terms) - resume_terms)

    role_profile = find_role_profile(job_description) if not job_skills else None
    if job_skills:
        required_items = job_skills
        matched_items = matched
        missing_items = missing
        match_basis = "job skills"
    elif role_profile:
        matched_items = [
            f"{label}: {', '.join(sorted(set(options) & set(resume_skills), key=str.lower))}"
            for label, options in role_profile
            if set(options) & set(resume_skills)
        ]
        missing_items = [
            label for label, options in role_profile
            if not set(options) & set(resume_skills)
        ]
        required_items = [label for label, _ in role_profile]
        match_basis = "general software developer role profile"
    else:
        required_items = job_terms
        matched_items = matched_terms
        missing_items = missing_terms
        match_basis = "job keywords"

    score = round(len(matched_items) / len(required_items) * 100) if required_items else 0
    category = "High Match" if score >= 70 else "Medium Match" if score >= 40 else "Low Match"
    return {
        "score": score,
        "category": category,
        "matched": matched,
        "missing": missing,
        "matched_items": matched_items,
        "missing_items": missing_items,
        "required_items": required_items,
        "match_basis": match_basis,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "job_terms": job_terms,
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
        if result["match_basis"] == "job skills":
            skills.metric(
                "Required Skills Covered",
                f"{len(result['matched_items'])}/{len(result['required_items'])}",
            )
        elif result["match_basis"] == "general software developer role profile":
            skills.metric(
                "Role Requirements Covered",
                f"{len(result['matched_items'])}/{len(result['required_items'])}",
            )
        else:
            skills.metric(
                "Job Keywords Matched",
                f"{len(result['matched_items'])}/{len(result['required_items'])}",
            )

        matched_col, missing_col = st.columns(2)
        with matched_col:
            st.subheader("Matched Skills" if result["match_basis"] != "job keywords" else "Matched Job Keywords")
            if result["matched_items"]:
                st.success("\n".join(f"[Matched] {item}" for item in result["matched_items"]))
            else:
                st.info("No matching job skills or keywords were found in the resume.")
        with missing_col:
            st.subheader("Missing Skills" if result["match_basis"] != "job keywords" else "Missing Job Keywords")
            if result["missing_items"]:
                st.warning("\n".join(f"[Missing] {item}" for item in result["missing_items"]))
                st.info("Recommendation: focus on " + ", ".join(result["missing_items"][:3]) + ".")
            else:
                st.success("No listed job skills or keywords are missing. Nice work.")

        with st.expander("Analysis details"):
            st.write("Resume skills detected:", ", ".join(result["resume_skills"]) or "None")
            st.write("Job skills detected:", ", ".join(result["job_skills"]) or "None")
            st.write("Job keywords detected:", ", ".join(result["job_terms"]) or "None")
            st.write("Matching method:", result["match_basis"])
            st.caption("A title-only Software Developer match uses broad skill groups and counts a group as covered when the resume contains at least one listed technology. When a description names skills, the score is based on those listed skills.")
