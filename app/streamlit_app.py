from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import re
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SmartHire | AI Recruitment Platform", layout="wide")

# Cached resource loaders
@st.cache_resource
def load_models_and_data():
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir = Path("../models")

    data_dir = Path("data/processed")
    if not data_dir.exists():
        data_dir = Path("../data/processed")

    clf = joblib.load(models_dir / "classifier.pkl")
    clf_tfidf = joblib.load(models_dir / "tfidf_vectorizer.pkl")
    rec_tfidf = joblib.load(models_dir / "job_tfidf_vectorizer.pkl")

    jobs = pd.read_csv(data_dir / "jobs_clean.csv")
    job_matrix = rec_tfidf.transform(jobs["text"].fillna(""))

    return clf, clf_tfidf, rec_tfidf, jobs, job_matrix

clf, clf_tfidf, rec_tfidf, jobs, job_matrix = load_models_and_data()

# Skill gap analysis taxonomy
TECH_SKILLS = [
    "python", "r", "sql", "machine learning", "deep learning", "nlp",
    "data analysis", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "tableau", "power bi", "matplotlib", "seaborn", "statistics", "aws",
    "docker", "kubernetes", "git", "spark", "hadoop", "excel", "java", "spring boot"
]

# UI Header
st.title("SmartHire: Intelligent Recruitment & Analytics")
st.markdown("Automated resume domain classification, job matching, and candidate readiness scoring.")

# Input layout
col_input, col_config = st.columns([2, 1])

with col_input:
    resume_input = st.text_area(
        "Paste Resume Text / Qualifications:",
        height=220,
        placeholder="e.g., Data Scientist with experience in Python, SQL, Pandas, Scikit-Learn, and Machine Learning..."
    )

with col_config:
    st.subheader("Preferences")
    top_n = st.slider("Number of Job Recommendations:", min_value=3, max_value=15, value=5)
    target_role = st.text_input("Target Role for Skill Gap Analysis:", value="Data Scientist")
    analyze_btn = st.button("Analyze Profile", type="primary", use_container_width=True)

if analyze_btn and resume_input.strip():
    clean_resume = resume_input.lower()

    # 1. Classification
    vec_clf = clf_tfidf.transform([clean_resume])
    predicted_cat = clf.predict(vec_clf)[0]
    confidence = clf.predict_proba(vec_clf).max() * 100

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Domain", predicted_cat)
    m2.metric("Classification Confidence", f"{confidence:.2f}%")

    # 2. Skill Gap Analysis
    matched_role_jobs = jobs[jobs["title"].str.contains(target_role, case=False, na=False)]
    if len(matched_role_jobs) > 0:
        role_corpus = " ".join(matched_role_jobs["text"].dropna().str.lower())
        demanded = [s for s in TECH_SKILLS if re.search(r"\b" + re.escape(s) + r"\b", role_corpus)]

        present = [s for s in demanded if re.search(r"\b" + re.escape(s) + r"\b", f" {clean_resume} ")]
        missing = [s for s in demanded if s not in present]
        readiness = (len(present) / max(len(demanded), 1)) * 100
        m3.metric("Role Readiness Score", f"{readiness:.1f}%")
    else:
        present, missing = [], []
        m3.metric("Role Readiness Score", "N/A")

    tab_rec, tab_gap = st.tabs(["Top Recommended Jobs", "Skill Gap Breakdown"])

    with tab_rec:
        vec_rec = rec_tfidf.transform([clean_resume])
        sim_scores = cosine_similarity(vec_rec, job_matrix).flatten()
        top_idx = sim_scores.argsort()[-top_n:][::-1]

        results = jobs.iloc[top_idx][["title", "company", "location"]].copy()

        # Handle null, None, nan, and empty strings
        missing_values = ["None", "none", "nan", "NaN", "", "null", "Null"]
        results["company"] = results["company"].fillna("Not Specified").replace(missing_values, "Not Specified")
        results["location"] = results["location"].fillna("Remote / Unspecified").replace(missing_values, "Remote / Unspecified")
        results["Match Score (%)"] = (sim_scores[top_idx] * 100).round(2)

        st.dataframe(results.reset_index(drop=True), use_container_width=True)

    with tab_gap:
        col_present, col_missing = st.columns(2)
        with col_present:
            st.success(f"**Identified Competencies ({len(present)})**")
            for skill in present:
                st.write(f"- {skill.title()}")
        with col_missing:
            st.warning(f"**Recommended Skills to Acquire ({len(missing)})**")
            for skill in missing:
                st.write(f"- {skill.title()}")

elif analyze_btn and not resume_input.strip():
    st.error("Please paste resume text before running the analysis.")