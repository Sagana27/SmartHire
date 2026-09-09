import os
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="SmartHire Candidate Screening", layout="wide")

MODELS_DIR = Path("models")
DATA_DIR = Path("data/processed")

GLOBAL_TECH_SKILLS = [
    "python", "r", "sql", "machine learning", "deep learning", "nlp",
    "data analysis", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "tableau", "power bi", "matplotlib", "seaborn", "statistics", "aws",
    "docker", "kubernetes", "git", "spark", "hadoop", "excel", "java", "spring boot"
]

@st.cache_resource
def load_artifacts():
    clf = joblib.load(MODELS_DIR / "classifier.pkl")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    rec_tfidf = joblib.load(MODELS_DIR / "job_tfidf_vectorizer.pkl")
    df_jobs = pd.read_csv(DATA_DIR / "jobs_clean.csv")
    rec_job_col = "clean_job_text" if "clean_job_text" in df_jobs.columns else "text"
    job_tfidf_vectors = rec_tfidf.transform(df_jobs[rec_job_col].fillna("").astype(str))
    return clf, tfidf, rec_tfidf, df_jobs, job_tfidf_vectors

classifier, tfidf, rec_tfidf, df_jobs, job_tfidf_vectors = load_artifacts()
classes = classifier.classes_
max_entropy = float(np.log2(len(classes)))

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+\s*", " ", text)
    text = re.sub(r"[@#]\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join([p.extract_text() or "" for p in reader.pages]).strip()
    elif name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()
    return ""

def extract_skills(text: str) -> set:
    found = set()
    cleaned = text.lower()
    for s in GLOBAL_TECH_SKILLS:
        if re.search(r"\b" + re.escape(s) + r"\b", cleaned):
            found.add(s)
    return found

def calculate_entropy(probabilities: np.ndarray) -> float:
    valid_probs = probabilities[probabilities > 0]
    return float(entropy(valid_probs, base=2))

def evaluate_candidate(domain_p: float, readiness: float, sim: float):
    prob = (0.45 * domain_p) + (0.35 * readiness) + (0.20 * sim)
    prob = float(np.clip(prob, 0.0, 99.9))
    if prob >= 75.0:
        status, action = "Strong Contender", "Fast-track to Technical Interview"
    elif prob >= 55.0:
        status, action = "High-Potential Pivot", "Schedule Phone Screening"
    elif prob >= 40.0:
        status, action = "Upskill Candidate", "Retain for Associate Pipeline"
    else:
        status, action = "Out of Scope", "Archive Profile"
    return round(prob, 2), status, action

st.title("SmartHire Candidate Screening & Batch Decision Matrix")

uploaded_files = st.file_uploader("Upload Resumes (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Generate Shortlist Decision Matrix"):
        records = []
        for file in uploaded_files:
            raw = extract_text_from_upload(file)
            if not raw:
                continue
            cleaned = clean_text(raw)
            res_vec = tfidf.transform([cleaned])
            probs = classifier.predict_proba(res_vec)[0]
            domain = classifier.predict(res_vec)[0]
            domain_p = float(np.max(probs)) * 100
            entropy_val = calculate_entropy(probs)
            spec_idx = max(0.0, (1.0 - (entropy_val / max_entropy)) * 100)

            skills = extract_skills(cleaned)
            readiness = (len(skills) / len(GLOBAL_TECH_SKILLS)) * 100

            cand_job_vec = rec_tfidf.transform([cleaned])
            sims = cosine_similarity(cand_job_vec, job_tfidf_vectors).flatten()
            top_idx = sims.argmax()
            top_score = round(float(sims[top_idx]) * 100, 2)
            top_job = df_jobs.iloc[top_idx].get("title", "Not Specified")

            p_shortlist, status, action = evaluate_candidate(domain_p, readiness, top_score)

            records.append({
                "Filename": file.name,
                "Shortlist Probability (%)": p_shortlist,
                "Verdict": status,
                "Recommended Action": action,
                "Primary Domain": domain,
                "Domain Certainty (%)": round(domain_p, 2),
                "Specialization Index (%)": round(spec_idx, 1),
                "Taxonomy Coverage (%)": round(readiness, 2),
                "Skills Count": len(skills),
                "Identified Skills": ", ".join(sorted([s.title() for s in skills])),
                "Top Role Match": top_job,
                "Role Similarity (%)": top_score
            })

        if records:
            df = pd.DataFrame(records).sort_values(by="Shortlist Probability (%)", ascending=False).reset_index(drop=True)
            df["Cohort Percentile"] = [f"Top {max(1, int(round((i + 1) / len(df) * 100)))}%" for i in range(len(df))]

            st.subheader("Shortlist Decision Matrix")
            st.dataframe(df, use_container_width=True)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)

            st.download_button(
                label="Download SmartHire_Batch_Rankings.xlsx",
                data=buffer.getvalue(),
                file_name="SmartHire_Batch_Rankings.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
