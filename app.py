import io
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from fpdf import FPDF
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="SmartHire | Resume Matcher & Fit Predictor",
    page_icon="💼",
    layout="wide",
)

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
    classifier = joblib.load(MODELS_DIR / "classifier.pkl")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    rec_tfidf = joblib.load(MODELS_DIR / "job_tfidf_vectorizer.pkl")
    fit_rf = joblib.load(MODELS_DIR / "fit_predictor.pkl")
    df_jobs = pd.read_csv(DATA_DIR / "jobs_clean.csv")
    return classifier, tfidf, rec_tfidf, fit_rf, df_jobs

try:
    classifier, tfidf, rec_tfidf, fit_rf, df_jobs = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

def extract_text_from_file(uploaded_file) -> str:
    file_bytes = io.BytesIO(uploaded_file.read())
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".pdf"):
        reader = PdfReader(file_bytes)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text).strip()
    elif filename.endswith(".docx"):
        doc = Document(file_bytes)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs).strip()
    elif filename.endswith(".txt"):
        return file_bytes.getvalue().decode("utf-8", errors="ignore").strip()
    return ""

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+\s*", " ", text)
    text = re.sub(r"[#@]\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def extract_skills(text: str, skill_pool: list) -> set:
    found = set()
    cleaned = text.lower()
    for skill in skill_pool:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned):
            found.add(skill)
    return found

def generate_pdf_report(domain_pred: str, domain_proba: float, readiness_score: float, 
                        fit_tier: str, identified_skills: set, missing_skills: set, 
                        df_rec: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "SmartHire - Candidate Evaluation Report", ln=True, align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "Generated via Automated Skill Taxonomy & Matching Pipeline", ln=True, align="C")
    pdf.ln(8)
    
    # Executive KPIs
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Executive KPIs & Alignment", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"- Predicted Domain: {domain_pred} (Confidence: {domain_proba:.2f}%)", ln=True)
    pdf.cell(0, 6, f"- Taxonomy Coverage: {readiness_score:.1f}% ({len(identified_skills)}/{len(GLOBAL_TECH_SKILLS)} Competencies)", ln=True)
    pdf.cell(0, 6, f"- Candidate Tier: {fit_tier}", ln=True)
    pdf.ln(4)
    
    # Skill Gap
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Skill Assessment", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Identified Competencies:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, ", ".join(sorted([s.title() for s in identified_skills])) if identified_skills else "None detected")
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Target Skills to Acquire:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, ", ".join(sorted([s.title() for s in missing_skills])) if missing_skills else "None")
    pdf.ln(4)
    
    # Top Jobs
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "3. Top Matched Vacancies", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for idx, row in df_rec.head(5).iterrows():
        title = str(row.get("title", "N/A"))
        company = str(row.get("company", "N/A"))
        match_s = row.get("Match Score (%)", 0.0)
        pdf.cell(0, 5, f"{idx+1}. {title} | {company} - Match: {match_s}%", ln=True)
    
    return bytes(pdf.output())

# Streamlit UI
st.title("💼 SmartHire Executive Pipeline Engine")
st.markdown("Automated resume domain classification, role readiness analysis, and candidate matching.")

st.sidebar.header("Configuration")
input_method = st.sidebar.radio("Input Format:", ["Upload Document", "Direct Text Input"])
top_k_jobs = st.sidebar.slider("Recommended Jobs Count:", min_value=3, max_value=15, value=5)

resume_raw_text = ""

if input_method == "Upload Document":
    uploaded_file = st.file_uploader("Upload candidate resume", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        resume_raw_text = extract_text_from_file(uploaded_file)
        if resume_raw_text:
            st.success(f"Parsed {uploaded_file.name} ({len(resume_raw_text.split())} words)")
            with st.expander("Inspect Parsed Document Text"):
                st.write(resume_raw_text[:1500] + ("..." if len(resume_raw_text) > 1500 else ""))
        else:
            st.error("No extractable plain text found inside the uploaded document.")
else:
    resume_raw_text = st.text_area(
        "Paste Resume Content:",
        height=220,
        placeholder="Paste plain text resume content here..."
    )

if st.button("Analyze Profile", type="primary"):
    if not resume_raw_text.strip():
        st.warning("Please submit a valid resume input before analyzing.")
        st.stop()

    cleaned_resume = clean_text(resume_raw_text)

    # 1. Domain Classification
    res_vec = tfidf.transform([cleaned_resume])
    domain_pred = classifier.predict(res_vec)[0]
    domain_proba = np.max(classifier.predict_proba(res_vec)) * 100

    # 2. Competencies & Readiness Score
    identified_skills = extract_skills(cleaned_resume, GLOBAL_TECH_SKILLS)
    missing_skills = set(GLOBAL_TECH_SKILLS) - identified_skills
    readiness_score = (len(identified_skills) / len(GLOBAL_TECH_SKILLS)) * 100

    # 3. Content-Based Job Recommendation
    rec_job_col = "clean_job_text" if "clean_job_text" in df_jobs.columns else "text"
    job_tfidf_vectors = rec_tfidf.transform(df_jobs[rec_job_col].fillna("").astype(str))
    res_rec_vec = rec_tfidf.transform([cleaned_resume])
    
    similarities = cosine_similarity(res_rec_vec, job_tfidf_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_k_jobs]

    rec_job_rows = []
    res_word_count = len(cleaned_resume.split())

    for idx in top_indices:
        job_record = df_jobs.iloc[idx]
        job_text = str(job_record[rec_job_col]).lower()
        job_skills = extract_skills(job_text, GLOBAL_TECH_SKILLS)
        shared_skills = identified_skills.intersection(job_skills)
        match_ratio = len(shared_skills) / max(len(job_skills), 1)

        fit_features = np.array([[
            similarities[idx],
            len(shared_skills),
            match_ratio,
            res_word_count,
            len(job_text.split())
        ]])

        try:
            fit_probability = fit_rf.predict_proba(fit_features)[0][1] * 100
        except Exception:
            fit_probability = similarities[idx] * 100

        rec_job_rows.append({
            "title": job_record.get("title", "Not Specified"),
            "company": job_record.get("company", "Not Specified"),
            "location": job_record.get("location", "Not Specified"),
            "Match Score (%)": round(similarities[idx] * 100, 2),
            "Fit Likelihood (%)": round(fit_probability, 2)
        })

    df_recommended = pd.DataFrame(rec_job_rows)

    # 4. Executive KPI Metrics Bar
    st.subheader("📊 Executive Candidate KPIs")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Predicted Role", str(domain_pred))
    kpi2.metric("Domain Fit", f"{domain_proba:.1f}%")
    kpi3.metric("Taxonomy Coverage", f"{readiness_score:.1f}%")
    kpi4.metric("Competency Count", f"{len(identified_skills)} / {len(GLOBAL_TECH_SKILLS)}")

    # 5. Dynamic Profile Synthesis Summary
    top_match_title = df_recommended.iloc[0]["title"] if not df_recommended.empty else "N/A"
    top_match_score = df_recommended.iloc[0]["Match Score (%)"] if not df_recommended.empty else 0.0

    fit_tier = (
        "Tier 1: High Market Alignment" if readiness_score >= 60 
        else "Tier 2: Solid Mid-Level Profile" if readiness_score >= 35 
        else "Tier 3: Early-Stage Competency"
    )

    st.markdown(
        f"""
        > **Executive Assessment Summary**
        > 
        > * **Candidate Tier**: **{fit_tier}** ({readiness_score:.1f}% taxonomy coverage).
        > * **Classification**: Primary specialization mapped to **{domain_pred}** with **{domain_proba:.2f}%** model certainty.
        > * **Core Competencies**: Verified across **{len(identified_skills)} technical areas**, including `{", ".join(sorted(list(identified_skills))[:5])}`.
        > * **Priority Job Match**: Closest vacancy is **{top_match_title}** ({top_match_score}% content match).
        > * **Target Upskilling**: Recommended next steps include adding `{", ".join(list(missing_skills)[:4])}`.
        """
    )

    # 6. PDF Report Download Button
    pdf_report_bytes = generate_pdf_report(
        domain_pred=domain_pred,
        domain_proba=domain_proba,
        readiness_score=readiness_score,
        fit_tier=fit_tier,
        identified_skills=identified_skills,
        missing_skills=missing_skills,
        df_rec=df_recommended
    )

    st.download_button(
        label="📥 Download Candidate Assessment Report (PDF)",
        data=pdf_report_bytes,
        file_name=f"SmartHire_Report_{domain_pred.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )

    st.markdown("---")

    # 7. Analysis Tabs
    tab_jobs, tab_skills = st.tabs(["🎯 Top Recommended Jobs", "🔍 Skill Gap Breakdown"])

    with tab_jobs:
        st.dataframe(df_recommended, use_container_width=True)

    with tab_skills:
        col_id, col_miss = st.columns(2)
        with col_id:
            st.success(f"**Identified Competencies ({len(identified_skills)})**")
            for skill in sorted(identified_skills):
                st.markdown(f"- {skill.title()}")
        with col_miss:
            st.warning(f"**Recommended Skills to Acquire ({len(missing_skills)})**")
            for skill in sorted(missing_skills):
                st.markdown(f"- {skill.title()}")