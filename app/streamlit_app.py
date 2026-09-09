from pathlib import Path
import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import docx

st.set_page_config(
    page_title="SmartHire | AI Recruitment & Screening",
    page_icon="🎯",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

CORE_SKILLS = [
    "python", "java", "c++", "c#", "sql", "nosql", "pandas", "numpy", 
    "scikit-learn", "deep learning", "machine learning", "nlp", "transformers", 
    "aws", "azure", "docker", "kubernetes", "ci/cd", "linux", "git", 
    "selenium", "pytest", "jira", "spring boot", "react", "tableau"
]

def extract_text(file):
    text = ""
    try:
        if file.name.endswith(".pdf"):
            reader = pypdf.PdfReader(file)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = " ".join([p.text for p in doc.paragraphs])
    except Exception as e:
        st.error(f"Error parsing {file.name}: {e}")
    return text.strip()

def detect_skills(text):
    text_lower = text.lower()
    found = [s.title() for s in CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]
    return found[:8]

@st.cache_resource
def load_resources():
    clf = joblib.load(MODELS_DIR / "classifier.pkl")
    clf_tfidf = joblib.load(MODELS_DIR / "clf_tfidf.pkl")
    rec_tfidf = joblib.load(MODELS_DIR / "rec_tfidf.pkl")
    
    jobs = pd.read_csv(DATA_DIR / "jobs_clean.csv")
    if "text" not in jobs.columns:
        desc = next((c for c in ['job_description', 'clean_job_description'] if c in jobs.columns), jobs.columns[-1])
        title = next((c for c in ['job_title', 'title'] if c in jobs.columns), jobs.columns[0])
        jobs["text"] = jobs[title].astype(str) + " " + jobs[desc].astype(str)
        
    job_matrix = rec_tfidf.transform(jobs["text"].fillna(""))
    return clf, clf_tfidf, rec_tfidf, jobs, job_matrix

try:
    clf, clf_tfidf, rec_tfidf, jobs, job_matrix = load_resources()
except Exception as e:
    st.error(f"Error loading models or datasets: {e}")
    st.stop()

st.title("🎯 SmartHire: Automated Candidate Screening")
st.caption("AI-powered batch screening, domain classification, and semantic role alignment.")

with st.sidebar:
    st.header("⚙️ Screening Controls")
    top_k = st.slider("Top Recommendations per Candidate", min_value=1, max_value=8, value=3)
    min_score = st.slider("Minimum Match Threshold (%)", min_value=0, max_value=100, value=20, step=5)
    st.markdown("---")
    st.markdown("**Engine Details**")
    st.write(f"• Active Positions: **{len(jobs)}**")
    st.write("• Vectorizer: TF-IDF (1-2 ngrams)")

col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload one or multiple resumes (PDF / DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    run_btn = st.button("Run Batch Evaluation", type="primary", use_container_width=True)

with col_right:
    st.subheader("Screening Overview")
    if run_btn:
        if not uploaded_files:
            st.warning("Please attach at least one PDF or DOCX resume to analyze.")
        else:
            summary_records = []
            detailed_results = []
            
            with st.spinner(f"Evaluating {len(uploaded_files)} resumes..."):
                for f in uploaded_files:
                    txt = extract_text(f)
                    if not txt:
                        continue
                        
                    vec = clf_tfidf.transform([txt])
                    pred_role = clf.predict(vec)[0]
                    
                    r_vec = rec_tfidf.transform([txt])
                    sim_scores = cosine_similarity(r_vec, job_matrix).flatten()
                    top_idx = sim_scores.argsort()[::-1][:top_k]
                    
                    best_match = ""
                    best_pct = 0.0
                    matches_list = []
                    
                    for i, idx in enumerate(top_idx):
                        pct = sim_scores[idx] * 100
                        title = jobs.iloc[idx].get("title", jobs.iloc[idx].get("job_title", "Position"))
                        if i == 0:
                            best_match = title
                            best_pct = pct
                        if pct >= min_score:
                            matches_list.append((title, pct, jobs.iloc[idx].get("category", "General")))
                    
                    skills = detect_skills(txt)
                    
                    summary_records.append({
                        "Candidate": f.name,
                        "Predicted Domain": pred_role,
                        "Top Role": best_match,
                        "Match Score": f"{best_pct:.1f}%",
                        "Key Skills": ", ".join(skills) if skills else "N/A"
                    })
                    
                    detailed_results.append({
                        "name": f.name,
                        "role": pred_role,
                        "skills": skills,
                        "matches": matches_list,
                        "text": txt[:400] + "..."
                    })

            df_summary = pd.DataFrame(summary_records)
            st.dataframe(df_summary, use_container_width=True)
            
            csv_data = df_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Screening Report to CSV",
                data=csv_data,
                file_name="smarthire_screening_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("### Candidate Breakdowns")
            for cand in detailed_results:
                with st.expander(f"📌 {cand['name']} — {cand['role']}"):
                    st.write("**Identified Core Skills:**")
                    if cand["skills"]:
                        st.markdown(" ".join([f"`{s}`" for s in cand["skills"]]))
                    else:
                        st.caption("No standard keywords identified.")
                    
                    st.write("**Ranked Matches:**")
                    for m_title, m_score, m_cat in cand["matches"]:
                        st.progress(min(int(m_score), 100), text=f"{m_title} ({m_cat}) — {m_score:.1f}%")
                    
                    st.caption(f"Preview: {cand['text']}")
