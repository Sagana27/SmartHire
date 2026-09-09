from pathlib import Path
import re
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import docx

st.set_page_config(
    page_title="SmartHire | Shortlist Decision Matrix",
    page_icon="💼",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

CORE_SKILLS_POOL = [
    "aws", "azure", "ci/cd", "c++", "c#", "deep learning", "docker", "git", 
    "hadoop", "java", "jenkins", "jira", "kubernetes", "linux", 
    "machine learning", "nlp", "nosql", "numpy", "pandas", "python", 
    "pytorch", "pytest", "r", "react", "scikit-learn", "selenium", 
    "spark", "spring boot", "sql", "statistics", "tableau", "tensorflow"
]
MAX_EXPECTED_SKILLS = 26

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

def extract_skills(text):
    text_lower = text.lower()
    found = [s.title() for s in CORE_SKILLS_POOL if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]
    return sorted(list(set(found)))

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

st.title("💼 SmartHire: Shortlist Decision Matrix")
st.caption("Batch parsing, domain entropy calculation, candidate readiness tiering, and Excel decision matrix generation.")

with st.sidebar:
    st.header("⚙️ Evaluation Parameters")
    interview_threshold = st.slider("Interview Probability Threshold (%)", 30, 90, 60, step=5)
    upskill_threshold = st.slider("Upskill Pipeline Threshold (%)", 20, 60, 40, step=5)
    st.divider()
    st.write(f"• Indexed Database Roles: **{len(jobs)}**")
    st.write(f"• Recognized Skill Library: **{len(CORE_SKILLS_POOL)} terms**")

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (PDF / DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if st.button("🚀 Generate Shortlist Decision Matrix", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Please upload one or more resumes to evaluate.")
    else:
        matrix_rows = []
        with st.spinner(f"Evaluating {len(uploaded_files)} candidate resumes..."):
            for f in uploaded_files:
                txt = extract_text(f)
                if not txt:
                    continue
                
                # 1. Classification & Domain Certainties
                vec = clf_tfidf.transform([txt])
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(vec)[0]
                    classes = clf.classes_
                    sorted_indices = np.argsort(probs)[::-1]
                    
                    p_domain = classes[sorted_indices[0]]
                    p_cert = float(probs[sorted_indices[0]] * 100)
                    
                    r_domain = classes[sorted_indices[1]] if len(classes) > 1 else "N/A"
                    r_cert = float(probs[sorted_indices[1]] * 100) if len(classes) > 1 else 0.0
                    
                    # Shannon Entropy (base 2)
                    clean_probs = probs[probs > 0]
                    shannon_ent = float(entropy(clean_probs, base=2))
                else:
                    pred = clf.predict(vec)[0]
                    p_domain, p_cert, r_domain, r_cert, shannon_ent = pred, 50.0, "N/A", 0.0, 1.0

                # 2. Skill Extraction & Readiness Score
                skills = extract_skills(txt)
                skills_cnt = len(skills)
                readiness_score = round(min((skills_cnt / MAX_EXPECTED_SKILLS) * 100, 100.0), 2)
                
                # Specialization Index (%)
                spec_index = round(abs(p_cert - r_cert), 1)

                # 3. Job Recommendation Match Score
                r_vec = rec_tfidf.transform([txt])
                sims = cosine_similarity(r_vec, job_matrix).flatten()
                best_job_idx = int(sims.argsort()[::-1][0])
                top_job_title = jobs.iloc[best_job_idx].get("title", jobs.iloc[best_job_idx].get("job_title", "General Role"))
                job_match_score = round(float(sims[best_job_idx] * 100), 2)

                # 4. Shortlist Probability & Action
                shortlist_prob = round((0.45 * job_match_score) + (0.35 * readiness_score) + (0.20 * p_cert), 2)
                
                if shortlist_prob >= interview_threshold:
                    cand_status = "Interview Candidate"
                    rec_action = "Schedule Technical Screening"
                    tier = "Tier 1 (High)"
                elif shortlist_prob >= upskill_threshold:
                    cand_status = "Upskill Candidate"
                    rec_action = "Retain for Associate Pipeline"
                    tier = "Tier 2 (Mid)" if readiness_score >= 40 else "Tier 3 (Developing)"
                else:
                    cand_status = "Out of Scope"
                    rec_action = "Archive Profile"
                    tier = "Tier 3 (Developing)"

                matrix_rows.append({
                    "Filename": f.name,
                    "Status": "Success",
                    "Shortlist Probability (%)": shortlist_prob,
                    "Candidate Status": cand_status,
                    "Recommended Action": rec_action,
                    "Primary Domain": p_domain,
                    "Domain Certainty (%)": round(p_cert, 2),
                    "Runner-up Domain": r_domain,
                    "Runner-up Certainty (%)": round(r_cert, 2),
                    "Shannon Entropy (bits)": round(shannon_ent, 2),
                    "Specialization Index (%)": spec_index,
                    "Readiness Score (%)": readiness_score,
                    "Candidate Tier": tier,
                    "Skills Count": skills_cnt,
                    "Identified Skills": ", ".join(skills),
                    "Top Matched Job": top_job_title,
                    "Job Match Score (%)": job_match_score
                })

        df_matrix = pd.DataFrame(matrix_rows)

        # 5. Cohort Percentile Ranking
        if not df_matrix.empty:
            df_matrix = df_matrix.sort_values(by="Shortlist Probability (%)", ascending=False).reset_index(drop=True)
            n_cands = len(df_matrix)
            percentiles = [f"Top {int(np.ceil(((i + 1) / n_cands) * 4) * 25)}%" for i in range(n_cands)]
            df_matrix["Cohort Percentile"] = percentiles

            st.subheader("📋 Decision Matrix Preview")
            st.dataframe(df_matrix, use_container_width=True)

            # Generate downloadable Excel matching 'Shortlist_Rankings' sheet format
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_matrix.to_excel(writer, index=False, sheet_name="Shortlist_Rankings")
            excel_data = excel_buffer.getvalue()

            st.download_button(
                label="📥 Download SmartHire_Shortlist_Decision_Matrix.xlsx",
                data=excel_data,
                file_name="SmartHire_Shortlist_Decision_Matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
