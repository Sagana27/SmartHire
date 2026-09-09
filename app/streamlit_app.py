from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import docx

# Page configuration
st.set_page_config(
    page_title="SmartHire | Intelligent Resume Screener",
    page_icon="💼",
    layout="wide"
)

# Robust path handling
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

def extract_text(file):
    """Extract clean text from PDF or DOCX uploads."""
    text = ""
    if file.name.endswith(".pdf"):
        reader = pypdf.PdfReader(file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = " ".join([p.text for p in doc.paragraphs])
    return text.strip()

@st.cache_resource
def load_resources():
    clf = joblib.load(MODELS_DIR / "classifier.pkl")
    clf_tfidf = joblib.load(MODELS_DIR / "clf_tfidf.pkl")
    rec_tfidf = joblib.load(MODELS_DIR / "rec_tfidf.pkl")
    
    jobs = pd.read_csv(DATA_DIR / "jobs_clean.csv")
    if "text" not in jobs.columns:
        desc_col = next((c for c in ['job_description', 'clean_job_description', 'description'] if c in jobs.columns), jobs.columns[-1])
        title_col = next((c for c in ['job_title', 'title'] if c in jobs.columns), jobs.columns[0])
        jobs["text"] = jobs[title_col].astype(str) + " " + jobs[desc_col].astype(str)
        
    job_matrix = rec_tfidf.transform(jobs["text"].fillna(""))
    return clf, clf_tfidf, rec_tfidf, jobs, job_matrix

try:
    clf, clf_tfidf, rec_tfidf, jobs, job_matrix = load_resources()
except Exception as e:
    st.error(f"Error loading models or datasets: {e}")
    st.stop()

# Header
st.title("💼 SmartHire: Intelligent Recruitment & Analytics")
st.markdown("Automated resume domain classification, job matching, and candidate readiness scoring.")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Matching Parameters")
    top_k = st.slider("Number of Job Recommendations", min_value=1, max_value=15, value=5)
    min_score = st.slider("Minimum Match Threshold (%)", min_value=0, max_value=100, value=15, step=5)
    st.divider()
    st.caption("SmartHire ML Engine • TF-IDF & MultinomialNB / Cosine Sim")

# Main dual-column layout
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.subheader("📄 Candidate Profile Input")
    uploaded_file = st.file_uploader("Upload candidate resume", type=["pdf", "docx"])
    
    manual_input = st.text_area(
        "Or paste resume text / qualifications directly:",
        height=200,
        placeholder="e.g., Data Scientist with experience in Python, SQL, Pandas, Scikit-Learn, and Machine Learning..."
    )
    
    resume_content = ""
    if uploaded_file is not None:
        resume_content = extract_text(uploaded_file)
        st.success(f"Attached file: {uploaded_file.name} ({len(resume_content.split())} words parsed)")
    elif manual_input.strip():
        resume_content = manual_input.strip()

    run_btn = st.button("🚀 Analyze Profile & Match Jobs", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Evaluation & Role Matches")
    if run_btn:
        if not resume_content:
            st.warning("Please upload a resume file or paste qualifications to begin analysis.")
        else:
            with st.spinner("Processing text and running inference..."):
                # Classification
                text_vec = clf_tfidf.transform([resume_content])
                predicted_role = clf.predict(text_vec)[0]
                
                # Confidence score
                confidence = None
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(text_vec)[0]
                    confidence = float(np.max(probs) * 100)

                # Metrics card
                m1, m2 = st.columns(2)
                m1.metric("Predicted Domain", predicted_role)
                m2.metric("Classifier Confidence", f"{confidence:.1f}%" if confidence else "N/A")

                # Cosine Similarity for Job Recommendation
                r_vec = rec_tfidf.transform([resume_content])
                similarity_scores = cosine_similarity(r_vec, job_matrix).flatten()
                
                top_indices = similarity_scores.argsort()[::-1][:top_k]
                results = []
                for idx in top_indices:
                    sim_pct = similarity_scores[idx] * 100
                    if sim_pct >= min_score:
                        results.append({
                            "Job Title": jobs.iloc[idx].get("title", jobs.iloc[idx].get("job_title", "Position")),
                            "Category": jobs.iloc[idx].get("category", "General"),
                            "Match Score": f"{sim_pct:.1f}%",
                            "Description": jobs.iloc[idx].get("job_description", "")[:120] + "..."
                        })

                st.markdown("### Top Matched Job Openings")
                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                else:
                    st.info("No available jobs exceeded your current minimum match score threshold.")
