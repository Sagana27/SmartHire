from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import docx

st.set_page_config(
    page_title="SmartHire | Intelligent Resume Screener",
    page_icon="💼",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

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

st.title("💼 SmartHire: Multi-Resume Screener & Matcher")
st.markdown("Automated batch resume screening, domain classification, and candidate job matching.")

with st.sidebar:
    st.header("⚙️ Matching Parameters")
    top_k = st.slider("Top Recommendations Per Candidate", min_value=1, max_value=10, value=3)
    min_score = st.slider("Minimum Match Threshold (%)", min_value=0, max_value=100, value=15, step=5)
    st.divider()
    st.caption("SmartHire ML Engine • TF-IDF + Classifier & Cosine Similarity")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📄 Candidate Upload")
    uploaded_files = st.file_uploader(
        "Upload one or more resumes (PDF / DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    manual_input = st.text_area(
        "Or paste resume text for a single candidate:",
        height=140,
        placeholder="Paste candidate skills, experience, or qualifications..."
    )

    run_btn = st.button("🚀 Screen All Resumes", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Screening Results")
    if run_btn:
        candidates = []
        
        if uploaded_files:
            for f in uploaded_files:
                txt = extract_text(f)
                if txt:
                    candidates.append({"name": f.name, "text": txt})
                else:
                    st.warning(f"Could not extract text from {f.name}")
        elif manual_input.strip():
            candidates.append({"name": "Manual Entry Candidate", "text": manual_input.strip()})

        if not candidates:
            st.warning("Please upload at least one resume file or paste text.")
        else:
            with st.spinner(f"Processing {len(candidates)} resume(s)..."):
                batch_summary = []
                
                for cand in candidates:
                    # 1. Classification & Confidence
                    vec = clf_tfidf.transform([cand["text"]])
                    pred_role = clf.predict(vec)[0]
                    confidence = float(np.max(clf.predict_proba(vec)[0]) * 100) if hasattr(clf, "predict_proba") else None
                    
                    # 2. Recommendations
                    r_vec = rec_tfidf.transform([cand["text"]])
                    sim_scores = cosine_similarity(r_vec, job_matrix).flatten()
                    top_idx = sim_scores.argsort()[::-1][:top_k]
                    
                    matched_roles = []
                    for idx in top_idx:
                        pct = sim_scores[idx] * 100
                        if pct >= min_score:
                            title = jobs.iloc[idx].get("title", jobs.iloc[idx].get("job_title", "Position"))
                            matched_roles.append(f"{title} ({pct:.1f}%)")
                    
                    batch_summary.append({
                        "Candidate / File": cand["name"],
                        "Predicted Domain": pred_role,
                        "Confidence": f"{confidence:.1f}%" if confidence else "N/A",
                        "Top Matches": ", ".join(matched_roles) if matched_roles else "No match above threshold"
                    })

                st.markdown(f"### Processed **{len(candidates)}** Candidate(s)")
                st.dataframe(pd.DataFrame(batch_summary), use_container_width=True)
