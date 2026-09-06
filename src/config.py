from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Raw data paths
DATA_RAW = BASE_DIR / "data" / "raw"
RESUME_CSV = DATA_RAW / "UpdatedResumeDataSet.csv"
NAUKRI_CSV = DATA_RAW / "naukri_com-job_sample.csv"
LINKEDIN_POSTINGS = DATA_RAW / "linkedin" / "postings.csv"

# Interim and processed paths
DATA_INTERIM = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

JOB_CORPUS_CSV = DATA_INTERIM / "job_corpus.csv"
JOBS_CLEAN_CSV = DATA_PROCESSED / "jobs_clean.csv"
RESUMES_CLEAN_CSV = DATA_PROCESSED / "resumes_clean.csv"