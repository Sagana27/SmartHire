"""
Download the SmartHire datasets from Kaggle into data/raw/.

Run this ONCE to set up your data:

    pip install kagglehub
    python download_data.py

The first run asks for your Kaggle credentials (username + API key).
Get them at: Kaggle -> profile -> Settings -> API -> Create New Token
(that downloads kaggle.json; your username and key are inside it).

After it finishes, data/raw/ will contain:
    UpdatedResumeDataSet.csv        resume classifier
    naukri_com-job_sample.csv       job recommender + skill-gap
    linkedin/postings.csv           extra jobs (recommender)
    linkedin/jobs/job_skills.csv
    linkedin/mappings/skills.csv
"""

import glob
import shutil
from pathlib import Path

import kagglehub

RAW = Path(__file__).resolve().parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Datasets that are a single CSV -> copied flat into data/raw/.
FLAT_DATASETS = [
    "jillanisofttech/updated-resume-dataset",
    "PromptCloudHQ/jobs-on-naukricom",
]

# LinkedIn is a folder of files; we keep only these three, preserving structure.
LINKEDIN_SLUG = "arshkon/linkedin-job-postings"
LINKEDIN_KEEP = {
    "postings.csv": "postings.csv",
    "job_skills.csv": "jobs/job_skills.csv",
    "skills.csv": "mappings/skills.csv",
}


def _find(cache_dir, filename):
    hits = glob.glob(str(Path(cache_dir) / "**" / filename), recursive=True)
    return hits[0] if hits else None


def download_flat():
    for slug in FLAT_DATASETS:
        print(f"Downloading {slug} ...")
        cache_dir = kagglehub.dataset_download(slug)
        csvs = glob.glob(str(Path(cache_dir) / "**" / "*.csv"), recursive=True)
        if not csvs:
            print(f"  WARNING: no CSV found (files are in {cache_dir})")
        for csv in csvs:
            shutil.copy(csv, RAW / Path(csv).name)
            print(f"  copied -> {Path(csv).name}")


def download_linkedin():
    print(f"Downloading {LINKEDIN_SLUG} ...")
    cache_dir = kagglehub.dataset_download(LINKEDIN_SLUG)
    for filename, rel_dest in LINKEDIN_KEEP.items():
        src = _find(cache_dir, filename)
        if not src:
            print(f"  WARNING: {filename} not found in download")
            continue
        dest = RAW / "linkedin" / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        print(f"  copied -> linkedin/{rel_dest}")


def main():
    download_flat()
    download_linkedin()

    print("\nDone. data/raw/ now contains:")
    for f in sorted(RAW.rglob("*.csv")):
        print(" -", f.relative_to(RAW).as_posix())


if __name__ == "__main__":
    main()
