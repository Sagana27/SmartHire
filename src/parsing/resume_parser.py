"""Extract resume text from TXT, PDF, and DOCX files."""

from pathlib import Path


def extract_text(path: str | Path) -> str:
    """Extract text from a supported resume file."""
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".txt":
        return source.read_text(encoding="utf-8")
    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(source) as document:
            return "\n".join(page.extract_text() or "" for page in document.pages)
    if suffix == ".docx":
        from docx import Document

        return "\n".join(paragraph.text for paragraph in Document(source).paragraphs)

    raise ValueError(f"Unsupported resume format: {suffix}")
