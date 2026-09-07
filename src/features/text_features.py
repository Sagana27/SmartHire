"""Text vectorization helpers shared by notebooks and application code."""

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer(**kwargs) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with project defaults."""
    defaults = {"stop_words": "english", "ngram_range": (1, 2)}
    defaults.update(kwargs)
    return TfidfVectorizer(**defaults)
