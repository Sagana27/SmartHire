"""Candidate-to-job matching feature helpers."""

import re


def skill_overlap(text: str, skills: list[str]) -> list[str]:
    """Return skills present in text using case-insensitive word matching."""
    normalized = text.lower()
    return [
        skill
        for skill in skills
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", normalized)
    ]
