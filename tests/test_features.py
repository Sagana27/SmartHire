from src.features.match_features import skill_overlap


def test_skill_overlap_is_case_insensitive():
    assert skill_overlap("Python and SQL", ["python", "sql", "java"]) == ["python", "sql"]
