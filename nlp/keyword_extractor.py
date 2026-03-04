"""
keyword_extractor.py
Extracts skills, tools, technologies, soft skills, and qualifications
from both resume and job description text using:
  - A curated skills taxonomy
  - spaCy NER for named entities (ORG, PRODUCT, etc.)
  - TF-IDF for domain-specific keyword extraction
"""

import re
from collections import Counter
from typing import Set, List, Dict

# ── Curated Skills Taxonomy ────────────────────────────────────────────────────

TECH_SKILLS = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
    # Web
    "react", "angular", "vue", "nextjs", "nodejs", "express", "django",
    "flask", "fastapi", "html", "css", "sass", "tailwind", "bootstrap",
    "graphql", "rest api", "restful", "soap",
    # Data / ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "opencv", "huggingface",
    "transformers", "bert", "gpt", "llm", "langchain", "rag",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra",
    "elasticsearch", "oracle", "sqlite", "dynamodb", "firestore",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ansible", "jenkins", "gitlab ci", "github actions",
    "ci/cd", "linux", "bash", "shell scripting",
    # Data Engineering
    "spark", "hadoop", "airflow", "kafka", "dbt", "snowflake",
    "bigquery", "redshift", "etl", "data pipeline", "data warehouse",
    # Tools
    "git", "github", "gitlab", "jira", "confluence", "figma",
    "postman", "swagger", "vs code", "intellij", "jupyter",
    # Mobile
    "android", "ios", "react native", "flutter", "xamarin",
}

SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "collaboration",
    "problem solving", "critical thinking", "time management",
    "project management", "agile", "scrum", "kanban",
    "attention to detail", "adaptability", "creativity",
    "analytical", "presentation", "mentoring", "stakeholder management",
    "cross-functional", "strategic thinking", "decision making",
}

CERTIFICATIONS = {
    "aws certified", "azure certified", "google certified", "pmp",
    "cissp", "ceh", "comptia", "ccna", "ccnp", "gcp certified",
    "certified scrum master", "csm", "safe", "itil",
    "tensorflow developer", "pytorch certified",
}

EDUCATION_KEYWORDS = {
    "bachelor", "master", "phd", "doctorate", "mba", "b.tech", "m.tech",
    "b.sc", "m.sc", "b.e", "m.e", "degree", "diploma", "associate",
    "computer science", "information technology", "data science",
    "software engineering", "electrical engineering", "mathematics",
    "statistics", "physics",
}

ALL_SKILLS = TECH_SKILLS | SOFT_SKILLS


# ── Extraction Functions ───────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, remove special chars, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\/\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills_from_taxonomy(text: str) -> Dict[str, Set[str]]:
    """Match text against curated skills taxonomy."""
    norm = normalize(text)
    found_tech = set()
    found_soft = set()
    found_certs = set()

    for skill in TECH_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", norm):
            found_tech.add(skill)

    for skill in SOFT_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", norm):
            found_soft.add(skill)

    for cert in CERTIFICATIONS:
        if re.search(r"\b" + re.escape(cert) + r"\b", norm):
            found_certs.add(cert)

    return {
        "technical": found_tech,
        "soft": found_soft,
        "certifications": found_certs,
    }


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """Extract n-grams from text for multi-word skill detection."""
    words = normalize(text).split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(" ".join(words[i : i + n]))
    return ngrams


def extract_keywords_tfidf(resume_text: str, jd_text: str, top_n: int = 30) -> Dict:
    """
    Use TF-IDF to find important terms from the JD
    that may or may not appear in the resume.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        corpus = [jd_text, resume_text]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=200,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # JD important terms (document 0)
        jd_scores = tfidf_matrix[0].toarray()[0]
        top_indices = np.argsort(jd_scores)[::-1][:top_n]
        jd_keywords = {feature_names[i]: round(float(jd_scores[i]), 4)
                       for i in top_indices if jd_scores[i] > 0}

        # Resume important terms (document 1)
        resume_scores = tfidf_matrix[1].toarray()[0]
        resume_keywords = {feature_names[i]: round(float(resume_scores[i]), 4)
                           for i in np.argsort(resume_scores)[::-1][:top_n]
                           if resume_scores[i] > 0}

        return {"jd": jd_keywords, "resume": resume_keywords}

    except ImportError:
        # Fallback: simple frequency count
        jd_words = Counter(normalize(jd_text).split())
        resume_words = Counter(normalize(resume_text).split())
        stopwords = {"the", "and", "is", "in", "to", "of", "a", "for",
                     "with", "you", "will", "we", "our", "an", "or", "be"}
        jd_kw = {w: c for w, c in jd_words.most_common(top_n) if w not in stopwords}
        res_kw = {w: c for w, c in resume_words.most_common(top_n) if w not in stopwords}
        return {"jd": jd_kw, "resume": res_kw}


def extract_years_of_experience(text: str) -> int:
    """Parse mentioned years of experience from text."""
    patterns = [
        r"(\d+)\+?\s+years?\s+of\s+experience",
        r"(\d+)\+?\s+years?\s+experience",
        r"experience\s+of\s+(\d+)\+?\s+years?",
    ]
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        years.extend([int(m) for m in matches])
    return max(years) if years else 0


def extract_education_level(text: str) -> str:
    """Detect highest education level mentioned."""
    norm = normalize(text)
    if re.search(r"\b(phd|doctorate|doctor of)\b", norm):
        return "PhD"
    elif re.search(r"\b(master|m\.s|m\.sc|m\.tech|mba|m\.e)\b", norm):
        return "Master's"
    elif re.search(r"\b(bachelor|b\.s|b\.sc|b\.tech|b\.e|undergraduate)\b", norm):
        return "Bachelor's"
    elif re.search(r"\b(associate|diploma|certificate)\b", norm):
        return "Associate/Diploma"
    return "Not specified"


def full_keyword_analysis(resume_text: str, jd_text: str) -> Dict:
    """
    Run complete keyword analysis on resume vs JD.
    Returns structured keyword data for scoring.
    """
    resume_skills = extract_skills_from_taxonomy(resume_text)
    jd_skills = extract_skills_from_taxonomy(jd_text)
    tfidf_data = extract_keywords_tfidf(resume_text, jd_text)

    # Keyword overlap
    tech_match = resume_skills["technical"] & jd_skills["technical"]
    tech_missing = jd_skills["technical"] - resume_skills["technical"]

    soft_match = resume_skills["soft"] & jd_skills["soft"]
    soft_missing = jd_skills["soft"] - resume_skills["soft"]

    cert_match = resume_skills["certifications"] & jd_skills["certifications"]
    cert_missing = jd_skills["certifications"] - resume_skills["certifications"]

    # TF-IDF based missing keywords
    jd_kw_set = set(tfidf_data["jd"].keys())
    resume_kw_set = set(tfidf_data["resume"].keys())
    tfidf_missing = jd_kw_set - resume_kw_set

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched": {
            "technical": sorted(tech_match),
            "soft": sorted(soft_match),
            "certifications": sorted(cert_match),
        },
        "missing": {
            "technical": sorted(tech_missing),
            "soft": sorted(soft_missing),
            "certifications": sorted(cert_missing),
        },
        "tfidf": tfidf_data,
        "tfidf_missing_keywords": sorted(tfidf_missing)[:15],
        "resume_experience_years": extract_years_of_experience(resume_text),
        "jd_required_years": extract_years_of_experience(jd_text),
        "resume_education": extract_education_level(resume_text),
        "jd_education": extract_education_level(jd_text),
    }