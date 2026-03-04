"""
ats_scorer.py
Core scoring engine. Computes a weighted ATS score from:
  - Semantic similarity     (30%)
  - Technical keyword match (30%)
  - Soft skill match        (10%)
  - Certification match     (10%)
  - Experience years match  (10%)
  - Education level match   (10%)

Returns a full report dict ready for display.
"""

from typing import Dict, Any
import re


# ── Scoring Weights ────────────────────────────────────────────────────────────

WEIGHTS = {
    "semantic_similarity": 0.30,
    "technical_keywords":  0.30,
    "soft_skills":         0.10,
    "certifications":      0.10,
    "experience":          0.10,
    "education":           0.10,
}

EDUCATION_LEVELS = {
    "Not specified": 0,
    "Associate/Diploma": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "PhD": 4,
}


def safe_ratio(matched: int, total: int) -> float:
    """Return match ratio, 1.0 if JD requires nothing."""
    if total == 0:
        return 1.0
    return min(matched / total, 1.0)


def score_technical_keywords(kw_analysis: Dict) -> Dict:
    matched = len(kw_analysis["matched"]["technical"])
    total_jd = len(kw_analysis["jd_skills"]["technical"])
    ratio = safe_ratio(matched, total_jd)
    return {
        "score": ratio,
        "matched": matched,
        "total_required": total_jd,
        "matched_keywords": kw_analysis["matched"]["technical"],
        "missing_keywords": kw_analysis["missing"]["technical"],
    }


def score_soft_skills(kw_analysis: Dict) -> Dict:
    matched = len(kw_analysis["matched"]["soft"])
    total_jd = len(kw_analysis["jd_skills"]["soft"])
    ratio = safe_ratio(matched, total_jd)
    return {
        "score": ratio,
        "matched": matched,
        "total_required": total_jd,
        "matched_keywords": kw_analysis["matched"]["soft"],
        "missing_keywords": kw_analysis["missing"]["soft"],
    }


def score_certifications(kw_analysis: Dict) -> Dict:
    matched = len(kw_analysis["matched"]["certifications"])
    total_jd = len(kw_analysis["jd_skills"]["certifications"])
    ratio = safe_ratio(matched, total_jd)
    return {
        "score": ratio,
        "matched": matched,
        "total_required": total_jd,
        "matched_keywords": kw_analysis["matched"]["certifications"],
        "missing_keywords": kw_analysis["missing"]["certifications"],
    }


def score_experience(kw_analysis: Dict) -> Dict:
    """Score based on years of experience match."""
    resume_years = kw_analysis["resume_experience_years"]
    jd_years = kw_analysis["jd_required_years"]

    if jd_years == 0:
        ratio = 1.0
        note = "No specific years required in JD."
    elif resume_years == 0:
        ratio = 0.5  # Partial credit if years not mentioned
        note = f"JD requires {jd_years}+ years. Not explicitly mentioned in resume."
    elif resume_years >= jd_years:
        ratio = 1.0
        note = f"✅ Resume shows {resume_years} years. JD requires {jd_years} years."
    else:
        ratio = resume_years / jd_years
        note = f"⚠️ Resume shows {resume_years} years. JD requires {jd_years} years."

    return {
        "score": ratio,
        "resume_years": resume_years,
        "jd_required_years": jd_years,
        "note": note,
    }


def score_education(kw_analysis: Dict) -> Dict:
    """Score based on education level match."""
    resume_edu = kw_analysis["resume_education"]
    jd_edu = kw_analysis["jd_education"]

    resume_level = EDUCATION_LEVELS.get(resume_edu, 0)
    jd_level = EDUCATION_LEVELS.get(jd_edu, 0)

    if jd_level == 0:
        ratio = 1.0
        note = "No specific education requirement in JD."
    elif resume_level >= jd_level:
        ratio = 1.0
        note = f"✅ {resume_edu} meets or exceeds the requirement ({jd_edu})."
    else:
        ratio = 0.5
        note = f"⚠️ Resume shows {resume_edu}. JD may require {jd_edu}."

    return {
        "score": ratio,
        "resume_education": resume_edu,
        "jd_education": jd_edu,
        "note": note,
    }


def compute_final_score(component_scores: Dict) -> float:
    """Compute weighted final score (0–100)."""
    total = 0.0
    total += component_scores["semantic_similarity"]["overall_score"] * WEIGHTS["semantic_similarity"]
    total += component_scores["technical_keywords"]["score"] * WEIGHTS["technical_keywords"]
    total += component_scores["soft_skills"]["score"] * WEIGHTS["soft_skills"]
    total += component_scores["certifications"]["score"] * WEIGHTS["certifications"]
    total += component_scores["experience"]["score"] * WEIGHTS["experience"]
    total += component_scores["education"]["score"] * WEIGHTS["education"]
    return round(total * 100, 2)


def get_grade(score: float) -> tuple:
    """Return letter grade and description."""
    if score >= 85:
        return "A", "Excellent Match 🟢", "#22c55e"
    elif score >= 70:
        return "B", "Good Match 🟡", "#eab308"
    elif score >= 55:
        return "C", "Fair Match 🟠", "#f97316"
    elif score >= 40:
        return "D", "Weak Match 🔴", "#ef4444"
    else:
        return "F", "Poor Match ⛔", "#dc2626"


def generate_suggestions(component_scores: Dict, final_score: float) -> list:
    """Generate actionable improvement suggestions."""
    suggestions = []

    # Technical keywords
    missing_tech = component_scores["technical_keywords"]["missing_keywords"]
    if missing_tech:
        top_missing = ", ".join(list(missing_tech)[:5])
        suggestions.append(
            f"🔧 **Add missing technical skills** to your resume: `{top_missing}` "
            f"(and {max(0, len(missing_tech)-5)} more). Add them to your Skills section if you have them."
        )

    # Soft skills
    missing_soft = component_scores["soft_skills"]["missing_keywords"]
    if missing_soft:
        top_soft = ", ".join(list(missing_soft)[:3])
        suggestions.append(
            f"💬 **Incorporate soft skills** into your bullet points: {top_soft}. "
            "Use them naturally in your experience descriptions."
        )

    # Certifications
    missing_certs = component_scores["certifications"]["missing_keywords"]
    if missing_certs:
        suggestions.append(
            f"📜 **Certifications gap**: JD mentions `{', '.join(list(missing_certs))}`. "
            "Consider pursuing these or highlight equivalent skills."
        )

    # Experience
    exp = component_scores["experience"]
    if exp["score"] < 1.0 and exp["jd_required_years"] > 0:
        suggestions.append(
            f"📅 **Experience gap**: JD requires {exp['jd_required_years']} years. "
            "Emphasize impactful achievements to compensate for fewer years."
        )

    # Semantic similarity
    sem = component_scores["semantic_similarity"]
    if sem["overall_score"] < 0.6:
        suggestions.append(
            "📝 **Rewrite your summary/objective** to mirror the language and tone of the job description. "
            "Use similar phrasing for key responsibilities."
        )

    # General
    if final_score < 70:
        suggestions.append(
            "✏️ **Tailor this resume specifically** for this job. Avoid using a generic resume — "
            "customize bullet points to match the JD's language and priorities."
        )

    if not suggestions:
        suggestions.append(
            "🌟 Your resume is a strong match! Focus on quantifying achievements "
            "(e.g., 'increased revenue by 30%') to make it even more compelling."
        )

    return suggestions


def full_ats_score(
    resume_text: str,
    jd_text: str,
    kw_analysis: Dict,
    similarity_result: Dict,
) -> Dict[str, Any]:
    """
    Compute the complete ATS score report.
    
    Args:
        resume_text: Raw resume text
        jd_text: Raw job description text
        kw_analysis: Output from keyword_extractor.full_keyword_analysis()
        similarity_result: Output from similarity.compute_similarity()
    
    Returns:
        Full report dict
    """
    component_scores = {
        "semantic_similarity": similarity_result,
        "technical_keywords": score_technical_keywords(kw_analysis),
        "soft_skills": score_soft_skills(kw_analysis),
        "certifications": score_certifications(kw_analysis),
        "experience": score_experience(kw_analysis),
        "education": score_education(kw_analysis),
    }

    final_score = compute_final_score(component_scores)
    grade, label, color = get_grade(final_score)
    suggestions = generate_suggestions(component_scores, final_score)

    return {
        "final_score": final_score,
        "grade": grade,
        "label": label,
        "color": color,
        "weights": WEIGHTS,
        "components": component_scores,
        "suggestions": suggestions,
        "tfidf_missing": kw_analysis.get("tfidf_missing_keywords", []),
        "keyword_analysis": kw_analysis,
    }