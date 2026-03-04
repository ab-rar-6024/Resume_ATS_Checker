"""
similarity.py
Computes semantic similarity between resume and job description
using Sentence Transformers (SBERT) with cosine similarity.

Falls back to TF-IDF cosine similarity if transformers are not installed.
"""

from typing import List, Tuple
import re


def get_sentences(text: str) -> List[str]:
    """Split text into sentences for fine-grained analysis."""
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def cosine_similarity_manual(vec_a, vec_b) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def compute_similarity_sbert(resume_text: str, jd_text: str) -> dict:
    """
    Use SentenceTransformers to compute semantic similarity.
    Returns overall score and section-level breakdown.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        import torch

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Overall document similarity
        resume_emb = model.encode(resume_text, convert_to_tensor=True)
        jd_emb = model.encode(jd_text, convert_to_tensor=True)
        overall_score = float(util.cos_sim(resume_emb, jd_emb)[0][0])

        # Sentence-level: find best matching resume sentences for each JD sentence
        jd_sentences = get_sentences(jd_text)
        resume_sentences = get_sentences(resume_text)

        if jd_sentences and resume_sentences:
            jd_embs = model.encode(jd_sentences, convert_to_tensor=True)
            res_embs = model.encode(resume_sentences, convert_to_tensor=True)
            cosine_scores = util.cos_sim(jd_embs, res_embs)

            # Top matches
            top_matches = []
            for i, jd_sent in enumerate(jd_sentences[:10]):  # Check top 10 JD sentences
                best_score = float(cosine_scores[i].max())
                best_idx = int(cosine_scores[i].argmax())
                top_matches.append({
                    "jd_sentence": jd_sent[:120],
                    "best_resume_match": resume_sentences[best_idx][:120],
                    "score": round(best_score, 3),
                })
            top_matches.sort(key=lambda x: x["score"], reverse=True)
        else:
            top_matches = []

        return {
            "method": "SBERT (all-MiniLM-L6-v2)",
            "overall_score": round(overall_score, 4),
            "overall_percent": round(overall_score * 100, 2),
            "top_sentence_matches": top_matches[:5],
        }

    except ImportError:
        return compute_similarity_tfidf(resume_text, jd_text)


def compute_similarity_tfidf(resume_text: str, jd_text: str) -> dict:
    """
    Fallback: TF-IDF cosine similarity.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([jd_text, resume_text])
        score = float(cosine_similarity(matrix[0], matrix[1])[0][0])

        return {
            "method": "TF-IDF Cosine Similarity (fallback)",
            "overall_score": round(score, 4),
            "overall_percent": round(score * 100, 2),
            "top_sentence_matches": [],
        }
    except Exception as e:
        return {
            "method": "Error",
            "overall_score": 0.0,
            "overall_percent": 0.0,
            "top_sentence_matches": [],
            "error": str(e),
        }


def compute_similarity(resume_text: str, jd_text: str) -> dict:
    """Main entry point — tries SBERT first, falls back to TF-IDF."""
    return compute_similarity_sbert(resume_text, jd_text)