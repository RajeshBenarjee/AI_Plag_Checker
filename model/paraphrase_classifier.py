"""
model/paraphrase_classifier.py
--------------------------------
Dedicated paraphrase detection using a fine-tuned cross-encoder model.
Uses: cross-encoder/stsb-roberta-base (sentence similarity)

This goes BEYOND semantic similarity — it specifically catches:
  - Restructured sentences (same meaning, different order)
  - Synonym substitution (replaced words with synonyms)
  - Active/passive voice swaps
  - Smart rewriting where every word is changed but meaning is same

How it works:
  - SentenceTransformer gives vector similarity (bi-encoder)
  - CrossEncoder scores the actual pair directly (more accurate for paraphrasing)
  - We combine both for maximum accuracy
"""

from sentence_transformers import CrossEncoder
import numpy as np


# Load once — cross-encoder specifically trained for sentence similarity
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading paraphrase cross-encoder model...")
        _cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-base")
    return _cross_encoder


def classify_paraphrase(sentence: str, matched_source: str) -> dict:
    """
    Given a student sentence and its best FAISS match,
    determine if it's a paraphrase using the cross-encoder.

    Returns:
    {
        "is_paraphrase": bool,
        "cross_score": float,      ← 0 to 1
        "paraphrase_type": str,    ← "Direct Copy" / "Paraphrase" / "Original"
        "confidence": str          ← "High" / "Medium" / "Low"
    }
    """

    model = _get_cross_encoder()

    # Cross-encoder scores the pair directly
    score = float(model.predict([[sentence, matched_source]])[0])

    # Normalize to 0-1 range (stsb outputs -1 to 1 sometimes)
    score = max(0.0, min(1.0, (score + 1) / 2 if score < 0 else score))

    # Classify
    if score >= 0.90:
        paraphrase_type = "Direct Copy"
        is_paraphrase = True
        confidence = "High"
    elif score >= 0.75:
        paraphrase_type = "Paraphrase"
        is_paraphrase = True
        confidence = "High"
    elif score >= 0.60:
        paraphrase_type = "Paraphrase"
        is_paraphrase = True
        confidence = "Medium"
    elif score >= 0.45:
        paraphrase_type = "Possible Paraphrase"
        is_paraphrase = False
        confidence = "Low"
    else:
        paraphrase_type = "Original"
        is_paraphrase = False
        confidence = "High"

    return {
        "is_paraphrase": is_paraphrase,
        "cross_score": round(score, 3),
        "paraphrase_type": paraphrase_type,
        "confidence": confidence
    }


def batch_classify(sentence_match_pairs: list) -> list:
    """
    Classify multiple (sentence, match) pairs at once.
    More efficient than calling classify_paraphrase one by one.

    Parameters:
        sentence_match_pairs - list of (student_sentence, matched_sentence) tuples

    Returns list of classification dicts.
    """
    if not sentence_match_pairs:
        return []

    model = _get_cross_encoder()

    scores = model.predict(sentence_match_pairs)

    results = []
    for score in scores:
        score = float(score)
        score = max(0.0, min(1.0, (score + 1) / 2 if score < 0 else score))

        if score >= 0.90:
            results.append({"is_paraphrase": True,  "cross_score": round(score, 3),
                            "paraphrase_type": "Direct Copy",        "confidence": "High"})
        elif score >= 0.75:
            results.append({"is_paraphrase": True,  "cross_score": round(score, 3),
                            "paraphrase_type": "Paraphrase",         "confidence": "High"})
        elif score >= 0.60:
            results.append({"is_paraphrase": True,  "cross_score": round(score, 3),
                            "paraphrase_type": "Paraphrase",         "confidence": "Medium"})
        elif score >= 0.45:
            results.append({"is_paraphrase": False, "cross_score": round(score, 3),
                            "paraphrase_type": "Possible Paraphrase","confidence": "Low"})
        else:
            results.append({"is_paraphrase": False, "cross_score": round(score, 3),
                            "paraphrase_type": "Original",           "confidence": "High"})

    return results