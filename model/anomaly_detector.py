"""
model/anomaly_detector.py
--------------------------
Detects anomalies in student submission patterns:
  1. Sudden quality jump  → avg sentence length / vocab richness spike
  2. AI-generated content → unusually perfect structure, low variance
  3. Score spike          → plagiarism % suddenly much higher than usual

Uses statistical outlier detection (z-score based).
"""

import numpy as np
import re
from utils.text_processing import sentence_tokenize


# -------------------------------------------------------
# Text feature extraction
# -------------------------------------------------------

def _extract_features(text: str) -> dict:
    sentences = sentence_tokenize(text)

    if not sentences:
        return {}

    words_per_sent = [len(s.split()) for s in sentences]
    all_words = text.split()
    all_chars = list(text)

    vocab_richness = len(set(all_words)) / len(all_words) if all_words else 0
    avg_sent_length = float(np.mean(words_per_sent))
    sent_length_variance = float(np.var(words_per_sent))
    avg_word_length = float(np.mean([len(w) for w in all_words])) if all_words else 0
    punct_density = sum(1 for c in all_chars if c in ".,;:!?") / len(all_chars) if all_chars else 0

    return {
        "avg_sent_length": round(avg_sent_length, 2),
        "sent_length_variance": round(sent_length_variance, 2),
        "vocab_richness": round(vocab_richness, 3),
        "avg_word_length": round(avg_word_length, 2),
        "punct_density": round(punct_density, 4)
    }


# -------------------------------------------------------
# Anomaly detection
# -------------------------------------------------------

def detect_anomalies(current_text: str, history: list) -> dict:
    """
    Compare current submission against student's history.

    Parameters:
        current_text  - raw text of current submission
        history       - list of past submission dicts from submission_tracker

    Returns dict with:
        flags         - list of anomaly flag strings
        verdict       - "Normal" / "Suspicious" / "High Risk"
        details       - per-check results
    """

    flags = []
    details = {}

    current_features = _extract_features(current_text)
    current_score = None  # will be set externally if available

    # Need at least 2 past submissions to compare
    if len(history) < 2:
        return {
            "flags": [],
            "verdict": "Not enough history",
            "details": {"note": "Need at least 2 past submissions to detect anomalies."}
        }

    past_scores = [r["percentage"] for r in history]
    avg_past = np.mean(past_scores)
    std_past = np.std(past_scores) if len(past_scores) > 1 else 0

    # -------------------------------------------------------
    # Check 1: Plagiarism score spike
    # -------------------------------------------------------
    latest_score = history[-1]["percentage"]

    if std_past > 0:
        z = (latest_score - avg_past) / std_past
        details["score_zscore"] = round(float(z), 2)

        if z > 2.0:
            flags.append(f"📈 Plagiarism score spiked to {latest_score}% (avg was {avg_past:.1f}%)")
    else:
        details["score_zscore"] = 0

    # -------------------------------------------------------
    # Check 2: Sudden writing quality jump (vocab richness)
    # -------------------------------------------------------
    # We store features if possible — here we use trust_score as proxy
    trust_scores = [r.get("trust_score", 0) for r in history]
    avg_trust = np.mean(trust_scores)
    latest_trust = trust_scores[-1] if trust_scores else 0

    if avg_trust > 0 and latest_trust > avg_trust * 1.5:
        flags.append(
            f"⚡ Trust score jumped sharply: {latest_trust:.1f} vs avg {avg_trust:.1f}"
        )
        details["trust_spike"] = True
    else:
        details["trust_spike"] = False

    # -------------------------------------------------------
    # Check 3: Low sentence length variance (AI writing pattern)
    # -------------------------------------------------------
    if current_features:
        variance = current_features.get("sent_length_variance", 99)
        vocab = current_features.get("vocab_richness", 0)

        details["sent_length_variance"] = variance
        details["vocab_richness"] = vocab

        # AI-generated text tends to have very uniform sentence lengths
        if variance < 8.0:
            flags.append(
                f"🤖 Very uniform sentence lengths (variance={variance}) — "
                f"possible AI-generated content"
            )

        # Unusually high vocab richness = possible AI
        if vocab > 0.85:
            flags.append(
                f"🤖 Unusually high vocabulary richness ({vocab:.2f}) — "
                f"possible AI-generated content"
            )

    # -------------------------------------------------------
    # Check 4: Submission frequency spike
    # -------------------------------------------------------
    if len(history) >= 3:
        timestamps = [r["timestamp"] for r in history[-3:]]
        # If last 3 submissions all within same day — suspicious
        dates = [t[:10] for t in timestamps]
        if len(set(dates)) == 1:
            flags.append(
                "⏰ 3+ submissions in a single day — unusual submission pattern"
            )

    # -------------------------------------------------------
    # Final verdict
    # -------------------------------------------------------
    if len(flags) >= 3:
        verdict = "🔴 High Risk"
    elif len(flags) >= 1:
        verdict = "🟡 Suspicious"
    else:
        verdict = "🟢 Normal"

    return {
        "flags": flags,
        "verdict": verdict,
        "details": details
    }