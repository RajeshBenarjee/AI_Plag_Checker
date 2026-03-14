def compute_trust_score(percentage, results, internet_matches, student_matches):
    """
    Compute a composite trust score from multiple signals.

    Parameters:
        percentage      - raw plagiarism % from engine
        results         - list of sentence result dicts
        internet_matches- list of internet match dicts (with web_source set)
        student_matches - list of cross-student match dicts
    """

    base_score = percentage

    internet_boost = len(internet_matches) * 5
    student_boost = len(student_matches) * 8

    high_confidence_count = len([r for r in results if r["score"] >= 0.90])
    confidence_boost = high_confidence_count * 3

    final_score = min(base_score + internet_boost + student_boost + confidence_boost, 100)
    final_score = round(final_score, 1)

    if final_score >= 80:
        verdict = "CONFIRMED PLAGIARISM"
        color = "#c0392b"
        emoji = "🔴"
    elif final_score >= 60:
        verdict = "LIKELY COPIED"
        color = "#e67e22"
        emoji = "🟠"
    elif final_score >= 35:
        verdict = "SUSPICIOUS"
        color = "#f1c40f"
        emoji = "🟡"
    else:
        verdict = "ORIGINAL"
        color = "#27ae60"
        emoji = "🟢"

    return {
        "score": final_score,
        "verdict": verdict,
        "color": color,
        "emoji": emoji,
        "breakdown": {
            "base": round(base_score, 1),
            "internet_boost": internet_boost,
            "student_boost": student_boost,
            "confidence_boost": confidence_boost
        }
    }