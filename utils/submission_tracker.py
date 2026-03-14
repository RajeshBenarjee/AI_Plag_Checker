"""
utils/submission_tracker.py
-----------------------------
Stores and retrieves submission history per student.
Used by:
  - Teacher dashboard (trends, flagged students)
  - Student timeline (per-student history)
  - Anomaly detection

Storage: database/submissions.json
"""

import os
import json
from datetime import datetime


SUBMISSIONS_FILE = "database/submissions.json"


def _load() -> dict:
    if not os.path.exists(SUBMISSIONS_FILE):
        return {}
    with open(SUBMISSIONS_FILE, "r") as f:
        return json.load(f)


def _save(data: dict):
    os.makedirs("database", exist_ok=True)
    with open(SUBMISSIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# -------------------------------------------------------
# Save a submission
# -------------------------------------------------------

def save_submission(username: str, filename: str, percentage: float,
                    total: int, plagiarized: int, trust_verdict: str,
                    trust_score: float):
    """Save a single submission record for a student."""

    data = _load()

    if username not in data:
        data[username] = []

    data[username].append({
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "percentage": round(percentage, 2),
        "total_sentences": total,
        "plagiarized_sentences": plagiarized,
        "trust_verdict": trust_verdict,
        "trust_score": round(trust_score, 2)
    })

    _save(data)


# -------------------------------------------------------
# Get a student's submission history
# -------------------------------------------------------

def get_student_history(username: str) -> list:
    """Return all submissions for a given student, sorted by time."""
    data = _load()
    return data.get(username, [])


# -------------------------------------------------------
# Teacher dashboard data
# -------------------------------------------------------

def get_all_submissions() -> dict:
    """Return all submissions for all students."""
    return _load()


def get_dashboard_stats() -> dict:
    """
    Returns aggregated stats for the teacher dashboard:
      - total_submissions
      - avg_plagiarism_score
      - top_flagged (sorted by avg score desc)
      - recent_submissions (last 10 across all students)
    """
    data = _load()

    total_submissions = 0
    all_scores = []
    student_stats = {}
    all_records = []

    for username, records in data.items():
        scores = [r["percentage"] for r in records]
        avg = round(sum(scores) / len(scores), 2) if scores else 0

        student_stats[username] = {
            "username": username,
            "submission_count": len(records),
            "avg_plagiarism": avg,
            "last_submission": records[-1]["timestamp"] if records else "",
            "highest_score": max(scores) if scores else 0
        }

        total_submissions += len(records)
        all_scores.extend(scores)

        for r in records:
            all_records.append({**r, "username": username})

    avg_plagiarism = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0

    top_flagged = sorted(
        student_stats.values(),
        key=lambda x: x["avg_plagiarism"],
        reverse=True
    )[:10]

    recent = sorted(all_records, key=lambda x: x["timestamp"], reverse=True)[:10]

    return {
        "total_submissions": total_submissions,
        "avg_plagiarism_score": avg_plagiarism,
        "top_flagged": top_flagged,
        "recent_submissions": recent,
        "student_stats": student_stats
    }   