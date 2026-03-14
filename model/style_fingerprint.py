import string
import numpy as np
import nltk

from utils.text_processing import sentence_tokenize

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


def _compute_features(sentences):
    """Compute stylometric features for a list of sentences."""

    if not sentences:
        return {
            "avg_sentence_length": 0,
            "vocab_richness": 0,
            "avg_word_length": 0,
            "punctuation_density": 0,
            "stopword_ratio": 0
        }

    all_text = " ".join(sentences)
    all_words = all_text.split()
    all_chars = list(all_text)

    # avg words per sentence
    avg_sentence_length = np.mean([len(s.split()) for s in sentences])

    # type-token ratio
    vocab_richness = len(set(all_words)) / len(all_words) if all_words else 0

    # avg character length of words
    avg_word_length = np.mean([len(w) for w in all_words]) if all_words else 0

    # punctuation density
    punct_count = sum(1 for c in all_chars if c in string.punctuation)
    punctuation_density = punct_count / len(all_chars) if all_chars else 0

    # stopword ratio
    stopword_count = sum(1 for w in all_words if w.lower() in STOPWORDS)
    stopword_ratio = stopword_count / len(all_words) if all_words else 0

    return {
        "avg_sentence_length": round(float(avg_sentence_length), 3),
        "vocab_richness": round(float(vocab_richness), 3),
        "avg_word_length": round(float(avg_word_length), 3),
        "punctuation_density": round(float(punctuation_density), 3),
        "stopword_ratio": round(float(stopword_ratio), 3)
    }


def _feature_vector(features):
    return np.array([
        features["avg_sentence_length"],
        features["vocab_richness"],
        features["avg_word_length"],
        features["punctuation_density"],
        features["stopword_ratio"]
    ])


def _euclidean(v1, v2):
    return float(np.linalg.norm(v1 - v2))


class StyleFingerprint:

    THRESHOLD = 0.35

    def analyze(self, text):

        sentences = sentence_tokenize(text)

        if len(sentences) < 5:
            return {
                "verdict": "Not enough text to analyze",
                "sections": {},
                "distances": {}
            }

        n = len(sentences)
        intro_end = max(1, int(n * 0.20))
        body_end = max(intro_end + 1, int(n * 0.80))

        intro_sents = sentences[:intro_end]
        body_sents = sentences[intro_end:body_end]
        conclusion_sents = sentences[body_end:]

        intro_f = _compute_features(intro_sents)
        body_f = _compute_features(body_sents)
        conclusion_f = _compute_features(conclusion_sents)

        v_intro = _feature_vector(intro_f)
        v_body = _feature_vector(body_f)
        v_conclusion = _feature_vector(conclusion_f)

        d_ib = round(_euclidean(v_intro, v_body), 4)
        d_bc = round(_euclidean(v_body, v_conclusion), 4)
        d_ic = round(_euclidean(v_intro, v_conclusion), 4)

        verdict = (
            "Possible Multiple Authors"
            if max(d_ib, d_bc, d_ic) > self.THRESHOLD
            else "Consistent Writing Style"
        )

        return {
            "verdict": verdict,
            "sections": {
                "intro": intro_f,
                "body": body_f,
                "conclusion": conclusion_f
            },
            "distances": {
                "intro_vs_body": d_ib,
                "body_vs_conclusion": d_bc,
                "intro_vs_conclusion": d_ic
            }
        }