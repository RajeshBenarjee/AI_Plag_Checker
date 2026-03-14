import os
import faiss
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.embedding_model import EmbeddingModel
from utils.text_processing import preprocess_text, sentence_tokenize
from utils.citation_checker import filter_cited_sentences


# -------------------------------------------------------
# Index paths per model type
# -------------------------------------------------------
INDEX_PATHS = {
    "english": {
        "index":     "database/faiss_index_english.bin",
        "sentences": "database/sentences_english.pkl"
    },
    "multilingual": {
        "index":     "database/faiss_index_multilingual.bin",
        "sentences": "database/sentences_multilingual.pkl"
    }
}

FALLBACK = {
    "index":     "database/faiss_index.bin",
    "sentences": "database/sentences.pkl"
}


class PlagiarismEngine:

    def __init__(self, model_type="english"):

        self.model_type = model_type
        self.model = EmbeddingModel(model_type=model_type)

        paths = INDEX_PATHS.get(model_type, INDEX_PATHS["english"])
        index_path     = paths["index"]
        sentences_path = paths["sentences"]

        if not os.path.exists(index_path):
            print(f"⚠️  Index not found: {index_path}")
            print(f"   Falling back to: {FALLBACK['index']}")
            print(f"   Run build_index_english.py or build_index_multilingual.py")
            index_path     = FALLBACK["index"]
            sentences_path = FALLBACK["sentences"]

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                "No FAISS index found. Run build_index_english.py first."
            )

        self.index = faiss.read_index(index_path)

        with open(sentences_path, "rb") as f:
            self.source_sentences = pickle.load(f)

        print(f"✅ FAISS loaded [{model_type}]: {len(self.source_sentences)} sentences")


    # ----------------------------------------
    # TF-IDF similarity
    # ----------------------------------------

    def tfidf_similarity(self, s1, s2):

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([s1, s2])
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        return float(sim)


    # ----------------------------------------
    # Classification
    # ----------------------------------------

    def classify_similarity(self, score):

        if score >= 0.90:
            return "High Plagiarism"
        elif score >= 0.80:
            return "Strong Similarity"
        elif score >= 0.70:
            return "Possible Paraphrasing"
        else:
            return "Clean"


    # ----------------------------------------
    # Detection
    # ----------------------------------------

    def detect(self, text):

        text = preprocess_text(text)
        all_sentences = sentence_tokenize(text)

        if len(all_sentences) == 0:
            return [], 0, 0, 0, []

        # --- Citation filter ---
        cited_sentences, sentences = filter_cited_sentences(all_sentences)

        if len(sentences) == 0:
            return [], 0, 0, 0, cited_sentences

        embeddings = self.model.encode(sentences)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        D, I = self.index.search(embeddings, 5)

        results = []
        plagiarized = 0

        for idx, sentence in enumerate(sentences):

            best_score = 0
            best_match = "No strong dataset match found"
            best_type = "None"

            for j in range(5):

                source_idx = I[idx][j]

                if source_idx < 0:
                    continue

                source_sentence = self.source_sentences[source_idx]
                semantic_score = float(D[idx][j])
                exact_score = self.tfidf_similarity(sentence, source_sentence)
                score = min(max(semantic_score, exact_score), 1.0)

                if score > best_score:
                    best_score = score
                    best_match = source_sentence

                    if semantic_score >= exact_score:
                        best_type = "Semantic"
                    else:
                        best_type = "Exact"

            classification = self.classify_similarity(best_score)

            if best_score >= 0.70:
                plagiarized += 1

            results.append({
                "sentence": sentence,
                "match": best_match,
                "score": best_score,
                "type": best_type,
                "classification": classification
            })

        total = len(sentences)
        plagiarism_percentage = (plagiarized / total) * 100 if total else 0

        return results, plagiarism_percentage, total, plagiarized, cited_sentences