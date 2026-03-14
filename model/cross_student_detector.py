
import os
import faiss
import pickle
import numpy as np

from model.embedding_model import EmbeddingModel
from utils.text_processing import preprocess_text, sentence_tokenize


class CrossStudentDetector:

    def __init__(self):

        self.model = EmbeddingModel()

        self.index_path = "database/student_index.bin"
        self.sentences_path = "database/student_sentences.pkl"

        # Load existing index if present
        if os.path.exists(self.index_path):

            self.index = faiss.read_index(self.index_path)

            with open(self.sentences_path, "rb") as f:
                self.sentences = pickle.load(f)

        else:

            # MiniLM embedding dimension = 384
            self.index = faiss.IndexFlatIP(384)
            self.sentences = []


    # ------------------------------------------------
    # Detect similarity with previous student submissions
    # ------------------------------------------------

    def detect_similarity(self, text):

        text = preprocess_text(text)

        sentences = sentence_tokenize(text)

        if len(sentences) == 0 or len(self.sentences) == 0:
            return []

        embeddings = self.model.encode(sentences).astype("float32")

        faiss.normalize_L2(embeddings)

        D, I = self.index.search(embeddings, 3)

        results = []

        for i, sentence in enumerate(sentences):

            best_score = float(D[i][0])
            best_index = I[i][0]

            if best_index < 0:
                continue

            if best_score >= 0.75:

                results.append({
                    "sentence": sentence,
                    "matched": self.sentences[best_index],
                    "similarity": round(best_score, 3)
                })

        return results


    # ------------------------------------------------
    # Add new submission to database
    # ------------------------------------------------

    def add_submission(self, text):

        text = preprocess_text(text)

        sentences = sentence_tokenize(text)

        if len(sentences) == 0:
            return

        embeddings = self.model.encode(sentences).astype("float32")

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.sentences.extend(sentences)

        # Save updated index
        faiss.write_index(self.index, self.index_path)

        with open(self.sentences_path, "wb") as f:
            pickle.dump(self.sentences, f)
