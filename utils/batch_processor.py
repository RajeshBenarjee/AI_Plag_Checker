import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.pdf_parser import extract_text_from_pdf
from utils.text_processing import sentence_tokenize, preprocess_text
from model.embedding_model import EmbeddingModel


class BatchProcessor:

    def __init__(self, engine):
        self.engine = engine
        self.embed_model = EmbeddingModel()

    def process_batch(self, uploaded_files):
        """
        Process multiple uploaded PDF files.

        Returns:
            summaries   - list of per-file result dicts
            sim_matrix  - dict {filenameA: {filenameB: score}}
        """

        summaries = []
        file_sentences = {}   # filename → list of sentences
        file_embeddings = {}  # filename → numpy embeddings

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            try:
                text = extract_text_from_pdf(uploaded_file)
            except Exception:
                text = uploaded_file.read().decode("utf-8", errors="ignore")

            results, percentage, total, plagiarized, cited = self.engine.detect(text)

            summaries.append({
                "filename": filename,
                "results": results,
                "percentage": round(percentage, 2),
                "total": total,
                "plagiarized": plagiarized,
                "cited": len(cited)
            })

            # Store sentences + embeddings for cross-file similarity
            clean_text = preprocess_text(text)
            sents = sentence_tokenize(clean_text)

            if sents:
                embs = self.embed_model.encode(sents).astype("float32")
                file_sentences[filename] = sents
                file_embeddings[filename] = embs

        # Build cross-file similarity matrix
        sim_matrix = {}
        filenames = list(file_embeddings.keys())

        for i, fnA in enumerate(filenames):

            sim_matrix[fnA] = {}

            for j, fnB in enumerate(filenames):

                if fnA == fnB:
                    sim_matrix[fnA][fnB] = 1.0
                    continue

                embA = file_embeddings[fnA]
                embB = file_embeddings[fnB]

                # Average of max similarity per sentence in A
                sims = cosine_similarity(embA, embB)
                avg_max = float(np.mean(sims.max(axis=1)))
                sim_matrix[fnA][fnB] = round(avg_max, 3)

        return summaries, sim_matrix