"""
build_index_multilingual.py
-----------------------------
Builds the Multilingual FAISS index from:
  data/multi_sentences.csv   — cleaned multilingual sentences
                               (output of utils/dataset_cleaner_multi.py)

Uses: paraphrase-multilingual-MiniLM-L12-v2
Supports: Hindi, Telugu, Tamil, Kannada, English + 50 more languages

Output:
  database/faiss_index_multilingual.bin
  database/sentences_multilingual.pkl

Run from project root:
  python build_index_multilingual.py
"""

import faiss
import pickle
import numpy as np
import pandas as pd

from model.embedding_model import EmbeddingModel

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
INPUT_CSV     = "data/multi_sentences.csv"
INDEX_PATH    = "database/faiss_index_multilingual.bin"
SENTENCE_PATH = "database/sentences_multilingual.pkl"


# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------
def load_sentences():
    print("Loading multilingual sentences...")

    df = pd.read_csv(INPUT_CSV)
    sentences = df["sentence"].dropna().tolist()
    sentences = [str(s).strip() for s in sentences if len(str(s).strip()) >= 10]

    print(f"  Loaded → {len(sentences)} sentences")
    return sentences


# -------------------------------------------------------
# Build FAISS index
# -------------------------------------------------------
def build_index(sentences):
    print("Encoding sentences (multilingual model)...")
    print("  Note: first run downloads ~420MB model — please wait...")

    model = EmbeddingModel(model_type="multilingual")

    batch_size = 512   # smaller batch for larger multilingual model
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        emb = model.encode(batch)
        all_embeddings.append(emb)

        if i % 20000 == 0:
            print(f"  Encoded {i}/{len(sentences)}...")

    embeddings = np.vstack(all_embeddings).astype("float32")

    print("Normalizing + building FAISS index...")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(SENTENCE_PATH, "wb") as f:
        pickle.dump(sentences, f)

    print(f"\n✅ Multilingual index built successfully!")
    print(f"   Sentences : {len(sentences)}")
    print(f"   Index     : {INDEX_PATH}")
    print(f"   Sentences : {SENTENCE_PATH}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    sentences = load_sentences()
    build_index(sentences)