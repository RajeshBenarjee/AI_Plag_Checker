"""
build_index_english.py
-----------------------
Builds the English FAISS index by merging two sources:
  1. data/dataset.txt       — custom domain sentences (AI/CS)
  2. data/wikisent2.txt     — Kaggle Wikipedia sentences (sampled)

Output:
  database/faiss_index_english.bin
  database/sentences_english.pkl

Run from project root:
  python build_index_english.py
"""

import faiss
import pickle
import random
import numpy as np

from model.embedding_model import EmbeddingModel
from utils.text_processing import preprocess_text, sentence_tokenize

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATASET_TXT   = "data/dataset.txt"       # your custom sentences
WIKI_TXT      = "data/wikisent2.txt"     # Kaggle Wikipedia sentences
WIKI_SAMPLE   = 500_000                  # how many wiki sentences to sample

INDEX_PATH    = "database/faiss_index_english.bin"
SENTENCE_PATH = "database/sentences_english.pkl"


# -------------------------------------------------------
# Step 1 — Load custom dataset.txt
# -------------------------------------------------------
def load_custom_dataset():
    print("Loading dataset.txt...")

    with open(DATASET_TXT, "r", encoding="utf-8") as f:
        text = f.read()

    text = preprocess_text(text)
    sentences = sentence_tokenize(text)

    print(f"  dataset.txt → {len(sentences)} sentences")
    return sentences


# -------------------------------------------------------
# Step 2 — Load + sample wikisent2.txt
# -------------------------------------------------------
def load_wiki_dataset():
    print("Loading wikisent2.txt (this may take a moment)...")

    sentences = []

    with open(WIKI_TXT, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if len(s) >= 10:
                sentences.append(s)

    print(f"  wikisent2.txt → {len(sentences)} total sentences")

    # Sample
    random.shuffle(sentences)
    sample = sentences[:WIKI_SAMPLE]

    print(f"  Sampled → {len(sample)} sentences")
    return sample


# -------------------------------------------------------
# Step 3 — Merge + deduplicate
# -------------------------------------------------------
def merge(custom, wiki):
    print("Merging datasets...")

    combined = custom + wiki

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in combined:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"  Total after merge + dedup → {len(unique)} sentences")
    return unique


# -------------------------------------------------------
# Step 4 — Build FAISS index
# -------------------------------------------------------
def build_index(sentences):
    print("Encoding sentences (English model)...")

    model = EmbeddingModel(model_type="english")

    # Encode in batches to avoid memory issues
    batch_size = 1024
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        emb = model.encode(batch)
        all_embeddings.append(emb)

        if i % 50000 == 0:
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

    print(f"\n✅ English index built successfully!")
    print(f"   Sentences : {len(sentences)}")
    print(f"   Index     : {INDEX_PATH}")
    print(f"   Sentences : {SENTENCE_PATH}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    custom   = load_custom_dataset()
    wiki     = load_wiki_dataset()
    combined = merge(custom, wiki)
    build_index(combined)