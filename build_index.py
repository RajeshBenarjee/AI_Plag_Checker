import faiss
import pickle
import numpy as np

from model.embedding_model import EmbeddingModel
from utils.text_processing import preprocess_text, sentence_tokenize

DATA_FILE = "data/wikisent2.txt"

INDEX_PATH = "database/faiss_index.bin"
SENTENCE_PATH = "database/sentences.pkl"


def load_dataset():

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    text = preprocess_text(text)
    sentences = sentence_tokenize(text)

    return sentences


def build_index():

    sentences = load_dataset()

    model = EmbeddingModel()

    embeddings = model.encode(sentences)

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(SENTENCE_PATH, "wb") as f:
        pickle.dump(sentences, f)

    print("FAISS index built successfully")


if __name__ == "__main__":
    build_index()