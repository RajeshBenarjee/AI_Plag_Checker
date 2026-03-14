from sentence_transformers import SentenceTransformer
import numpy as np


MODELS = {
    "english": "all-MiniLM-L6-v2",
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"
}


class EmbeddingModel:

    def __init__(self, model_type="english"):
        model_name = MODELS.get(model_type, MODELS["english"])
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings.astype("float32")


# ---------------------------------------------------------
# Global helper so Streamlit can reuse the same model
# ---------------------------------------------------------

_model_instance = None
_current_model_type = None


def embed_sentences(sentences, model_type="english"):

    global _model_instance, _current_model_type

    if _model_instance is None or _current_model_type != model_type:
        _model_instance = EmbeddingModel(model_type=model_type)
        _current_model_type = model_type

    return _model_instance.encode(sentences)