import re
import nltk

def ensure_nltk_data():
    resources = ["punkt", "punkt_tab"]
    for resource in resources:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download(resource, quiet=True)

ensure_nltk_data()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def sentence_tokenize(text):
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        ensure_nltk_data()
        sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]