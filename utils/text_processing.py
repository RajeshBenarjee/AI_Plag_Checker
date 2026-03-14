import re
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sentence_tokenize(text):
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]