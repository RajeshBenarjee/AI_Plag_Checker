from serpapi.google_search import GoogleSearch
from utils.web_scraper import fetch_webpage_text
from utils.text_processing import sentence_tokenize
from model.embedding_model import EmbeddingModel

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class InternetDetector:

    def __init__(self, api_key):

        self.api_key = api_key
        self.model = EmbeddingModel()


    # --------------------------------------------------
    # Search Google using SerpAPI
    # --------------------------------------------------
    def search_web(self, query):

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": 5
        }

        search = GoogleSearch(params)

        results = search.get_dict()

        links = []

        if "organic_results" in results:

            for r in results["organic_results"][:5]:

                if "link" in r:
                    links.append(r["link"])

        return links


    # --------------------------------------------------
    # Detect plagiarism from web sources
    # --------------------------------------------------
    def detect(self, sentence):

        links = self.search_web(sentence)[:120]

        best_score = 0.0
        best_match = ""
        best_source = ""

        # encode query once
        query_embedding = self.model.encode([sentence]).astype("float32")

        for link in links:

            page_text = fetch_webpage_text(link)

            if not page_text:
                continue

            sentences = sentence_tokenize(page_text)

            if len(sentences) == 0:
                continue

            page_embeddings = self.model.encode(sentences).astype("float32")

            sims = cosine_similarity(query_embedding, page_embeddings)[0]

            idx = int(np.argmax(sims))
            score = float(sims[idx])

            # clamp similarity
            score = min(max(score, 0.0), 1.0)

            if score > best_score:

                best_score = score
                best_match = sentences[idx]
                best_source = link

        return best_score, best_match, best_source