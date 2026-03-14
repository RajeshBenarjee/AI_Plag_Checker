import requests
from bs4 import BeautifulSoup


def fetch_webpage_text(url):

    try:

        response = requests.get(url, timeout=5)

        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")

        text = " ".join([p.get_text() for p in paragraphs])

        return text

    except:
        return ""