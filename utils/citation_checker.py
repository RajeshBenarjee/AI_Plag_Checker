import re


# Patterns that indicate a sentence is properly cited
CITATION_PATTERNS = [
    r'"[^"]+"',                          # quoted text "..."
    r'\([A-Z][a-z]+,\s*\d{4}\)',         # APA: (Author, 2020)
    r'\[\d+\]',                          # numbered ref: [1], [23]
    r'\bet\s+al\.',                      # et al.
    r'\bAccording\s+to\b',              # According to
    r'\bAs\s+stated\s+by\b',            # As stated by
    r'\bAs\s+cited\s+in\b',             # As cited in
    r'\bAs\s+noted\s+by\b',             # As noted by
    r'\bquoted\s+from\b',               # quoted from
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in CITATION_PATTERNS]


def is_cited(sentence):
    for pattern in COMPILED:
        if pattern.search(sentence):
            return True
    return False


def filter_cited_sentences(sentences):
    """
    Returns (cited, uncited) lists.
    cited   → properly cited, excluded from plagiarism check
    uncited → needs to be checked
    """
    cited = []
    uncited = []

    for s in sentences:
        if is_cited(s):
            cited.append(s)
        else:
            uncited.append(s)

    return cited, uncited