import os
import anthropic
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@st.cache_data
def explain(sentence, match, score, typ):

    prompt = f"""You are a plagiarism analysis expert.

Student sentence: {sentence}

Matched source: {match}

Similarity score: {score}

Detection type: {typ}

In 2-3 sentences, explain WHY this is flagged as plagiarism. 
Be specific — is it word-for-word, paraphrased structure, same argument, 
or coincidental similarity?"""

    res = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.content[0].text