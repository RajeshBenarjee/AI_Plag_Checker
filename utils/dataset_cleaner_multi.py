"""
dataset_cleaner_multi.py
-------------------------
Cleans and samples the multilingual dataset from Kaggle.

Recommended Kaggle dataset:
  IndicNLP Corpus — covers Telugu, Hindi, Tamil, Kannada + more
  https://www.kaggle.com/datasets/surajms02/indic-nlp-corpus

  OR CC-100 Multilingual Corpus
  https://www.kaggle.com/datasets/imvkashyap/cc-100-multilingual-corpus

Instructions:
  1. Download the dataset from Kaggle
  2. Place the raw .txt files inside data/multilingual/
     Example structure:
       data/multilingual/hindi.txt
       data/multilingual/telugu.txt
       data/multilingual/tamil.txt
       data/multilingual/english.txt   (optional — wiki already covers this)
  3. Run: python utils/dataset_cleaner_multi.py

Output:
  data/multi_sentences.csv
"""

import os
import random
import pandas as pd

INPUT_DIR   = "data/multilingual"       # folder with your language .txt files
OUTPUT_FILE = "data/multi_sentences.csv"
SAMPLE_PER_FILE = 100_000               # sentences per language file


def clean_sentence(s):
    s = s.strip()
    # Skip very short or very long sentences
    if len(s) < 10 or len(s) > 1000:
        return None
    return s


def load_language_file(filepath):
    sentences = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = clean_sentence(line)
            if cleaned:
                sentences.append(cleaned)

    return sentences


def main():

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Directory not found: {INPUT_DIR}")
        print("   Please create data/multilingual/ and add your language .txt files.")
        return

    all_sentences = []
    lang_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]

    if not lang_files:
        print("❌ No .txt files found in data/multilingual/")
        return

    for fname in lang_files:
        fpath = os.path.join(INPUT_DIR, fname)
        lang = fname.replace(".txt", "")

        print(f"Loading {lang}...")
        sents = load_language_file(fpath)

        # Sample per language
        random.shuffle(sents)
        sents = sents[:SAMPLE_PER_FILE]

        print(f"  → {len(sents)} sentences from {lang}")
        all_sentences.extend(sents)

    # Deduplicate
    all_sentences = list(set(all_sentences))
    random.shuffle(all_sentences)

    df = pd.DataFrame(all_sentences, columns=["sentence"])
    df.drop_duplicates(inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Multilingual dataset saved: {OUTPUT_FILE}")
    print(f"   Total sentences: {len(df)}")


if __name__ == "__main__":
    main()