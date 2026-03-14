import pandas as pd
import random

INPUT_FILE = "data/wikisent2.txt"
OUTPUT_FILE = "data/wiki_sentences.csv"

sentences = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()

        if len(s) < 10:
            continue

        sentences.append(s)

random.shuffle(sentences)

sample = sentences[:500000]

df = pd.DataFrame(sample, columns=["sentence"])

df.drop_duplicates(inplace=True)

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset created:", len(df))