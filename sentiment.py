# src/sentiment.py
import pandas as pd
from transformers import pipeline
from pathlib import Path

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def normalize_label(lbl: str) -> str:
    mapping = {
        "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
        "NEGATIVE": "negative", "NEUTRAL": "neutral", "POSITIVE": "positive",
        "negative": "negative", "neutral": "neutral", "positive": "positive"
    }
    return mapping.get(lbl, lbl).lower()

def run(inp="data/reddit_clean.csv", out="data/reddit_with_sentiment.csv", batch=32):
    df = pd.read_csv(inp)

    clf = pipeline("sentiment-analysis", model=MODEL, truncation=True, max_length=512)

    texts = df["clean_text"].astype(str).tolist()
    preds = []

    print(f"Sentiment on {len(texts)} posts...")
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        preds.extend(clf(chunk))

    df["sent_label"] = [normalize_label(p["label"]) for p in preds]
    df["sent_score"] = [float(p.get("score", 0.0)) for p in preds]

    Path("data").mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Sentiment added → {out}")

if __name__ == "__main__":
    run()
