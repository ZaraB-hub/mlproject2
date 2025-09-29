# src/preprocess.py
import re
import pandas as pd
from pathlib import Path

# simple regexes
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"u/[A-Za-z0-9_-]+|@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#\w+")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def extract_hashtags(t: str):
    return [h.lower() for h in HASHTAG_RE.findall(t or "")]

def run(inp="reddit_tech_posts.csv", out="data/reddit_clean.csv"):
    df = pd.read_csv(inp)

    df["clean_text"] = df["text"].apply(clean_text)
    df["hashtags"]   = df["text"].apply(extract_hashtags)

    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

    Path("data").mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Preprocessed {len(df)} rows → {out}")

if __name__ == "__main__":
    run()
