# src/aggregate.py
import pandas as pd
from pathlib import Path

def run(
    docs_topics_csv="data/reddit_with_topics.csv",     # has: post_id, subreddit, clean_text, topic_id
    sentiment_csv="data/reddit_with_sentiment.csv",    # has: post_id, sent_label, sent_score
    topic_info_csv="data/topic_info.csv",              # has: topic_id, top_terms (from KMeans)
    out_topics_csv="data/topic_sentiment_summary.csv",
    out_subtopic_csv="data/subreddit_topic_sentiment.csv",
):
    # ---- Load
    dt = pd.read_csv(docs_topics_csv)
    ds = pd.read_csv(sentiment_csv)
    ti = pd.read_csv(topic_info_csv)  # columns: topic_id, top_terms

    need_dt = {"post_id", "subreddit", "clean_text", "topic_id"}
    need_ds = {"post_id", "sent_label", "sent_score"}
    assert need_dt.issubset(dt.columns), f"Missing in docs_topics: {need_dt - set(dt.columns)}"
    assert need_ds.issubset(ds.columns), f"Missing in sentiment: {need_ds - set(ds.columns)}"

    # ---- Merge on post_id to align rows
    df = dt.merge(ds[["post_id", "sent_label", "sent_score"]], on="post_id", how="inner")
    df["sent_label"] = df["sent_label"].str.lower()

    def count_eq(s, v): return (s == v).sum()

    # ---- Per-topic summary
    topics = (
        df.groupby("topic_id")
          .agg(
              n=("clean_text", "count"),
              pos=("sent_label", lambda s: count_eq(s, "positive")),
              neu=("sent_label", lambda s: count_eq(s, "neutral")),
              neg=("sent_label", lambda s: count_eq(s, "negative")),
              avg_score=("sent_score", "mean"),
          )
          .reset_index()
          .sort_values("n", ascending=False)
    )
    for c in ["pos", "neu", "neg"]:
        topics[f"{c}_pct"] = (topics[c] / topics["n"]).round(3)

    # attach human-readable topic name (top terms)
    topics = topics.merge(ti[["topic_id", "top_terms"]], on="topic_id", how="left")

    # ---- Per-subreddit × topic (compare communities)
    subtopic = (
        df.groupby(["subreddit", "topic_id"])
          .agg(
              n=("clean_text", "count"),
              pos=("sent_label", lambda s: count_eq(s, "positive")),
              neu=("sent_label", lambda s: count_eq(s, "neutral")),
              neg=("sent_label", lambda s: count_eq(s, "negative")),
              avg_score=("sent_score", "mean"),
          )
          .reset_index()
          .sort_values(["subreddit", "n"], ascending=[True, False])
    )
    for c in ["pos", "neu", "neg"]:
        subtopic[f"{c}_pct"] = (subtopic[c] / subtopic["n"]).round(3)
    subtopic = subtopic.merge(ti[["topic_id", "top_terms"]], on="topic_id", how="left")

    # ---- Save
    Path("data").mkdir(exist_ok=True)
    topics.to_csv(out_topics_csv, index=False)
    subtopic.to_csv(out_subtopic_csv, index=False)

    print("✅ Topic × sentiment →", out_topics_csv)
    print(topics.head(10)[["topic_id","n","pos_pct","neu_pct","neg_pct","top_terms"]])
    print("\n✅ Subreddit × topic × sentiment →", out_subtopic_csv)
    print(subtopic.head(10)[["subreddit","topic_id","n","pos_pct","neu_pct","neg_pct","top_terms"]])

if __name__ == "__main__":
    run()
