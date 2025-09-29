# src/topics_kmeans.py
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def top_terms_per_cluster(vectorizer, kmeans, n_terms=10):
    terms = vectorizer.get_feature_names_out()
    centers = kmeans.cluster_centers_
    top = []
    for cid in range(centers.shape[0]):
        idx = centers[cid].argsort()[::-1][:n_terms]
        top_terms = [terms[i] for i in idx]
        top.append({"topic_id": cid, "top_terms": ", ".join(top_terms)})
    return pd.DataFrame(top)

def run(inp="data/reddit_clean.csv",
        docs_out="data/reddit_with_topics.csv",
        topics_out="data/topic_info.csv",
        n_clusters=10,
        max_features=20000):
    df = pd.read_csv(inp)
    texts = df["clean_text"].astype(str).tolist()

    # TF-IDF features (basic but effective)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,2),       # unigrams + bigrams
        min_df=3,                # ignore very rare terms
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    # Attach topic labels
    df["topic_id"] = labels
    Path("data").mkdir(exist_ok=True)
    df.to_csv(docs_out, index=False)

    # Describe topics by their top terms
    info = top_terms_per_cluster(vectorizer, kmeans, n_terms=12)
    info.to_csv(topics_out, index=False)

    print(f"✅ Docs+topics → {docs_out}")
    print(f"✅ Topic summary → {topics_out}")
    print("\n📌 Sample topics:")
    print(info.head())

if __name__ == "__main__":
    run()
