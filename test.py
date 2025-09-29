import pandas as pd
df = pd.read_csv("data/reddit_with_sentiment.csv")
print(df["sent_label"].value_counts(normalize=True).round(3))
