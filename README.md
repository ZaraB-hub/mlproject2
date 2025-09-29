Reddit Tech Trends Analysis

This project collects and analyses posts from technology-focused subreddits to detect trending topics and sentiment. It demonstrates a complete NLP pipeline from data collection to clustering and visualization.

Features

Scrapes subreddit posts (title and body) with PRAW

Cleans and preprocesses text (lowercasing, URL/mention removal, hashtag extraction)

Applies sentiment analysis using a pretrained Hugging Face model

Detects trending topics with TF-IDF and KMeans clustering

Aggregates results to show sentiment distribution across topics and subreddits


Tech Stack

Python (pandas, scikit-learn)

Hugging Face Transformers (RoBERTa sentiment model)

TF-IDF + KMeans for topic detection

Streamlit for visualization

Reddit API (PRAW)

