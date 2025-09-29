import os
import time
from datetime import datetime
import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Connect to Reddit
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

tech_subreddits = ["technology", "gadgets", "technews", "Futurology", "tech", "computing", "techsupport", "programming", "cybersecurity", "MachineLearning","ArtificialInteligence"]

HOT_POST_LIMIT = 100
NEW_POST_LIMIT = 100

OUTPUT_CSV = "reddit_tech_posts.csv"

def fetch_posts(subreddit_name, limit, sort_by="hot"):
    subreddit = reddit.subreddit(subreddit_name)
    if sort_by == "hot":
        posts = subreddit.hot(limit=limit)
    elif sort_by == "new":
        posts = subreddit.new(limit=limit)
    else:
        raise ValueError("sort_by must be 'hot' or 'new'")

    post_list = []
    for post in posts:
        post_data = {
            "subreddit": subreddit_name,
            "post_id": post.id,
            "text": post.title + "\n\n" + (post.selftext if post.selftext else ""),
            "url": post.url,
            "created_utc": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            "score": post.score,
            "num_comments": post.num_comments,
            "author": str(post.author),
            "permalink": f"https://www.reddit.com{post.permalink}",
            "fetched_at": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }
        post_list.append(post_data)
    return post_list
def main():
    all_posts = []
    for subreddit in tech_subreddits:
        print(f"Fetching hot posts from r/{subreddit}")
        hot_posts = fetch_posts(subreddit, HOT_POST_LIMIT, sort_by="hot")
        all_posts.extend(hot_posts)
        time.sleep(2) 

        print(f"Fetching new posts from r/{subreddit}")
        new_posts = fetch_posts(subreddit, NEW_POST_LIMIT, sort_by="new")
        all_posts.extend(new_posts)
        time.sleep(2) 

    df = pd.DataFrame(all_posts)
    df.drop_duplicates(subset=["post_id"], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} posts to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
