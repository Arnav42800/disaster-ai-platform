import tweepy
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# Load API credentials from .env
load_dotenv()

API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# Authenticate
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

# Keywords to filter disaster-related tweets
KEYWORDS = ["earthquake", "flood", "fire", "hurricane", "tornado", "help", "disaster"]
MAX_TWEETS = 100

def fetch_disaster_tweets():
    tweets_data = []

    for tweet in tweepy.Cursor(api.search_tweets, q=" OR ".join(KEYWORDS), lang="en", tweet_mode='extended').items(MAX_TWEETS):
        if tweet.coordinates or tweet.place:
            data = {
                "id": tweet.id_str,
                "text": tweet.full_text,
                "created_at": tweet.created_at.isoformat(),
                "user": tweet.user.screen_name,
                "geo": tweet.coordinates or (tweet.place.full_name if tweet.place else None)
            }
            tweets_data.append(data)

    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)

    with open("data/disaster_tweets.json", "w") as f:
        json.dump(tweets_data, f, indent=2)

    print(f"Saved {len(tweets_data)} disaster-related tweets with geolocation.")

# Run the function
if __name__ == "__main__":
    fetch_disaster_tweets()

