# src/analyze_tweets.py
from transformers import pipeline
import json

# Load tweets from a file (assumes one tweet per line in JSON format)
def load_tweets(file_path="data/tweets.json"):
    with open(file_path, "r") as f:
        return [json.loads(line)["text"] for line in f]

# Sentiment classifier (binary urgency classifier)
sentiment_pipeline = pipeline("sentiment-analysis")

# Named Entity Recognizer (to extract locations)
ner_pipeline = pipeline("ner", grouped_entities=True)

def analyze_tweets():
    tweets = load_tweets()
    for tweet in tweets:
        sentiment = sentiment_pipeline(tweet)[0]
        entities = ner_pipeline(tweet)
        
        locations = [ent["word"] for ent in entities if ent["entity_group"] == "LOC"]
        
        print(f"\nTweet: {tweet}")
        print(f"Urgency: {sentiment['label']} ({sentiment['score']:.2f})")
        print(f"Locations: {locations}")

if __name__ == "__main__":
    analyze_tweets()