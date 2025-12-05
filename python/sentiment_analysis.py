# sentiment_analysis.py
# Place this file in a 'python' folder at the root of your project.
# Requires: pip install torch transformers
import sys
import json
import torch
from transformers import pipeline

# Import debugpy utilities for remote debugging support
try:
    from debugpy_utils import init_debugpy_if_enabled
    DEBUGPY_UTILS_AVAILABLE = True
except ImportError:
    DEBUGPY_UTILS_AVAILABLE = False

def main():
    # Initialize debugpy remote debugging if DEBUGPY environment variable is set
    if DEBUGPY_UTILS_AVAILABLE:
        init_debugpy_if_enabled()
    
    # Read JSON list of tweets from stdin
    input_json = sys.stdin.readline()
    try:
        tweets = json.loads(input_json)
    except Exception as e:
        print(0.0)
        sys.exit(1)

    if not isinstance(tweets, list) or len(tweets) == 0:
        print(0.0)
        sys.exit(0)

    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

    # Analyze each tweet
    scores = []
    for tweet in tweets:
        if not tweet or not isinstance(tweet, str):
            continue
        try:
            result = sentiment_pipeline(tweet[:512])[0]  # Truncate to 512 tokens
            label = result["label"].upper()
            score = result["score"]
            # Map to -1 (NEGATIVE), +1 (POSITIVE)
            if label == "POSITIVE":
                scores.append(score)
            elif label == "NEGATIVE":
                scores.append(-score)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)

    # Compute average sentiment
    avg_sentiment = sum(scores) / len(scores) if scores else 0.0
    print(f"{avg_sentiment:.4f}")

if __name__ == "__main__":
    main()
