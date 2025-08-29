# earnings_transcript_analysis.py
# Advanced NLP analysis of earnings call transcripts
# Requires: pip install torch transformers spacy nltk bertopic
import sys
import json
import torch
import numpy as np
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic
import nltk
import spacy
import re

# Download NLTK resources (if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback if model isn't installed
    import subprocess
    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def analyze_sentiment(text_chunks, device):
    """Analyze sentiment of text chunks using FinBERT, a financial domain-specific BERT model"""
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",  # Financial domain-specific BERT
        tokenizer="ProsusAI/finbert",
        device=device
    )
    
    # Process each chunk and collect scores
    scores = []
    sentiment_labels = []
    
    for chunk in text_chunks:
        if not chunk or not isinstance(chunk, str):
            continue
            
        try:
            result = sentiment_analyzer(chunk[:512])[0]  # Truncate to 512 tokens
            label = result["label"].upper()
            score = result["score"]
            
            # Convert to numeric scores
            if label == "POSITIVE":
                scores.append(score)
                sentiment_labels.append("POSITIVE")
            elif label == "NEGATIVE":
                scores.append(-score)
                sentiment_labels.append("NEGATIVE")
            else:
                scores.append(0.0)
                sentiment_labels.append("NEUTRAL")
                
        except Exception as e:
            scores.append(0.0)
            sentiment_labels.append("ERROR")
    
    # Calculate overall sentiment score
    avg_sentiment = sum(scores) / len(scores) if scores else 0.0
    sentiment_distribution = {
        "positive": sentiment_labels.count("POSITIVE") / len(sentiment_labels) if sentiment_labels else 0,
        "negative": sentiment_labels.count("NEGATIVE") / len(sentiment_labels) if sentiment_labels else 0,
        "neutral": sentiment_labels.count("NEUTRAL") / len(sentiment_labels) if sentiment_labels else 0
    }
    
    return {
        "score": avg_sentiment,
        "distribution": sentiment_distribution
    }

def extract_named_entities(text):
    """Extract named entities using spaCy"""
    entities = {
        "ORG": [],  # Organizations
        "PERSON": [],  # People
        "GPE": [],  # Countries, cities
        "MONEY": [],  # Monetary values
        "PERCENT": [],  # Percentages
        "PRODUCT": []  # Products
    }
    
    # Process text with spaCy
    doc = nlp(text[:1000000])  # Limit text size to prevent memory issues
    
    # Extract and count entities
    entity_counts = {}
    for ent in doc.ents:
        if ent.label_ in entities:
            # Add entity to the right category
            if ent.text.lower() not in [e.lower() for e in entities[ent.label_]]:
                entities[ent.label_].append(ent.text)
            
            # Count occurrences for significance
            if ent.text not in entity_counts:
                entity_counts[ent.text] = 0
            entity_counts[ent.text] += 1
    
    # Get top entities by occurrence count
    top_entities = {k: sorted(v, key=lambda x: entity_counts.get(x, 0), reverse=True)[:10] 
                   for k, v in entities.items() if v}
    
    return top_entities

def identify_key_topics(chunks):
    """Identify main topics in the text using BERTopic"""
    if len(chunks) < 5:  # Need minimum documents for topic modeling
        # Fallback to simple keyword extraction for small documents
        all_text = " ".join(chunks)
        # Extract keywords by frequency (simple approach)
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', all_text.lower())
        word_counts = {}
        for word in words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
            
        # Filter common English words
        common_words = {'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but', 
                        'from', 'they', 'will', 'would', 'there', 'their', 'what', 'about', 'which', 
                        'when', 'make', 'like', 'time', 'just', 'him', 'know', 'take', 'people', 'year'}
        topics = {}
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            if word not in common_words and len(word) > 3:
                if len(topics) < 5:  # Get top 5 topics
                    topics[word] = count
        
        return list(topics.keys())
    else:
        try:
            # Use BERTopic for topic modeling
            topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=False)
            topics, _ = topic_model.fit_transform(chunks)
            topic_info = topic_model.get_topic_info()
            
            # Get the main topics (excluding -1 which is the noise topic)
            main_topics = []
            for idx, topic in enumerate(topic_info['Name']):
                if idx > 0 and len(main_topics) < 5:  # Get top 5 topics
                    # Clean topic name - it's in format "0_word1_word2_word3"
                    topic_name = '_'.join(topic.split('_')[1:])
                    main_topics.append(topic_name)
                    
            return main_topics
        except Exception as e:
            # Fallback in case of errors
            return ["Error in topic modeling"]

def split_text_into_chunks(text, max_length=500):
    """Split text into smaller chunks for analysis"""
    # Use NLTK to split by sentences first
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max length, start a new chunk
        if len(current_chunk) + len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def analyze_transcript(transcript):
    """Main function to analyze an earnings call transcript"""
    # First, preprocess the text to remove excessive whitespace
    cleaned_text = ' '.join(transcript.split())
    
    # Split text into chunks for analysis
    chunks = split_text_into_chunks(cleaned_text)
    
    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    
    # Perform sentiment analysis
    sentiment_results = analyze_sentiment(chunks, device)
    
    # Extract named entities
    entities = extract_named_entities(cleaned_text)
    
    # Identify key topics
    topics = identify_key_topics(chunks)
    
    return {
        "sentiment": sentiment_results,
        "entities": entities,
        "topics": topics
    }

def main():
    # Read JSON with transcript data from stdin
    input_json = sys.stdin.readline().strip()
    try:
        data = json.loads(input_json)
        transcript = data.get("transcript", "")
        if not transcript:
            print(json.dumps({"error": "No transcript provided"}))
            sys.exit(1)
            
        # Analyze transcript
        results = analyze_transcript(transcript)
        print(json.dumps(results))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()