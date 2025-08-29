# openai_sentiment_analysis.py
# Integrates OpenAI's API for advanced sentiment analysis
import sys
import json
import os
import re
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('openai_sentiment')

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available. Please install with: pip install openai")
    OPENAI_AVAILABLE = False

def extract_sentiment_score(text):
    """Extract a sentiment score from OpenAI's response text."""
    
    # First try to find a direct score mention
    score_pattern = r"[-+]?\d*\.\d+|\d+"
    score_matches = re.findall(score_pattern, text)
    
    if score_matches:
        try:
            # Try to extract the score, prioritizing numbers between -1 and 1
            for match in score_matches:
                score = float(match)
                if -1 <= score <= 1:
                    return score
            
            # If no score in the right range, take the first score and normalize it
            score = float(score_matches[0])
            return max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
        except (ValueError, IndexError) as e:
            logger.error(f"Error extracting sentiment score: {e}")
    
    # If no direct score, analyze the text for sentiment keywords
    bullish_keywords = ['bullish', 'positive', 'strong', 'optimistic', 'growth', 'uptrend']
    bearish_keywords = ['bearish', 'negative', 'weak', 'pessimistic', 'decline', 'downtrend']
    
    text_lower = text.lower()
    bullish_count = sum(text_lower.count(keyword) for keyword in bullish_keywords)
    bearish_count = sum(text_lower.count(keyword) for keyword in bearish_keywords)
    
    if bullish_count > bearish_count:
        return 0.5  # Default positive score
    elif bearish_count > bullish_count:
        return -0.5  # Default negative score
    else:
        return 0.0  # Neutral

def analyze_sentiment_with_openai(texts, api_key, model="gpt-3.5-turbo"):
    """
    Analyze sentiment of text content using OpenAI API.
    
    Args:
        texts: List of text strings to analyze
        api_key: OpenAI API key
        model: Model to use (default: gpt-3.5-turbo)
        
    Returns:
        Float sentiment score from -1 (bearish) to 1 (bullish)
    """
    if not OPENAI_AVAILABLE:
        logger.error("OpenAI package not available. Cannot analyze sentiment.")
        return 0.0
    
    if not api_key:
        logger.error("No OpenAI API key provided.")
        return 0.0
    
    if not texts or len(texts) == 0:
        logger.warning("No text content to analyze.")
        return 0.0
    
    # Truncate and join texts to prevent token limit issues
    max_texts = min(5, len(texts))
    combined_text = "\n---\n".join(texts[:max_texts])
    
    # Truncate if too long (approximate token count)
    max_chars = 6000  # Approximately 1500 tokens
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars] + "..."
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Create a detailed prompt for sentiment analysis
        prompt = (
            f"Analyze the sentiment in the following financial texts related to stocks or companies. "
            f"Is the sentiment bullish or bearish? "
            f"Provide a sentiment score between -1.0 (extremely bearish) and 1.0 (extremely bullish), "
            f"with 0 being neutral. Return your analysis followed by the numeric score.\n\n"
            f"Text: {combined_text}"
        )
        
        # Call the API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyst specialized in stock market analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent results
            max_tokens=1000
        )
        
        # Extract the sentiment score from the response
        response_text = response.choices[0].message.content
        sentiment_score = extract_sentiment_score(response_text)
        
        logger.info(f"OpenAI sentiment analysis completed. Score: {sentiment_score}")
        return sentiment_score
        
    except Exception as e:
        logger.error(f"Error in OpenAI sentiment analysis: {str(e)}")
        return 0.0

def enrich_prediction_with_openai(prediction_data, texts, api_key, model="gpt-3.5-turbo"):
    """
    Enhances a stock prediction with OpenAI-generated insights.
    
    Args:
        prediction_data: Dictionary containing the prediction
        texts: List of text content for context
        api_key: OpenAI API key
        model: Model to use
        
    Returns:
        Enhanced prediction dictionary
    """
    if not OPENAI_AVAILABLE or not api_key or not texts:
        return prediction_data
    
    try:
        # Extract key prediction parameters
        symbol = prediction_data.get('symbol', 'Unknown')
        action = prediction_data.get('action', 'HOLD')
        target_price = prediction_data.get('targetPrice', 0)
        current_price = prediction_data.get('currentPrice', 0)
        confidence = prediction_data.get('confidence', 0.5)
        
        # Truncate and join texts
        max_texts = min(3, len(texts))
        combined_text = "\n---\n".join(texts[:max_texts])
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000] + "..."
        
        client = OpenAI(api_key=api_key)
        
        # Create prompt for prediction enhancement
        prompt = (
            f"Based on the following market information about {symbol} and a prediction model that suggests "
            f"a {action} action with {confidence*100:.1f}% confidence and a target price of ${target_price:.2f} "
            f"(current price: ${current_price:.2f}):\n\n{combined_text}\n\n"
            f"1. Provide a brief explanation (2-3 sentences) of whether this prediction aligns with the market sentiment "
            f"2. Identify any key risks or catalysts mentioned in the text that could impact this prediction "
            f"3. Suggest one additional factor that might not be captured by the model"
        )
        
        # Call the API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst that helps explain trading predictions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500
        )
        
        # Extract the response and add it to the prediction
        explanation = response.choices[0].message.content
        
        # Add the explanation to the prediction data
        if 'analysisDetails' not in prediction_data:
            prediction_data['analysisDetails'] = ""
            
        prediction_data['analysisDetails'] += f"\n\n## OpenAI Enhanced Analysis\n{explanation}"
        prediction_data['openAiEnhanced'] = True
        
        logger.info(f"Enhanced prediction for {symbol} with OpenAI analysis")
        return prediction_data
        
    except Exception as e:
        logger.error(f"Error enhancing prediction with OpenAI: {str(e)}")
        return prediction_data

def main():
    """Main function to process input from stdin."""
    try:
        # Read JSON from stdin
        input_json = sys.stdin.readline()
        data = json.loads(input_json)
        
        # Check what type of analysis is requested
        analysis_type = data.get("type", "sentiment")
        api_key = data.get("apiKey", os.environ.get("OPENAI_API_KEY", ""))
        model = data.get("model", "gpt-3.5-turbo")
        
        if analysis_type == "sentiment":
            # Basic sentiment analysis
            texts = data.get("texts", [])
            score = analyze_sentiment_with_openai(texts, api_key, model)
            result = {"score": score}
            
        elif analysis_type == "enhance_prediction":
            # Enhance a prediction with OpenAI insights
            prediction = data.get("prediction", {})
            texts = data.get("texts", [])
            enhanced_prediction = enrich_prediction_with_openai(prediction, texts, api_key, model)
            result = {"prediction": enhanced_prediction}
            
        else:
            logger.error(f"Unknown analysis type: {analysis_type}")
            result = {"error": f"Unknown analysis type: {analysis_type}"}
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(json.dumps({"error": str(e), "score": 0.0}))

if __name__ == "__main__":
    main()