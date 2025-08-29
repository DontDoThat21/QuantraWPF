#!/usr/bin/env python3
"""
Simple test for YouTube sentiment analysis without heavy dependencies
"""
import json
import sys

def main():
    """Test function"""
    try:
        input_data = sys.stdin.read().strip()
        data = json.loads(input_data)
        
        # Mock result for testing
        result = {
            'url': data.get('url', ''),
            'timestamp': '2025-05-30T23:20:00.000000',
            'sentiment_score': 0.5,  # Mock positive sentiment
            'transcription': 'Mock financial analysis transcript about market conditions',
            'success': True,
            'error': None,
            'context': data.get('context', 'Bloomberg financial news')
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'success': False,
            'sentiment_score': 0.0
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()