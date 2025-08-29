#!/usr/bin/env python3
"""
YouTube Sentiment Analysis CLI Tool
Demonstrates the YouTube sentiment analysis functionality
"""

import json
import sys
import asyncio
from datetime import datetime

def test_youtube_sentiment_analysis():
    """Test the YouTube sentiment analysis with various inputs"""
    
    print("=== YouTube Sentiment Analysis Test ===")
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Test cases
    test_cases = [
        {
            "name": "Single Bloomberg URL",
            "input": {
                "url": "https://www.youtube.com/watch?v=dp8PhLsUcFE",
                "context": "Bloomberg TV Live"
            }
        },
        {
            "name": "Multiple Financial URLs",
            "input": [
                {
                    "url": "https://www.youtube.com/watch?v=dp8PhLsUcFE",
                    "context": "Bloomberg TV Live"
                },
                {
                    "url": "https://www.youtube.com/watch?v=Ga3maNZ0x0w",
                    "context": "Bloomberg Markets"
                }
            ]
        },
        {
            "name": "Company Earnings Call",
            "input": {
                "url": "https://www.youtube.com/watch?v=example123",
                "context": "Apple earnings call"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 50)
        
        # Prepare input
        input_json = json.dumps(test_case['input'])
        print(f"Input: {input_json}")
        print()
        
        # Call the Python script (simulated)
        try:
            # In a real implementation, this would call the actual script
            from youtube_sentiment_analysis import YouTubeSentimentAnalyzer
            
            analyzer = YouTubeSentimentAnalyzer()
            
            if isinstance(test_case['input'], dict):
                # Single URL
                result = analyzer.analyze_youtube_sentiment(
                    test_case['input']['url'],
                    test_case['input']['context']
                )
                print(f"Result: {json.dumps(result, indent=2)}")
            else:
                # Multiple URLs
                results = []
                for item in test_case['input']:
                    result = analyzer.analyze_youtube_sentiment(
                        item['url'],
                        item['context']
                    )
                    results.append(result)
                print(f"Results: {json.dumps(results, indent=2)}")
                
        except ImportError:
            print("Note: YouTube sentiment analysis module not fully available")
            print("This would normally call the Python script")
        except Exception as e:
            print(f"Error: {e}")
        
        print()
        print("=" * 70)
        print()
    
    print("Test completed!")

def demonstrate_c_sharp_integration():
    """Show how the C# service would integrate with this"""
    
    print("=== C# Integration Example ===")
    print()
    
    csharp_example = '''
// Example C# usage:

var youTubeService = new YouTubeSentimentService(logger, configManager);

// Analyze sentiment for a stock symbol
var appleSentiment = await youTubeService.GetSymbolSentimentAsync("AAPL");
Console.WriteLine($"Apple YouTube Sentiment: {appleSentiment:F2}");

// Analyze specific Bloomberg video
var bloombergSentiment = await youTubeService.AnalyzeYouTubeUrlSentimentAsync(
    "https://www.youtube.com/watch?v=dp8PhLsUcFE", 
    "Bloomberg TV Live"
);
Console.WriteLine($"Bloomberg Video Sentiment: {bloombergSentiment:F2}");

// Get detailed breakdown
var detailedSentiment = await youTubeService.GetDetailedSourceSentimentAsync("AAPL");
foreach (var source in detailedSentiment)
{
    Console.WriteLine($"{source.Key}: {source.Value:F2}");
}
'''
    
    print(csharp_example)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--csharp":
        demonstrate_c_sharp_integration()
    else:
        test_youtube_sentiment_analysis()