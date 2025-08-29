# YouTube Sentiment Analysis Integration

## Overview

The YouTube Sentiment Analysis module provides the capability to analyze sentiment from YouTube videos, particularly Bloomberg 24/7 live streams and financial content. It integrates audio extraction, speech-to-text transcription, and financial sentiment analysis.

## Components

### C# Service: YouTubeSentimentService

The `YouTubeSentimentService` implements `ISocialMediaSentimentService` and provides:

- **Symbol-based analysis**: Get sentiment for specific stock symbols from YouTube content
- **URL-based analysis**: Analyze sentiment from specific YouTube URLs
- **Multi-source analysis**: Combine Bloomberg, company, and news YouTube content
- **Caching**: Cache results to avoid redundant processing
- **Integration**: Works with existing sentiment analysis ecosystem

#### Key Methods

```csharp
// Get overall sentiment for a symbol
Task<double> GetSymbolSentimentAsync(string symbol)

// Analyze sentiment from a list of URLs
Task<double> AnalyzeSentimentAsync(List<string> urls)

// Get detailed sentiment breakdown by source
Task<Dictionary<string, double>> GetDetailedSourceSentimentAsync(string symbol)

// Analyze a single YouTube URL
Task<double> AnalyzeYouTubeUrlSentimentAsync(string url, string context)
```

### Python Script: youtube_sentiment_analysis.py

The Python script handles the heavy lifting:

1. **Audio Extraction**: Uses `yt-dlp` to extract audio from YouTube videos
2. **Transcription**: Uses OpenAI Whisper for speech-to-text conversion
3. **Sentiment Analysis**: Uses OpenAI GPT models for financial sentiment analysis
4. **Fallback Mode**: Provides mock results when dependencies aren't available

#### Dependencies

```bash
pip install yt-dlp openai-whisper openai
```

#### Usage

```python
# Single URL analysis
echo '{"url": "https://youtube.com/watch?v=...", "context": "Bloomberg financial news"}' | python3 youtube_sentiment_analysis.py

# Multiple URLs
echo '[{"url": "...", "context": "..."}, {"url": "...", "context": "..."}]' | python3 youtube_sentiment_analysis.py
```

## Integration Example

```csharp
// Initialize the service
var youTubeService = new YouTubeSentimentService(logger, configManager);

// Analyze sentiment for Apple stock from YouTube content
var appleSentiment = await youTubeService.GetSymbolSentimentAsync("AAPL");

// Analyze specific Bloomberg video
var bloombergSentiment = await youTubeService.AnalyzeYouTubeUrlSentimentAsync(
    "https://www.youtube.com/watch?v=dp8PhLsUcFE", 
    "Bloomberg TV Live"
);

// Get detailed breakdown by source
var detailedSentiment = await youTubeService.GetDetailedSourceSentimentAsync("AAPL");
// Returns: {"Bloomberg_YouTube": 0.3, "Company_YouTube": 0.1, "News_YouTube": 0.2}
```

## Configuration

The service uses existing configuration classes:

```json
{
  "ApiConfig": {
    "OpenAI": {
      "ApiKey": "your-openai-api-key",
      "BaseUrl": "https://api.openai.com/v1",
      "DefaultTimeout": 30
    }
  },
  "SentimentAnalysisConfig": {
    "OpenAI": {
      // OpenAI sentiment analysis configuration
    }
  }
}
```

## Bloomberg 24/7 Live Stream Focus

The service is specifically designed for Bloomberg's financial content:

- **Live Streams**: Supports Bloomberg TV Live and Bloomberg Markets
- **Financial Context**: Uses financial-specific sentiment prompts
- **Speaker Awareness**: Handles multiple speakers in financial broadcasts
- **Market Relevance**: Focuses on market-moving content and analysis

## Performance Considerations

- **Caching**: Results are cached for 30 minutes to avoid redundant processing
- **Fallback Mode**: Gracefully handles missing dependencies
- **Async Processing**: All operations are asynchronous
- **Error Handling**: Comprehensive error handling with logging
- **Resource Management**: Proper cleanup of temporary audio files

## Testing

The module includes comprehensive tests:

```csharp
// Run basic functionality tests
var results = await YouTubeSentimentServiceTests.RunBasicTestAsync();

// Test URL analysis
var urlResults = await YouTubeSentimentServiceTests.RunUrlAnalysisTestAsync();
```

## Future Enhancements

1. **Real-time Processing**: Live stream sentiment monitoring
2. **Speaker Identification**: Identify and weight different speakers
3. **Enhanced Search**: Automatic YouTube content discovery for symbols
4. **Sentiment Alerts**: Real-time alerts based on sentiment changes
5. **Visualization**: Charts showing sentiment trends over time