# Prediction Analysis Control: Sentiment Analysis Integration

## Overview

The Prediction Analysis Control (PAC) incorporates advanced sentiment analysis capabilities, leveraging social media, news, and earnings transcript data to enhance prediction accuracy. This document details the sentiment analysis architecture, methodologies, and integration points within the PAC.

## Sentiment Data Sources

The PAC integrates multiple sentiment data sources:

```csharp
// Sentiment service instances
private static TwitterSentimentService _twitterSentimentService = new TwitterSentimentService();
private readonly ISocialMediaSentimentService _financialNewsSentimentService;
private readonly IEarningsTranscriptService _earningsTranscriptService;
private readonly IAnalystRatingService _analystRatingService;
private readonly IInsiderTradingService _insiderTradingService;
```

### Social Media Sentiment

1. **Twitter Sentiment Analysis**
   - Real-time tweet collection for specified cashtags
   - NLP-based sentiment scoring (positive/negative/neutral)
   - Volume-weighted sentiment aggregation
   - Influential account identification and weighting

### News Sentiment

The `FinancialNewsSentimentService` processes financial news articles:

- Major financial news sources monitoring
- Article relevance scoring
- Sentiment classification (bullish/bearish/neutral)
- Topic extraction and categorization
- Entity recognition (companies, products, executives)

### Earnings Transcript Analysis

The `EarningsTranscriptService` performs deep analysis of earnings call transcripts:

- Automated transcript retrieval
- Sentiment analysis of management language
- Key topic extraction
- Forward-looking statement identification
- Q&A sentiment divergence analysis

### Analyst Ratings

The `AnalystRatingService` aggregates professional analyst opinions:

- Consensus rating tracking
- Rating change monitoring
- Price target analysis
- Analyst accuracy weighting

### Insider Trading

The `InsiderTradingService` monitors insider trading patterns:

- Form 4 filing analysis
- Transaction categorization (planned vs. opportunistic)
- Volumetric analysis of insider activity
- Notable executive transaction highlighting

## Sentiment Processing Pipeline

The PAC implements a multi-stage sentiment processing pipeline:

1. **Collection**: Gathering raw sentiment data from multiple sources
2. **Filtering**: Removing noise and irrelevant content
3. **Analysis**: Applying NLP techniques to extract sentiment
4. **Integration**: Combining sentiment signals with technical analysis
5. **Correlation**: Analyzing sentiment-price relationships
6. **Visualization**: Rendering sentiment metrics in the UI

## Core Sentiment Integration Methods

### Sentiment Price Correlation Analysis

```csharp
// Analyze correlation between sentiment and price movements
private async Task AnalyzeSentimentPriceCorrelation(string symbol)
{
    try
    {
        // Get historical prices
        var historicalPrices = await _alphaVantageService.GetHistoricalPricesAsync(
            symbol, "daily", 90);
            
        // Get Twitter sentiment for same period
        var twitterSentiment = await _twitterSentimentService.GetHistoricalSentimentAsync(
            symbol, 90);
            
        // Calculate correlation between price changes and sentiment
        var priceChanges = CalculatePriceChanges(historicalPrices);
        var correlation = CalculatePearsonCorrelation(
            priceChanges.ToArray(), 
            twitterSentiment.Select(s => s.Score).ToArray());
            
        _lastSentimentCorrelation = new SentimentCorrelation
        {
            Symbol = symbol,
            TwitterSentimentCorrelation = correlation,
            SampleSize = Math.Min(priceChanges.Count, twitterSentiment.Count),
            CalculationDate = DateTime.Now
        };
        
        // Store sentiment metrics
        sentimentScore = twitterSentiment.Average(s => s.Score);
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, 
            $"Error analyzing sentiment-price correlation for {symbol}");
    }
}
```

### Adding Sentiment Data to Predictions

The PAC enriches prediction models with sentiment data:

```csharp
// Simplified sentiment integration
private void IntegrateSentimentIntoPrediction(ref PredictionModel predictionModel)
{
    // Add Twitter sentiment
    if (sentimentScore != 0)
    {
        predictionModel.Indicators["TwitterSentiment"] = sentimentScore;
        
        // Adjust confidence based on sentiment-price correlation
        if (_lastSentimentCorrelation != null && 
            Math.Abs(_lastSentimentCorrelation.TwitterSentimentCorrelation) > 0.4)
        {
            // Strong correlation - sentiment is predictive for this symbol
            double sentimentAdjustment = sentimentScore * 0.1 * 
                Math.Sign(_lastSentimentCorrelation.TwitterSentimentCorrelation);
                
            predictionModel.Confidence = Math.Min(0.95, 
                Math.Max(0.05, predictionModel.Confidence + sentimentAdjustment));
        }
    }
    
    // Add analyst consensus
    if (indicators.ContainsKey("AnalystConsensus"))
    {
        double analystConsensus = indicators["AnalystConsensus"];
        int analystBuyCount = (int)indicators["AnalystBuyCount"];
        int analystSellCount = (int)indicators["AnalystSellCount"];
        int analystHoldCount = (int)indicators["AnalystHoldCount"];
        
        // Format analyst breakdown for notes
        string analystBreakdown = $"Analyst Ratings: Buy: {analystBuyCount}, " +
            $"Hold: {analystHoldCount}, Sell: {analystSellCount}";
            
        predictionModel.Notes += string.IsNullOrEmpty(predictionModel.Notes) ? 
            analystBreakdown : $"\n{analystBreakdown}";
    }
    
    // Additional sentiment sources integration...
}
```

### Earnings Transcript Analysis

The PAC extracts valuable insights from earnings call transcripts:

```csharp
// Simplified earnings transcript analysis
private async Task<EarningsTranscriptAnalysisResult> AnalyzeEarningsTranscript(string symbol)
{
    try
    {
        // Get latest transcript
        var transcript = await _earningsTranscriptService.GetLatestTranscriptAsync(symbol);
        if (transcript == null)
            return null;
            
        // Perform NLP analysis
        var analysis = await _earningsTranscriptService.AnalyzeTranscriptAsync(transcript);
        
        // Extract key information
        earningsKeyTopics = string.Join(", ", analysis.KeyTopics);
        if (analysis.NamedEntities.TryGetValue("ORG", out var orgs))
        {
            earningsKeyEntities = string.Join(", ", orgs.Take(5));
        }
        earningsQuarter = analysis.Quarter;
        
        return analysis;
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, 
            $"Error analyzing earnings transcript for {symbol}");
        return null;
    }
}
```

### Insider Trading Analysis

The PAC incorporates insider trading analysis into predictions:

```csharp
// Simplified insider trading analysis
private async Task<Dictionary<string, double>> AnalyzeInsiderTrading(string symbol)
{
    try
    {
        // Get recent insider transactions
        var transactions = await _insiderTradingService.GetRecentTransactionsAsync(symbol);
        if (transactions == null || !transactions.Any())
            return new Dictionary<string, double>();
            
        // Calculate net insider sentiment
        double buyVolume = transactions
            .Where(t => t.TransactionType == "BUY")
            .Sum(t => t.SharesTraded * t.Price);
            
        double sellVolume = transactions
            .Where(t => t.TransactionType == "SELL")
            .Sum(t => t.SharesTraded * t.Price);
            
        double totalVolume = buyVolume + sellVolume;
        double insiderSentiment = totalVolume > 0 ? 
            (buyVolume - sellVolume) / totalVolume : 0;
            
        // Extract notable insiders
        var notableInsiders = transactions
            .Where(t => t.IsNotableInsider)
            .GroupBy(t => t.InsiderName)
            .ToDictionary(
                g => $"Notable_{g.Key.Replace(" ", "_")}", 
                g => g.Sum(t => t.TransactionType == "BUY" ? 1.0 : -1.0)
            );
            
        // Combine results
        var results = new Dictionary<string, double>
        {
            ["InsiderTradingSentiment"] = insiderSentiment
        };
        
        foreach (var insider in notableInsiders)
        {
            results[insider.Key] = insider.Value;
        }
        
        return results;
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, 
            $"Error analyzing insider trading for {symbol}");
        return new Dictionary<string, double>();
    }
}
```

## Sentiment Visualization

The PAC includes sophisticated sentiment visualization components:

### Sentiment Visualization Control

```xml
<!-- Sentiment visualization in XAML -->
<local:SentimentVisualizationControl 
    x:Name="SentimentVisualizer"
    Symbol="{Binding Symbol}"
    SentimentScore="{Binding SentimentScore}" 
    SentimentTrend="{Binding SentimentTrend}"
    SentimentBreakdown="{Binding SentimentBreakdown}"
    Height="200" />
```

This control provides:

- Heat map visualization of sentiment intensity
- Source-specific sentiment breakdown
- Time-series sentiment trend charts
- Correlation visualization with price movements

## Natural Language Processing Techniques

The PAC employs multiple NLP techniques for sentiment analysis:

### 1. BERT-Based Sentiment Classification

Financial domain-specific BERT models fine-tuned for:
- Financial sentiment classification
- Prediction of price movement from text

```csharp
// BERT model integration (simplified)
private async Task<double> GetBERTSentimentScore(string text)
{
    // Call Python service for BERT inference
    var result = await _nlpService.GetSentimentScoreAsync(text, "BERT");
    return result.Score; // -1.0 to 1.0 range
}
```

### 2. Named Entity Recognition

Extracts relevant entities from text:
- Company names and tickers
- Products and services
- Executives and key personnel
- Geographic locations
- Regulatory agencies

### 3. Topic Modeling

Identifies key topics in text documents:
- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)
- BERTopic for contextual topic extraction

### 4. Emotion Detection

Beyond positive/negative sentiment:
- Confidence/uncertainty detection
- Forward-looking statement identification
- Risk assessment language detection

## Sentiment-Enhanced Signal Generation

The PAC leverages sentiment in multiple ways to enhance trading signals:

### 1. Sentiment Extremes Signal

```csharp
// Simplified sentiment extreme detection
private bool IsSentimentExtreme(double sentimentScore, double threshold = 0.7)
{
    return Math.Abs(sentimentScore) > threshold;
}
```

When sentiment reaches extreme levels, it can indicate potential reversals.

### 2. Sentiment-Technical Confirmation

```csharp
// Simplified sentiment-technical confirmation
private bool IsSentimentTechnicalConfirmation(
    string technicalSignal, double sentimentScore)
{
    if (technicalSignal == "BUY" && sentimentScore > 0.3)
        return true;
    if (technicalSignal == "SELL" && sentimentScore < -0.3)
        return true;
    return false;
}
```

Combining sentiment with technical signals enhances prediction confidence.

### 3. Sentiment Divergence

```csharp
// Simplified sentiment divergence detection
private bool IsSentimentDivergence(
    double priceChange, double sentimentChange)
{
    return Math.Sign(priceChange) != Math.Sign(sentimentChange) && 
        Math.Abs(sentimentChange) > 0.4;
}
```

When sentiment diverges from price action, it can indicate potential trend changes.

## OpenAI Integration

The PAC integrates with OpenAI's GPT models for enhanced sentiment analysis:

```csharp
// OpenAI service integration
private readonly OpenAIPredictionEnhancementService _openAiService = 
    new OpenAIPredictionEnhancementService();
```

This integration provides:

- Enhanced news interpretation
- Contextual sentiment analysis
- Narrative extraction from financial texts
- Pattern identification in market sentiment

## Sentiment Database Schema

The PAC stores sentiment data using the following schema:

```
Table: SentimentScores
- ID (Primary Key)
- Symbol (varchar)
- Source (varchar, e.g., "Twitter", "News")
- Score (float, -1.0 to 1.0)
- Volume (int, number of mentions/posts)
- Date (datetime)
- RawData (text, optional sample of raw data)

Table: SentimentCorrelations
- ID (Primary Key)
- Symbol (varchar)
- Source (varchar)
- Correlation (float, -1.0 to 1.0)
- Timeframe (varchar, e.g., "daily", "weekly")
- SampleSize (int)
- CalculationDate (datetime)
```

## Performance Considerations

Sentiment analysis has specific performance implications:

1. **API Rate Limiting**: Social media APIs have usage limits
2. **NLP Processing Overhead**: NLP operations can be CPU-intensive
3. **Data Storage**: Sentiment history requires substantial storage
4. **Real-time Requirements**: Trade-off between recency and processing time

The PAC addresses these with:
- Asynchronous processing
- Scheduled batch updates
- Caching of sentiment results
- Parallelized source processing

## Next Steps

For information on how the PAC integrates automated trading capabilities with its sentiment-enhanced predictions, see [Automation and Trading Features](5_Automation_and_Trading_Features.md).