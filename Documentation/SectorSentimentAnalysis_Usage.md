# Sector-Specific Sentiment Analysis

This document provides a guide on how to use the sector-specific sentiment analysis features in Quantra.

## Overview

The sector-specific sentiment analysis module identifies, evaluates, and visualizes sentiment trends uniquely within individual market sectors (e.g., technology, healthcare, financials), enabling targeted insights, precise market assessments, and more informed, sector-oriented trading decisions.

## Key Features

1. **Sector-level sentiment aggregation**: Analyze sentiment across all stocks in a specific sector
2. **Cross-sector comparison**: Compare sentiment trends across different market sectors
3. **Sector news relevance detection**: Identify news articles most relevant to specific sectors
4. **Sector sentiment visualization**: View sentiment trends, breakdowns, and correlations for sectors
5. **Sector sentiment/performance correlation**: Analyze how sector sentiment correlates with sector performance

## Using the API

### Basic Sector Sentiment Analysis

```csharp
// Create the service
var sectorSentimentService = new SectorSentimentAnalysisService();

// Get overall sentiment for a specific sector
double techSentiment = await sectorSentimentService.GetSectorSentimentAsync("Technology");
Console.WriteLine($"Technology sector sentiment: {techSentiment}");

// Get detailed sentiment for all stocks in a sector
var techStockSentiments = await sectorSentimentService.GetDetailedSectorSentimentAsync("Technology");
foreach (var stock in techStockSentiments)
{
    Console.WriteLine($"{stock.Key}: {stock.Value}");
}

// Get sentiment for all sectors (for comparison)
var allSectorSentiments = await sectorSentimentService.GetAllSectorsSentimentAsync();
```

### Advanced Sector Analysis

```csharp
// Get detailed news analysis for a sector
var (sentimentBySource, articles) = 
    await sectorSentimentService.GetSectorNewsAnalysisAsync("Healthcare");

// Display sentiment by news source
foreach (var source in sentimentBySource)
{
    Console.WriteLine($"{source.Key}: {source.Value}");
}

// Display the most relevant sector articles
foreach (var article in articles.Take(5))
{
    Console.WriteLine($"Title: {article.Title}");
    Console.WriteLine($"Sentiment: {article.SentimentScore}");
    Console.WriteLine($"Sector Relevance: {article.SectorRelevance}");
}

// Get sector sentiment trend over time
var trend = await sectorSentimentService.GetSectorSentimentTrendAsync("Financial", 30);
foreach (var point in trend)
{
    Console.WriteLine($"{point.Date.ToShortDateString()}: {point.Sentiment}");
}

// Compare multiple sectors
var sectors = new List<string> { "Technology", "Financial", "Healthcare" };
var comparison = await sectorSentimentService.CompareSectorSentimentTrendsAsync(sectors);
```

### Sector Sentiment Correlation Analysis

```csharp
// Create the correlation analysis service
var correlationAnalysis = new SentimentPriceCorrelationAnalysis();

// Analyze correlation between sector sentiment and performance
var correlation = await correlationAnalysis.AnalyzeSectorSentimentCorrelation("Technology");

Console.WriteLine($"Correlation: {correlation.OverallCorrelation}");
Console.WriteLine($"Lead/Lag Relationship: {correlation.LeadLagRelationship} days");

// Check sentiment shift events
foreach (var shiftEvent in correlation.SentimentShiftEvents)
{
    Console.WriteLine($"Date: {shiftEvent.Date.ToShortDateString()}");
    Console.WriteLine($"Sentiment Shift: {shiftEvent.SentimentShift}");
    Console.WriteLine($"Subsequent Price Change: {shiftEvent.SubsequentPriceChange}%");
    Console.WriteLine($"Price Followed Sentiment: {shiftEvent.PriceFollowedSentiment}");
}
```

## UI Components

### Sector Sentiment Visualization View

The `SectorSentimentVisualizationView` provides a complete UI for visualizing sector sentiment data:

```csharp
// Add the view to your window/control
var sectorSentimentView = new SectorSentimentVisualizationView();
MainGrid.Children.Add(sectorSentimentView);

// Set up the controller
var controller = new SectorSentimentController();
controller.Initialize(sectorSentimentView);

// The controller handles data loading and visualization updates
```

### Features of the Visualization View:

- Sector selection dropdown
- Comparison mode selection (All sectors, Top 5, Bottom 5, etc.)
- Sector sentiment comparison chart
- Historical sentiment trend chart
- News articles with sentiment indicators
- Source sentiment breakdown
- Trending topics in the sector

## Incorporating into Trading Strategies

Sector sentiment data can be used to enhance trading strategies:

1. **Sector rotation strategies**: Invest in sectors with improving sentiment trends
2. **Cross-sector correlations**: Identify leading/lagging relationships between sector sentiment shifts
3. **News-based trading signals**: Generate trading signals based on significant sector sentiment shifts
4. **Sector momentum enhancement**: Combine sector sentiment with price momentum for stronger signals
5. **Risk management**: Adjust position sizing based on sector sentiment volatility

## Database and Caching

The sector sentiment analysis services use caching to improve performance:

- Default cache duration is 30 minutes
- Call `ClearCache()` to force a refresh of all cached data
- Call `ClearCache("Technology")` to refresh data for a specific sector

## Example Workflow

A typical workflow for using sector sentiment analysis:

1. Monitor overall sentiment across all sectors
2. Identify sectors with notable sentiment changes or extremes
3. Drill down into specific sectors for detailed analysis
4. Examine sector news articles and trending topics
5. Analyze correlation with sector performance
6. Generate trading signals based on significant sentiment shifts
7. Monitor sentiment trends for position management