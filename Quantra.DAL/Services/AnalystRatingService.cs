using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Utilities;
using Quantra.Utilities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to fetch and analyze analyst ratings for stocks
    /// </summary>
    public class AnalystRatingService : IAnalystRatingService
    {
        private readonly HttpClient _client;
        private readonly string _apiKey;
        private readonly UserSettings _userSettings;
        private readonly IAlertPublisher _alertPublisher;
        
        // Cache for ratings to reduce API calls
        private readonly ConcurrentDictionary<string, (List<AnalystRating> Ratings, DateTime Timestamp)> _ratingsCache = 
            new ConcurrentDictionary<string, (List<AnalystRating>, DateTime)>();
            
        // Cache for aggregates to reduce recalculations
        private readonly ConcurrentDictionary<string, (AnalystRatingAggregate Aggregate, DateTime Timestamp)> _aggregateCache = 
            new ConcurrentDictionary<string, (AnalystRatingAggregate, DateTime)>();
            
        // Cache for historical consensus data to reduce DB calls
        private readonly ConcurrentDictionary<string, (List<AnalystRatingAggregate> History, DateTime Timestamp)> _consensusHistoryCache =
            new ConcurrentDictionary<string, (List<AnalystRatingAggregate>, DateTime)>();

        /// <summary>
        /// Constructs a new AnalystRatingService
        /// </summary>
        public AnalystRatingService(UserSettings userSettings = null, IAlertPublisher alertPublisher = null)
        {
            _client = new HttpClient();
            _apiKey = GetApiKey();
            _userSettings = userSettings ?? new UserSettings();
            _alertPublisher = alertPublisher; // can be null in tests
        }
        
        /// <summary>
        /// Gets the API key from settings or environment variables
        /// </summary>
        private string GetApiKey()
        {
            try
            {
                // First try environment variable
                string key = Environment.GetEnvironmentVariable("FINANCIAL_API_KEY");
                if (!string.IsNullOrEmpty(key))
                    return key;
                
                // Since FinancialApiKey was removed, return fallback
                DatabaseMonolith.Log("Info", "FinancialApiKey not configured, using fallback for synthetic data generation");
                return "YOUR_API_KEY";
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to get Financial API key", ex.ToString());
                return "YOUR_API_KEY"; // Fallback
            }
        }
        
        /// <summary>
        /// Retrieves recent analyst ratings for a stock symbol
        /// </summary>
        public async Task<List<AnalystRating>> GetRecentRatingsAsync(string symbol, int count = 20)
        {
            // Check cache first
            string cacheKey = $"ratings_{symbol.ToLower()}";
            if (_ratingsCache.TryGetValue(cacheKey, out var cachedData))
            {
                // If cache is still valid (not older than configured interval)
                var cacheAgeHours = (DateTime.Now - cachedData.Timestamp).TotalHours;
                int maxCacheAgeHours = _userSettings.RatingsCacheExpiryHours ?? 24; // Default 24 hours
                if (cacheAgeHours < maxCacheAgeHours)
                {
                    return cachedData.Ratings.OrderByDescending(r => r.RatingDate).Take(count).ToList();
                }
            }
            
            var ratings = new List<AnalystRating>();
            
            try
            {
                // In a real API implementation, replace this with actual API call
                // For now, generate synthetic data for demonstration purposes
                ratings = GenerateSyntheticRatingData(symbol);
                
                // In a real implementation, this would parse API response:
                /*
                string url = $"https://financial-api.example.com/analyst-ratings/{symbol}?apiKey={_apiKey}";
                var response = await _client.GetAsync(url);
                if (!response.IsSuccessStatusCode)
                {
                    DatabaseMonolith.Log("Warning", $"Analyst ratings API call failed for {symbol}: {response.StatusCode}");
                    return ratings;
                }
                
                var json = await response.Content.ReadAsStringAsync();
                var ratingsData = JsonSerializer.Deserialize<List<AnalystRating>>(json);
                ratings.AddRange(ratingsData);
                */
                
                // Cache the results
                _ratingsCache[cacheKey] = (ratings, DateTime.Now);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error fetching analyst ratings for {symbol}", ex.ToString());
            }
            
            return ratings.OrderByDescending(r => r.RatingDate).Take(count).ToList();
        }
        
        /// <summary>
        /// Gets aggregated analyst rating data for a symbol
        /// </summary>
        public async Task<AnalystRatingAggregate> GetAggregatedRatingsAsync(string symbol)
        {
            // Check cache first
            string cacheKey = $"aggregate_{symbol.ToLower()}";
            if (_aggregateCache.TryGetValue(cacheKey, out var cachedData))
            {
                // If cache is still valid (not older than configured interval)
                var cacheAgeHours = (DateTime.Now - cachedData.Timestamp).TotalHours;
                int maxCacheAgeHours = _userSettings.RatingsCacheExpiryHours ?? 24; // Default 24 hours
                if (cacheAgeHours < maxCacheAgeHours)
                {
                    return cachedData.Aggregate;
                }
            }
            
            // Fetch recent ratings
            var ratings = await GetRecentRatingsAsync(symbol, 50); // Get more ratings for better aggregation
            
            // Create aggregate
            var aggregate = new AnalystRatingAggregate
            {
                Symbol = symbol,
                Ratings = ratings
            };
            
            // Calculate aggregates
            aggregate.RecalculateAggregates();
            
            // Cache the result
            _aggregateCache[cacheKey] = (aggregate, DateTime.Now);
            
            return aggregate;
        }
        
        /// <summary>
        /// Gets a sentiment score based on analyst ratings for a symbol (-1.0 to 1.0)
        /// </summary>
        public async Task<double> GetRatingSentimentAsync(string symbol)
        {
            var aggregate = await GetAggregatedRatingsAsync(symbol);
            return aggregate.ConsensusScore;
        }
        
        /// <summary>
        /// Detects changes in analyst ratings since a specified date
        /// </summary>
        public async Task<List<AnalystRating>> GetRatingChangesAsync(string symbol, DateTime since)
        {
            var ratings = await GetRecentRatingsAsync(symbol, 50);
            
            // Filter for changes since the specified date
            var changes = ratings.Where(r => 
                r.RatingDate >= since && 
                (r.ChangeType == RatingChangeType.Upgrade || 
                 r.ChangeType == RatingChangeType.Downgrade ||
                 r.ChangeType == RatingChangeType.Initiation)).ToList();
            
            return changes;
        }

        /// <summary>
        /// Analyzes historical consensus trends for a symbol
        /// </summary>
        public async Task<AnalystRatingAggregate> AnalyzeConsensusHistoryAsync(string symbol, int days = 30)
        {
            try
            {
                // Get current aggregate
                var currentAggregate = await GetAggregatedRatingsAsync(symbol);
                
                // Get historical consensus data
                var historyData = await GetConsensusHistoryAsync(symbol, days);
                
                if (historyData.Count == 0)
                {
                    // No history available, return current aggregate
                    currentAggregate.ConsensusTrend = "No historical data";
                    return currentAggregate;
                }
                
                // Get oldest data point for comparison
                var oldestData = historyData.OrderBy(h => h.LastUpdated).FirstOrDefault();
                
                if (oldestData != null)
                {
                    // Store the previous consensus data for comparison
                    currentAggregate.PreviousConsensusScore = oldestData.ConsensusScore;
                    currentAggregate.PreviousConsensusRating = oldestData.ConsensusRating;
                    
                    // Calculate the consensus trend
                    double scoreDelta = currentAggregate.ConsensusScore - oldestData.ConsensusScore;
                    
                    if (Math.Abs(scoreDelta) < 0.1)
                        currentAggregate.ConsensusTrend = "Stable";
                    else if (scoreDelta > 0)
                        currentAggregate.ConsensusTrend = "Improving";
                    else
                        currentAggregate.ConsensusTrend = "Deteriorating";
                    
                    // Calculate ratings strength index based on historical data
                    int totalUpgrades = historyData.Sum(h => h.UpgradeCount);
                    int totalDowngrades = historyData.Sum(h => h.DowngradeCount);
                    
                    if (totalUpgrades + totalDowngrades > 0)
                    {
                        currentAggregate.RatingsStrengthIndex = ((double)totalUpgrades - totalDowngrades) / (totalUpgrades + totalDowngrades);
                    }
                }
                
                return currentAggregate;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error analyzing consensus history for {symbol}", ex.ToString());
                return await GetAggregatedRatingsAsync(symbol); // Return current aggregate on error
            }
        }
        
        /// <summary>
        /// Gets historical analyst consensus data for trend analysis
        /// </summary>
        public async Task<List<AnalystRatingAggregate>> GetConsensusHistoryAsync(string symbol, int days = 90)
        {
            // Check cache first
            string cacheKey = $"history_{symbol.ToLower()}_{days}";
            if (_consensusHistoryCache.TryGetValue(cacheKey, out var cachedData))
            {
                // If cache is still valid (not older than configured interval)
                var cacheAgeHours = (DateTime.Now - cachedData.Timestamp).TotalHours;
                int maxCacheAgeHours = _userSettings.RatingsCacheExpiryHours ?? 24; // Default 24 hours
                if (cacheAgeHours < maxCacheAgeHours)
                {
                    return cachedData.History;
                }
            }
            
            // Not in cache or cache expired, get from database
            var historyData = DatabaseMonolith.GetConsensusHistory(symbol, days);
            
            // Cache the results
            _consensusHistoryCache[cacheKey] = (historyData, DateTime.Now);
            
            return historyData;
        }

        /// <summary>
        /// Gets AI-powered analyst sentiment using ChatGPT/AI models for enhanced analysis
        /// </summary>
        public async Task<double> GetAnalystSentimentAsync(string symbol)
        {
            try
            {
                // First get traditional analyst rating sentiment as a baseline
                double ratingSentiment = await GetRatingSentimentAsync(symbol);
                
                // Get recent analyst ratings to provide context for AI analysis
                var recentRatings = await GetRecentRatingsAsync(symbol, 10);
                
                // Check if we have any API key for AI integration
                string aiApiKey = null;
                try
                {
                    aiApiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") 
                        ?? Environment.GetEnvironmentVariable("CHATGPT_API_KEY")
                        ?? Quantra.DAL.Utilities.Utilities.GetOpenAiApiKey();
                }
                catch (Exception)
                {
                    // API key not found in settings, will fall back to traditional sentiment
                }
                
                if (string.IsNullOrEmpty(aiApiKey) || aiApiKey == "YOUR_API_KEY")
                {
                    DatabaseMonolith.Log("Warning", $"No AI API key configured for enhanced analyst sentiment analysis of {symbol}. Using traditional rating sentiment.");
                    return ratingSentiment;
                }
                
                // Prepare data for AI analysis
                var analysisContext = PrepareAnalystContextForAI(symbol, recentRatings, ratingSentiment);
                
                // Call AI service for enhanced sentiment analysis
                double aiSentiment = await CallAIForAnalystSentiment(symbol, analysisContext, aiApiKey);
                
                // Return AI sentiment if available, otherwise fall back to rating sentiment
                return aiSentiment != 0 ? aiSentiment : ratingSentiment;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error getting AI analyst sentiment for {symbol}, falling back to rating sentiment", ex.ToString());
                // Fall back to traditional rating sentiment
                return await GetRatingSentimentAsync(symbol);
            }
        }
        
        /// <summary>
        /// Refreshes rating data for a symbol
        /// </summary>
        public async Task<bool> RefreshRatingsAsync(string symbol)
        {
            try
            {
                // Clear cached data for this symbol
                ClearCache(symbol);
                
                // Fetch fresh data
                var ratings = await GetRecentRatingsAsync(symbol);
                var aggregate = await GetAggregatedRatingsAsync(symbol);
                
                // Save data to database for historical tracking
                DatabaseMonolith.SaveAnalystRatings(symbol, ratings);
                DatabaseMonolith.SaveConsensusHistory(aggregate);
                
                // Check for significant changes and create alerts if needed
                await CheckForSignificantChanges(symbol, ratings);
                
                // Analyze trends based on historical data
                await AnalyzeConsensusHistoryAsync(symbol);
                
                return true;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to refresh analyst ratings for {symbol}", ex.ToString());
                return false;
            }
        }
        
        /// <summary>
        /// Clears cached rating data
        /// </summary>
        public void ClearCache(string symbol = null)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                // Clear all cache
                _ratingsCache.Clear();
                _aggregateCache.Clear();
            }
            else
            {
                // Clear specific symbol cache
                string ratingsCacheKey = $"ratings_{symbol.ToLower()}";
                string aggregateCacheKey = $"aggregate_{symbol.ToLower()}";
                
                _ratingsCache.TryRemove(ratingsCacheKey, out _);
                _aggregateCache.TryRemove(aggregateCacheKey, out _);
            }
        }
        
        /// <summary>
        /// Checks for significant changes in analyst ratings and creates alerts
        /// </summary>
        private async Task CheckForSignificantChanges(string symbol, List<AnalystRating> ratings)
        {
            if (ratings == null || ratings.Count == 0)
                return;
                
            // Look for recent changes (last 24 hours)
            var recentChanges = ratings.Where(r => 
                r.RatingDate >= DateTime.Now.AddHours(-24) && 
                (r.ChangeType == RatingChangeType.Upgrade || 
                 r.ChangeType == RatingChangeType.Downgrade ||
                 r.ChangeType == RatingChangeType.Initiation ||
                 r.ChangeType == RatingChangeType.PriceTargetChange))
                .ToList();
                
            // Group by analyst to aggregate multiple changes
            var analystChanges = recentChanges
                .GroupBy(r => r.AnalystName)
                .Select(g => g.OrderByDescending(r => r.RatingDate).First())
                .ToList();
                
            foreach (var change in analystChanges)
            {
                // Create alerts for significant rating changes
                string alertMessage = string.Empty;
                string condition = string.Empty;
                int priority = 2; // Medium by default
                AlertCategory category = AlertCategory.Standard;
                
                switch (change.ChangeType)
                {
                    case RatingChangeType.Upgrade:
                        alertMessage = $"{change.AnalystName} upgraded {symbol} from {change.PreviousRating} to {change.Rating}";
                        condition = $"Rating Upgrade: {change.PreviousRating} → {change.Rating}";
                        category = AlertCategory.Opportunity;
                        
                        // Higher priority for stronger upgrades
                        if (change.SentimentScore > 0.5)
                            priority = 1; // High priority
                        break;
                        
                    case RatingChangeType.Downgrade:
                        alertMessage = $"{change.AnalystName} downgraded {symbol} from {change.PreviousRating} to {change.Rating}";
                        condition = $"Rating Downgrade: {change.PreviousRating} → {change.Rating}";
                        
                        // Higher priority for stronger downgrades
                        if (change.SentimentScore < -0.5)
                            priority = 1; // High priority
                        break;
                        
                    case RatingChangeType.Initiation:
                        alertMessage = $"{change.AnalystName} initiated coverage on {symbol} with {change.Rating}";
                        condition = $"New Coverage: {change.Rating}";
                        
                        // Higher priority for strong initial ratings
                        if (Math.Abs(change.SentimentScore) > 0.5)
                            priority = 1;
                        break;
                        
                    case RatingChangeType.PriceTargetChange:
                        if (change.PriceTarget > 0 && change.PreviousPriceTarget > 0)
                        {
                            double pctChange = (change.PriceTarget - change.PreviousPriceTarget) / change.PreviousPriceTarget * 100;
                            
                            // Only alert if change is significant (>5%)
                            if (Math.Abs(pctChange) >= 5)
                            {
                                string direction = pctChange > 0 ? "raised" : "lowered";
                                alertMessage = $"{change.AnalystName} {direction} {symbol} price target from ${change.PreviousPriceTarget:F2} to ${change.PriceTarget:F2} ({(pctChange > 0 ? "+" : "")}{pctChange:F1}%)";
                                condition = $"Price Target Change: {(pctChange > 0 ? "+" : "")}{pctChange:F1}%";
                                
                                if (pctChange > 0)
                                    category = AlertCategory.Opportunity;
                                    
                                // Higher priority for larger changes
                                if (Math.Abs(pctChange) > 15)
                                    priority = 1;
                            }
                            else
                            {
                                // Skip small price target changes
                                continue;
                            }
                        }
                        break;
                }
                
                if (!string.IsNullOrEmpty(alertMessage))
                {
                    // Add price target info if available and not already a price target change
                    if (change.ChangeType != RatingChangeType.PriceTargetChange && change.PriceTarget > 0)
                    {
                        string priceTargetInfo = $"PT: ${change.PriceTarget:F2}";
                        
                        if (change.PreviousPriceTarget > 0)
                        {
                            double pctChange = (change.PriceTarget - change.PreviousPriceTarget) / change.PreviousPriceTarget * 100;
                            string direction = pctChange >= 0 ? "↑" : "↓";
                            priceTargetInfo += $" ({direction}{Math.Abs(pctChange):F1}%)";
                        }
                        
                        alertMessage += $" with {priceTargetInfo}";
                    }
                    
                    // Calculate importance based on analyst firm's historical impact
                    string analystImportance = "Standard";
                    string additionalNotes = string.Empty;
                    
                    var analystFirmStats = await GetAnalystFirmStatistics(change.AnalystName);
                    if (analystFirmStats != null)
                    {
                        analystImportance = analystFirmStats.AccuracyRating >= 0.7 ? "High Impact" : 
                                          analystFirmStats.AccuracyRating >= 0.5 ? "Medium Impact" : "Low Impact";
                         
                        additionalNotes = $"Analyst: {change.AnalystName} ({analystImportance})\n" +
                                        $"Historical Accuracy: {analystFirmStats.AccuracyRating:P1}\n" +
                                        $"Date: {change.RatingDate}";
                                        
                        // Adjust priority based on analyst importance
                        if (analystFirmStats.AccuracyRating >= 0.7 && priority > 1)
                            priority = priority - 1;
                    }
                    else
                    {
                        additionalNotes = $"Analyst: {change.AnalystName}\nDate: {change.RatingDate}";
                    }
                    
                    // Create the alert
                    var alert = new AlertModel
                    {
                        Name = alertMessage,
                        Symbol = symbol,
                        Condition = condition,
                        AlertType = "Analyst Rating",
                        IsActive = true,
                        Priority = priority,
                        Category = category,
                        CreatedDate = DateTime.Now,
                        Notes = additionalNotes
                    };
                    
                    // Publish via DI publisher if available; otherwise fallback to AlertManager
                    if (_alertPublisher != null)
                    {
                        _alertPublisher.EmitGlobalAlert(alert);
                    }
                    else
                    {
                        Alerting.EmitGlobalAlert(alert);
                    }
                }
            }
            
            // Check for trend of multiple analysts making similar changes (consensus shift)
            int upgradeCount = recentChanges.Count(r => r.ChangeType == RatingChangeType.Upgrade);
            int downgradeCount = recentChanges.Count(r => r.ChangeType == RatingChangeType.Downgrade);
            
            if (upgradeCount >= 2 && upgradeCount > downgradeCount * 2)
            {
                var alert = new AlertModel
                {
                    Name = $"Multiple analyst upgrades for {symbol} ({upgradeCount} upgrades in 24h)",
                    Symbol = symbol,
                    Condition = $"Multiple Upgrades: {upgradeCount}",
                    AlertType = "Consensus Shift",
                    IsActive = true,
                    Priority = 1,
                    Category = AlertCategory.Opportunity,
                    CreatedDate = DateTime.Now,
                    Notes = $"Significant positive shift with {upgradeCount} upgrades vs {downgradeCount} downgrades in the last 24 hours"
                };
                if (_alertPublisher != null) _alertPublisher.EmitGlobalAlert(alert); else Alerting.EmitGlobalAlert(alert);
            }
            else if (downgradeCount >= 2 && downgradeCount > upgradeCount * 2)
            {
                var alert = new AlertModel
                {
                    Name = $"Multiple analyst downgrades for {symbol} ({downgradeCount} downgrades in 24h)",
                    Symbol = symbol,
                    Condition = $"Multiple Downgrades: {downgradeCount}",
                    AlertType = "Consensus Shift",
                    IsActive = true,
                    Priority = 1,
                    Category = AlertCategory.Standard,
                    CreatedDate = DateTime.Now,
                    Notes = $"Significant negative shift with {downgradeCount} downgrades vs {upgradeCount} upgrades in the last 24 hours"
                };
                if (_alertPublisher != null) _alertPublisher.EmitGlobalAlert(alert); else Alerting.EmitGlobalAlert(alert);
            }
            
            // Check for consensus change
            await CheckConsensusChange(symbol);
        }

        /// <summary>
        /// Gets statistics about an analyst firm's historical accuracy
        /// </summary>
        private async Task<AnalystFirmStatistics> GetAnalystFirmStatistics(string analystName)
        {
            // In a real implementation, this would query a database of historical analyst predictions
            // and their accuracy. For the demo, we'll return synthetic data.
            
            // This would typically be sourced from a database of historical performance
            var random = new Random(analystName.GetHashCode());
            
            return new AnalystFirmStatistics
            {
                AnalystName = analystName,
                AccuracyRating = 0.4 + random.NextDouble() * 0.5, // 40% - 90% accuracy
                AvgPriceTargetAccuracy = 0.5 + random.NextDouble() * 0.4, // 50% - 90% accuracy
                TotalRatings = 10 + random.Next(90),
                UpgradeSuccessRate = 0.4 + random.NextDouble() * 0.5,
                DowngradeSuccessRate = 0.4 + random.NextDouble() * 0.5
            };
        }
        
        /// <summary>
        /// Checks for changes in the overall analyst consensus and creates alerts
        /// </summary>
        private async Task CheckConsensusChange(string symbol)
        {
            try
            {
                var currentAggregate = await GetAggregatedRatingsAsync(symbol);
                var historyAnalysis = await AnalyzeConsensusHistoryAsync(symbol);
                var previousConsensus = GetPreviousConsensus(symbol);
                
                if (previousConsensus != null)
                {
                    if (previousConsensus.ConsensusRating != currentAggregate.ConsensusRating)
                    {
                        string alertMessage = $"Analyst consensus for {symbol} changed from {previousConsensus.ConsensusRating} to {currentAggregate.ConsensusRating}";
                        
                        AlertCategory category = AlertCategory.Standard;
                        int priority = 2;
                        
                        if (previousConsensus.ConsensusRating == "Hold" && currentAggregate.ConsensusRating == "Buy" ||
                            previousConsensus.ConsensusRating == "Sell" && currentAggregate.ConsensusRating == "Hold")
                        {
                            category = AlertCategory.Opportunity;
                            priority = 1;
                        }
                        else if (previousConsensus.ConsensusRating == "Buy" && currentAggregate.ConsensusRating == "Hold" ||
                                 previousConsensus.ConsensusRating == "Hold" && currentAggregate.ConsensusRating == "Sell")
                        {
                            priority = 1;
                        }
                        
                        var alert = new AlertModel
                        {
                            Name = alertMessage,
                            Symbol = symbol,
                            Condition = $"Consensus Change: {previousConsensus.ConsensusRating} → {currentAggregate.ConsensusRating}",
                            AlertType = "Consensus Rating",
                            IsActive = true,
                            Priority = priority,
                            Category = category,
                            CreatedDate = DateTime.Now,
                            Notes = $"Current breakdown: {currentAggregate.BuyCount} Buy, {currentAggregate.HoldCount} Hold, {currentAggregate.SellCount} Sell\nTrend: {currentAggregate.ConsensusTrend} (Upgrades: {currentAggregate.UpgradeCount}, Downgrades: {currentAggregate.DowngradeCount})"
                        };
                        if (_alertPublisher != null) _alertPublisher.EmitGlobalAlert(alert); else Alerting.EmitGlobalAlert(alert);
                    }
                    else if (Math.Abs(previousConsensus.ConsensusScore - currentAggregate.ConsensusScore) > 0.2)
                    {
                        string direction = currentAggregate.ConsensusScore > previousConsensus.ConsensusScore ? "strengthened" : "weakened";
                        string alertMessage = $"Analyst consensus for {symbol} has significantly {direction} while remaining at {currentAggregate.ConsensusRating}";
                        
                        AlertCategory category = direction == "strengthened" ? AlertCategory.Opportunity : AlertCategory.Standard;
                        int priority = 2;
                        
                        var alert = new AlertModel
                        {
                            Name = alertMessage,
                            Symbol = symbol,
                            Condition = $"Consensus Strength Change: {previousConsensus.ConsensusScore:F2} → {currentAggregate.ConsensusScore:F2}",
                            AlertType = "Consensus Strength",
                            IsActive = true,
                            Priority = priority,
                            Category = category,
                            CreatedDate = DateTime.Now,
                            Notes = $"Current breakdown: {currentAggregate.BuyCount} Buy, {currentAggregate.HoldCount} Hold, {currentAggregate.SellCount} Sell\nTrend: {currentAggregate.ConsensusTrend}"
                        };
                        if (_alertPublisher != null) _alertPublisher.EmitGlobalAlert(alert); else Alerting.EmitGlobalAlert(alert);
                    }
                    else if (previousConsensus.AveragePriceTarget > 0 && 
                             Math.Abs(currentAggregate.AveragePriceTarget - previousConsensus.AveragePriceTarget) / previousConsensus.AveragePriceTarget > 0.05)
                    {
                        double pctChange = (currentAggregate.AveragePriceTarget - previousConsensus.AveragePriceTarget) / previousConsensus.AveragePriceTarget * 100;
                        string direction = pctChange > 0 ? "increased" : "decreased";
                        
                        string alertMessage = $"Average price target for {symbol} has {direction} by {Math.Abs(pctChange):F1}%";
                        
                        AlertCategory category = pctChange > 0 ? AlertCategory.Opportunity : AlertCategory.Standard;
                        
                        var alert = new AlertModel
                        {
                            Name = alertMessage,
                            Symbol = symbol,
                            Condition = $"Price Target Change: ${previousConsensus.AveragePriceTarget:F2} → ${currentAggregate.AveragePriceTarget:F2}",
                            AlertType = "Price Target",
                            IsActive = true,
                            Priority = 2,
                            Category = category,
                            CreatedDate = DateTime.Now,
                            Notes = $"Consensus rating: {currentAggregate.ConsensusRating}\nRange: ${currentAggregate.LowestPriceTarget:F2} - ${currentAggregate.HighestPriceTarget:F2}"
                        };
                        if (_alertPublisher != null) _alertPublisher.EmitGlobalAlert(alert); else Alerting.EmitGlobalAlert(alert);
                    }
                }
                
                StorePreviousConsensus(symbol, currentAggregate);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error checking consensus change for {symbol}", ex.ToString());
            }
        }
        
        // For demonstration - this would typically use a database
        private static readonly ConcurrentDictionary<string, AnalystRatingAggregate> _previousConsensusData = 
            new ConcurrentDictionary<string, AnalystRatingAggregate>();
            
        private AnalystRatingAggregate GetPreviousConsensus(string symbol)
        {
            string key = symbol.ToLower();
            if (_previousConsensusData.TryGetValue(key, out var aggregate))
                return aggregate;
                
            return null;
        }
        
        private void StorePreviousConsensus(string symbol, AnalystRatingAggregate aggregate)
        {
            string key = symbol.ToLower();
            _previousConsensusData[key] = aggregate;
        }
        
        /// <summary>
        /// Prepares analyst context data for AI analysis
        /// </summary>
        private string PrepareAnalystContextForAI(string symbol, List<AnalystRating> recentRatings, double ratingSentiment)
        {
            var context = new StringBuilder();
            context.AppendLine($"Stock Symbol: {symbol}");
            context.AppendLine($"Traditional Rating Sentiment: {ratingSentiment:F2}");
            context.AppendLine($"Recent Analyst Ratings ({recentRatings.Count} ratings):");
            
            foreach (var rating in recentRatings.Take(5)) // Limit to most recent 5 for context size
            {
                context.AppendLine($"- {rating.AnalystName}: {rating.Rating}");
                if (rating.PriceTarget > 0)
                    context.AppendLine($"  Price Target: ${rating.PriceTarget:F2}");
                if (!string.IsNullOrEmpty(rating.PreviousRating))
                    context.AppendLine($"  Previous: {rating.PreviousRating} ({rating.ChangeType})");
                context.AppendLine($"  Date: {rating.RatingDate:yyyy-MM-dd}");
            }
            
            return context.ToString();
        }
        
        /// <summary>
        /// Calls AI service for enhanced analyst sentiment analysis
        /// </summary>
        private async Task<double> CallAIForAnalystSentiment(string symbol, string context, string apiKey)
        {
            try
            {
                // Prepare the AI prompt for sentiment analysis
                string prompt = $@"
You are an expert financial analyst. Based on the following analyst rating data for {symbol}, provide a sentiment score between -1.0 (very negative) and 1.0 (very positive).

Consider factors like:
- Recent rating changes (upgrades vs downgrades)
- Price target adjustments
- Analyst reputation and track record
- Consensus trends
- Market context

{context}

Respond with only a decimal number between -1.0 and 1.0 representing the overall analyst sentiment. Do not include any explanation, just the number.";

                using (var httpClient = new HttpClient())
                {
                    httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
                    httpClient.Timeout = TimeSpan.FromSeconds(30);
                    
                    var requestBody = new
                    {
                        model = "gpt-3.5-turbo",
                        messages = new[]
                        {
                            new { role = "system", content = "You are a financial analysis assistant that returns numerical sentiment scores." },
                            new { role = "user", content = prompt }
                        },
                        max_tokens = 10,
                        temperature = 0.3 // Lower temperature for more consistent results
                    };
                    
                    string json = JsonSerializer.Serialize(requestBody);
                    var content = new StringContent(json, Encoding.UTF8, "application/json");
                    
                    var response = await httpClient.PostAsync("https://api.openai.com/v1/chat/completions", content);
                    
                    if (response.IsSuccessStatusCode)
                    {
                        string responseJson = await response.Content.ReadAsStringAsync();
                        using (JsonDocument doc = JsonDocument.Parse(responseJson))
                        {
                            var choices = doc.RootElement.GetProperty("choices");
                            if (choices.GetArrayLength() > 0)
                            {
                                var messageContent = choices[0].GetProperty("message").GetProperty("content").GetString();
                                
                                // Parse the AI response to extract sentiment score
                                if (double.TryParse(messageContent?.Trim(), out double sentiment))
                                {
                                    // Ensure the sentiment is within valid range
                                    sentiment = Math.Max(-1.0, Math.Min(1.0, sentiment));
                                    DatabaseMonolith.Log("Info", $"AI analyst sentiment for {symbol}: {sentiment:F2}");
                                    return sentiment;
                                }
                            }
                        }
                    }
                    else
                    {
                        DatabaseMonolith.Log("Warning", $"AI API call failed for {symbol}: {response.StatusCode} - {await response.Content.ReadAsStringAsync()}");
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error calling AI service for analyst sentiment of {symbol}", ex.ToString());
            }
            
            return 0; // Return 0 if AI call fails
        }
        
        /// <summary>
        /// Generates synthetic rating data for demonstration purposes
        /// </summary>
        private List<AnalystRating> GenerateSyntheticRatingData(string symbol)
        {
            // This method creates synthetic data for demonstration
            // In production, this would be replaced with real API calls
            
            var random = new Random(symbol.GetHashCode()); // Use symbol as seed for consistency
            
            var analystFirms = new[] 
            {
                "Morgan Stanley", "Goldman Sachs", "JP Morgan", "Bank of America", 
                "Credit Suisse", "Barclays", "UBS", "Wells Fargo", "Citi", "Deutsche Bank"
            };
            
            var ratingTypes = new[] 
            {
                "Buy", "Hold", "Sell", "Overweight", "Underweight", "Neutral", 
                "Outperform", "Market Perform", "Strong Buy"
            };
            
            var ratings = new List<AnalystRating>();
            
            // Generate 15 synthetic ratings
            for (int i = 0; i < 15; i++)
            {
                // Random firm
                string firm = analystFirms[random.Next(analystFirms.Length)];
                
                // Rating
                string currentRating = ratingTypes[random.Next(ratingTypes.Length)]
                    .Replace("Buy", "Strong Buy") // Promote some Buys to Strong Buy for variety
                    .Replace("Sell", "Strong Sell"); // Promote some Sells to Strong Sell
                
                // Random date in last 90 days
                DateTime ratingDate = DateTime.Now.AddDays(-random.Next(90));
                
                // Price target - reasonable range based on symbol
                double basePrice = 50.0 + symbol.Length * 10; // Just a synthetic formula
                double priceTarget = basePrice * (0.8 + random.NextDouble() * 0.4); // +/- 20%
                
                // Previous price target and rating
                string previousRating = null;
                double previousPriceTarget = 0;
                RatingChangeType changeType;
                
                int changeTypeRandom = random.Next(100);
                if (changeTypeRandom < 10) // 10% chance of initiation
                {
                    changeType = RatingChangeType.Initiation;
                }
                else if (changeTypeRandom < 40) // 30% chance of upgrade
                {
                    changeType = RatingChangeType.Upgrade;
                    int prevRatingIndex = Array.IndexOf(ratingTypes, currentRating) + 1 + random.Next(3);
                    prevRatingIndex = Math.Min(ratingTypes.Length - 1, Math.Max(0, prevRatingIndex));
                    previousRating = ratingTypes[prevRatingIndex];
                    previousPriceTarget = priceTarget * (0.85 + random.NextDouble() * 0.1); // Slightly lower
                }
                else if (changeTypeRandom < 70) // 30% chance of downgrade
                {
                    changeType = RatingChangeType.Downgrade;
                    int prevRatingIndex = Array.IndexOf(ratingTypes, currentRating) - 1 - random.Next(3);
                    prevRatingIndex = Math.Min(ratingTypes.Length - 1, Math.Max(0, prevRatingIndex));
                    previousRating = ratingTypes[prevRatingIndex];
                    previousPriceTarget = priceTarget * (1.05 + random.NextDouble() * 0.1); // Slightly higher
                }
                else // 30% chance of reiteration
                {
                    changeType = RatingChangeType.Reiteration;
                    previousRating = currentRating;
                    previousPriceTarget = priceTarget * (0.95 + random.NextDouble() * 0.1); // Small change
                }
                
                var rating = new AnalystRating
                {
                    Id = i + 1,
                    Symbol = symbol,
                    AnalystName = firm,
                    Rating = currentRating,
                    PreviousRating = previousRating,
                    PriceTarget = priceTarget,
                    PreviousPriceTarget = previousPriceTarget,
                    RatingDate = ratingDate,
                    ChangeType = changeType
                };
                
                ratings.Add(rating);
            }
            
            // Add one very recent rating to trigger alerts in demo
            if (ratings.Count > 0)
            {
                var lastRating = ratings[0].Clone();
                lastRating.RatingDate = DateTime.Now.AddHours(-2); // 2 hours ago
                
                // Make it an upgrade or downgrade
                bool isUpgrade = random.Next(2) == 0;
                lastRating.ChangeType = isUpgrade ? RatingChangeType.Upgrade : RatingChangeType.Downgrade;
                lastRating.PreviousRating = lastRating.Rating;
                
                if (isUpgrade)
                {
                    lastRating.Rating = "Buy";
                    lastRating.PriceTarget *= 1.1;
                }
                else
                {
                    lastRating.Rating = "Sell";
                    lastRating.PriceTarget *= 0.9;
                }
                
                ratings.Add(lastRating);
            }
            
            return ratings.OrderByDescending(r => r.RatingDate).ToList();
        }
    }
    
    // Extension method to clone AnalystRating objects
    public static class AnalystRatingExtensions
    {
        public static AnalystRating Clone(this AnalystRating original)
        {
            return new AnalystRating
            {
                Id = original.Id,
                Symbol = original.Symbol,
                AnalystName = original.AnalystName,
                Rating = original.Rating,
                PreviousRating = original.PreviousRating,
                PriceTarget = original.PriceTarget,
                PreviousPriceTarget = original.PreviousPriceTarget,
                RatingDate = original.RatingDate,
                ChangeType = original.ChangeType
            };
        }
    }
}