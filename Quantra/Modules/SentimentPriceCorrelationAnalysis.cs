using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;
using Quantra.Repositories;

namespace Quantra.Modules
{
    /// <summary>
    /// Provides analysis of correlations between market sentiment shifts and price movements
    /// to identify predictive patterns and enhance trading signal reliability.
    /// </summary>
    public class SentimentPriceCorrelationAnalysis
    {
        private readonly FinancialNewsSentimentService _financialNewsSentimentService;
        private readonly ISocialMediaSentimentService _twitterSentimentService;
        private readonly IAnalystRatingService _analystRatingService;
        private readonly IInsiderTradingService _insiderTradingService;
        private readonly SectorSentimentAnalysisService _sectorSentimentService;
        private readonly PredictionAnalysisRepository _predictionAnalysisRepository;
        private readonly SectorMomentumService _sectorMomentumService;
        private readonly UserSettings _userSettings;
        private readonly UserSettingsService _userSettingsService;
        private readonly LoggingService _loggingService;

        public SentimentPriceCorrelationAnalysis(UserSettings userSettings, UserSettingsService userSettingsService, LoggingService loggingService)
        {
            _financialNewsSentimentService = new FinancialNewsSentimentService(userSettings);
            _twitterSentimentService = new TwitterSentimentService();
            _analystRatingService = ServiceLocator.Resolve<IAnalystRatingService>();
            _insiderTradingService = ServiceLocator.Resolve<IInsiderTradingService>();
            _sectorSentimentService = new SectorSentimentAnalysisService(userSettings);
            _predictionAnalysisRepository = new PredictionAnalysisRepository();
            _loggingService = loggingService;
            _sectorMomentumService = new SectorMomentumService(userSettingsService, loggingService);
            _userSettings = userSettings ?? new UserSettings();
            _userSettingsService = userSettingsService;
            _loggingService = loggingService;
        }

        /// <summary>
        /// Analyzes correlation between sentiment changes and price movements for a specific symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="lookbackDays">Number of days to analyze</param>
        /// <param name="sentimentSources">List of sentiment sources to include (news, social, analysts, insiders)</param>
        /// <returns>Correlation analysis results</returns>
        public async Task<SentimentPriceCorrelationResult> AnalyzeSentimentPriceCorrelation(
            string symbol,
            int lookbackDays = 30,
            List<string> sentimentSources = null)
        {
            // Default to all sources if none specified
            sentimentSources ??= new List<string> {
                "News", "Twitter", "Reddit", "AnalystRatings", "InsiderTrading"
            };

            // Get historical price data
            var priceHistory = _predictionAnalysisRepository.GetHistoricalPrices(symbol)
                .Where(p => p.Date >= DateTime.Now.AddDays(-lookbackDays))
                .OrderBy(p => p.Date)
                .ToList();

            if (priceHistory.Count < 5)
            {
                // Not enough data for analysis
                return new SentimentPriceCorrelationResult
                {
                    Symbol = symbol,
                    OverallCorrelation = 0,
                    SourceCorrelations = new Dictionary<string, double>(),
                    LeadLagRelationship = 0,
                    PredictiveAccuracy = 0,
                    SentimentShiftEvents = new List<SentimentShiftEvent>()
                };
            }

            // Calculate daily price changes (percentage)
            var priceChanges = new List<double>();
            for (int i = 1; i < priceHistory.Count; i++)
            {
                double previousClose = priceHistory[i - 1].Close;
                double currentClose = priceHistory[i].Close;
                double percentChange = (currentClose - previousClose) / previousClose * 100.0;
                priceChanges.Add(percentChange);
            }

            // Get sentiment data for the same period
            var sentimentHistory = await GetHistoricalSentimentData(symbol, lookbackDays, sentimentSources);

            // Align dates (sentiment and price data may have different collection points)
            var alignedData = AlignSentimentAndPriceData(sentimentHistory, priceHistory);

            // Calculate correlations
            double overallCorrelation = 0;
            var sourceCorrelations = new Dictionary<string, double>();

            // Calculate correlation for each source
            foreach (var source in sentimentSources.Where(s => alignedData.SentimentBySource.ContainsKey(s)))
            {
                var sentimentValues = alignedData.SentimentBySource[source];
                if (sentimentValues.Count > 1)
                {
                    double correlation = CalculatePearsonCorrelation(sentimentValues, alignedData.PriceChanges);
                    sourceCorrelations[source] = correlation;
                }
            }

            // Overall sentiment (combined weighted average)
            if (alignedData.CombinedSentiment.Count > 1)
            {
                overallCorrelation = CalculatePearsonCorrelation(
                    alignedData.CombinedSentiment,
                    alignedData.PriceChanges);
            }

            // Calculate lead/lag relationship
            double leadLagDays = CalculateLeadLagRelationship(
                alignedData.CombinedSentiment,
                alignedData.PriceChanges);

            // Identify significant sentiment shift events
            var sentimentShiftEvents = IdentifySentimentShiftEvents(
                alignedData.SentimentBySource,
                alignedData.Dates,
                alignedData.PriceChanges);

            // Calculate predictive accuracy
            double predictiveAccuracy = CalculatePredictiveAccuracy(sentimentShiftEvents);

            return new SentimentPriceCorrelationResult
            {
                Symbol = symbol,
                OverallCorrelation = overallCorrelation,
                SourceCorrelations = sourceCorrelations,
                LeadLagRelationship = leadLagDays,
                PredictiveAccuracy = predictiveAccuracy,
                SentimentShiftEvents = sentimentShiftEvents,
                AlignedData = alignedData
            };
        }

        /// <summary>
        /// Gets visualization data for sentiment-price correlation
        /// </summary>
        public async Task<SentimentPriceVisualData> GetVisualizationData(
            string symbol,
            int lookbackDays = 30,
            List<string> sentimentSources = null)
        {
            var correlation = await AnalyzeSentimentPriceCorrelation(symbol, lookbackDays, sentimentSources);

            return new SentimentPriceVisualData
            {
                Symbol = symbol,
                Dates = correlation.AlignedData.Dates,
                Prices = correlation.AlignedData.Prices,
                PriceChanges = correlation.AlignedData.PriceChanges,
                SentimentBySource = correlation.AlignedData.SentimentBySource,
                CombinedSentiment = correlation.AlignedData.CombinedSentiment,
                OverallCorrelation = correlation.OverallCorrelation,
                SourceCorrelations = correlation.SourceCorrelations,
                LeadLagRelationship = correlation.LeadLagRelationship,
                PredictiveAccuracy = correlation.PredictiveAccuracy,
                SentimentShiftEvents = correlation.SentimentShiftEvents
            };
        }

        /// <summary>
        /// Analyzes correlation between sector sentiment and sector performance
        /// </summary>
        /// <param name="sector">Market sector to analyze</param>
        /// <param name="lookbackDays">Number of days to analyze</param>
        /// <returns>Correlation analysis results for the sector</returns>
        public async Task<SectorSentimentCorrelationResult> AnalyzeSectorSentimentCorrelation(
            string sector,
            int lookbackDays = 30)
        {
            // Get sector sentiment trend
            var sectorSentimentTrend = await _sectorSentimentService.GetSectorSentimentTrendAsync(sector, lookbackDays);

            // Get sector performance data (using SectorMomentumService to get mock performance)
            var timeframe = lookbackDays <= 7 ? "1w" : lookbackDays <= 30 ? "1m" : "3m";
            var momentumData = _sectorMomentumService.GetSectorMomentumData(timeframe);

            // If sector not found, return empty result
            if (!momentumData.ContainsKey(sector))
            {
                return new SectorSentimentCorrelationResult
                {
                    Sector = sector,
                    OverallCorrelation = 0,
                    SectorSentiment = sectorSentimentTrend,
                    SectorPerformance = new List<(DateTime Date, double Value)>(),
                    LeadLagRelationship = 0,
                    SentimentShiftEvents = new List<SentimentShiftEvent>()
                };
            }

            // Generate historical sector performance based on current momentum data
            // In a real implementation, this would come from historical data
            var sectorPerformance = GenerateHistoricalSectorPerformance(
                sector,
                momentumData[sector],
                sectorSentimentTrend.Select(s => s.Date).ToList());

            // Align dates and calculate correlation
            var alignedData = AlignSectorData(sectorSentimentTrend, sectorPerformance);

            // Calculate overall correlation
            double overallCorrelation = 0;
            if (alignedData.SentimentValues.Count > 1 && alignedData.PerformanceValues.Count > 1)
            {
                overallCorrelation = CalculatePearsonCorrelation(
                    alignedData.SentimentValues,
                    alignedData.PerformanceValues);
            }

            // Calculate lead/lag relationship
            double leadLagDays = CalculateLeadLagRelationship(
                alignedData.SentimentValues,
                alignedData.PerformanceValues);

            // Identify significant sentiment shift events
            var sentimentShiftEvents = IdentifySectorSentimentShiftEvents(
                alignedData.Dates,
                alignedData.SentimentValues,
                alignedData.PerformanceValues);

            return new SectorSentimentCorrelationResult
            {
                Sector = sector,
                OverallCorrelation = overallCorrelation,
                SectorSentiment = sectorSentimentTrend,
                SectorPerformance = sectorPerformance,
                LeadLagRelationship = leadLagDays,
                SentimentShiftEvents = sentimentShiftEvents,
                AlignedDates = alignedData.Dates
            };
        }

        /// <summary>
        /// Gets sector sentiment correlation visualization data for all or specified sectors
        /// </summary>
        /// <param name="sectors">List of sectors to include (null for all)</param>
        /// <param name="lookbackDays">Number of days to analyze</param>
        /// <returns>Dictionary mapping sectors to their correlation results</returns>
        public async Task<Dictionary<string, SectorSentimentCorrelationResult>> GetAllSectorCorrelationsAsync(
            List<string> sectors = null,
            int lookbackDays = 30)
        {
            // If no sectors specified, use all standard sectors
            sectors ??= new List<string> {
                "Technology", "Financial", "Healthcare", "Energy", "Industrial",
                "Materials", "Consumer Discretionary", "Consumer Staples",
                "Utilities", "Real Estate", "Communication"
            };

            var results = new Dictionary<string, SectorSentimentCorrelationResult>();

            foreach (var sector in sectors)
            {
                try
                {
                    var result = await AnalyzeSectorSentimentCorrelation(sector, lookbackDays);
                    results[sector] = result;
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Warning", $"Error analyzing sector correlation for {sector}", ex.ToString());
                }
            }

            return results;
        }

        #region Helper Methods

        /// <summary>
        /// Retrieves historical sentiment data from various sources
        /// </summary>
        private async Task<Dictionary<string, List<(DateTime Date, double Value)>>> GetHistoricalSentimentData(
            string symbol,
            int lookbackDays,
            List<string> sentimentSources)
        {
            var result = new Dictionary<string, List<(DateTime Date, double Value)>>();

            // This would typically involve retrieving historical sentiment data from databases,
            // but for this implementation we'll use a simplified approach to simulate historical data

            // In a real implementation, we would:
            // 1. Query a database table that stores daily sentiment scores
            // 2. Or use APIs that provide historical sentiment data

            DateTime startDate = DateTime.Now.AddDays(-lookbackDays);

            // Get financial news sentiment
            if (sentimentSources.Contains("News"))
            {
                var newsHistory = new List<(DateTime Date, double Value)>();

                // For now, we'll make an approximation by getting current sentiment from different sources
                // and creating synthetic history using random variations
                double currentSentiment = await _financialNewsSentimentService.GetSymbolSentimentAsync(symbol);

                // Generate synthetic historical data based on current sentiment
                var random = new Random();
                for (int i = 0; i < lookbackDays; i++)
                {
                    double variation = random.NextDouble() * 0.4 - 0.2; // -0.2 to +0.2 variation
                    double historicalSentiment = Math.Max(-1.0, Math.Min(1.0, currentSentiment + variation));
                    DateTime historicalDate = startDate.AddDays(i);
                    newsHistory.Add((historicalDate, historicalSentiment));
                }

                result["News"] = newsHistory;
            }

            // Similar approach for other sentiment sources
            if (sentimentSources.Contains("Twitter"))
            {
                var twitterHistory = new List<(DateTime Date, double Value)>();
                double currentTwitterSentiment = await _twitterSentimentService.GetSymbolSentimentAsync(symbol);

                var random = new Random(1); // Different seed
                for (int i = 0; i < lookbackDays; i++)
                {
                    double variation = random.NextDouble() * 0.5 - 0.25; // -0.25 to +0.25 variation
                    double historicalSentiment = Math.Max(-1.0, Math.Min(1.0, currentTwitterSentiment + variation));
                    DateTime historicalDate = startDate.AddDays(i);
                    twitterHistory.Add((historicalDate, historicalSentiment));
                }

                result["Twitter"] = twitterHistory;
            }

            if (sentimentSources.Contains("AnalystRatings"))
            {
                var analystHistory = new List<(DateTime Date, double Value)>();
                double currentAnalystSentiment = await _analystRatingService.GetAnalystSentimentAsync(symbol);

                var random = new Random(3); // Different seed
                for (int i = 0; i < lookbackDays; i++)
                {
                    double variation = random.NextDouble() * 0.2 - 0.1; // -0.1 to +0.1 variation (more stable)
                    double historicalSentiment = Math.Max(-1.0, Math.Min(1.0, currentAnalystSentiment + variation));
                    DateTime historicalDate = startDate.AddDays(i);
                    analystHistory.Add((historicalDate, historicalSentiment));
                }

                result["AnalystRatings"] = analystHistory;
            }

            if (sentimentSources.Contains("InsiderTrading"))
            {
                var insiderHistory = new List<(DateTime Date, double Value)>();
                double currentInsiderSentiment = await _insiderTradingService.GetInsiderSentimentAsync(symbol);

                var random = new Random(4); // Different seed
                for (int i = 0; i < lookbackDays; i++)
                {
                    // Insider sentiment tends to be more discrete/step-function like
                    if (i % 7 == 0) // Occasional change
                    {
                        double variation = random.NextDouble() * 0.8 - 0.4; // -0.4 to +0.4 variation
                        currentInsiderSentiment = Math.Max(-1.0, Math.Min(1.0, currentInsiderSentiment + variation));
                    }

                    DateTime historicalDate = startDate.AddDays(i);
                    insiderHistory.Add((historicalDate, currentInsiderSentiment));
                }

                result["InsiderTrading"] = insiderHistory;
            }

            return result;
        }

        /// <summary>
        /// Aligns sentiment and price data to have the same dates
        /// </summary>
        private SentimentPriceAlignedData AlignSentimentAndPriceData(
            Dictionary<string, List<(DateTime Date, double Value)>> sentimentHistory,
            List<HistoricalPrice> priceHistory)
        {
            var result = new SentimentPriceAlignedData
            {
                Dates = new List<DateTime>(),
                Prices = new List<double>(),
                PriceChanges = new List<double>(),
                SentimentBySource = new Dictionary<string, List<double>>(),
                CombinedSentiment = new List<double>()
            };

            // Initialize sentiment source lists
            foreach (var source in sentimentHistory.Keys)
            {
                result.SentimentBySource[source] = new List<double>();
            }

            // Iterate through price history and find matching sentiment data
            for (int i = 1; i < priceHistory.Count; i++) // Start at 1 to calculate price changes
            {
                DateTime date = priceHistory[i].Date;
                double price = priceHistory[i].Close;
                double previousPrice = priceHistory[i - 1].Close;
                double priceChange = (price - previousPrice) / previousPrice * 100.0;

                // Try to find sentiment data for this date
                bool hasSentimentData = false;
                double combinedSentiment = 0;
                int sourcesWithData = 0;

                foreach (var source in sentimentHistory.Keys)
                {
                    var sourceHistory = sentimentHistory[source];
                    var closestData = sourceHistory
                        .OrderBy(s => Math.Abs((s.Date - date).TotalDays))
                        .FirstOrDefault();

                    if (closestData != default && Math.Abs((closestData.Date - date).TotalDays) <= 1)
                    {
                        result.SentimentBySource[source].Add(closestData.Value);
                        combinedSentiment += closestData.Value;
                        sourcesWithData++;
                        hasSentimentData = true;
                    }
                    else
                    {
                        result.SentimentBySource[source].Add(0); // No data
                    }
                }

                // Only add data point if we have sentiment data
                if (hasSentimentData)
                {
                    result.Dates.Add(date);
                    result.Prices.Add(price);
                    result.PriceChanges.Add(priceChange);
                    result.CombinedSentiment.Add(sourcesWithData > 0 ? combinedSentiment / sourcesWithData : 0);
                }
            }

            return result;
        }

        /// <summary>
        /// Calculate Pearson correlation coefficient between two series
        /// </summary>
        private double CalculatePearsonCorrelation(List<double> x, List<double> y)
        {
            if (x.Count != y.Count || x.Count == 0)
            {
                return 0;
            }

            int n = x.Count;

            // Calculate means
            double meanX = x.Average();
            double meanY = y.Average();

            // Calculate covariance and variances
            double covariance = 0;
            double varianceX = 0;
            double varianceY = 0;

            for (int i = 0; i < n; i++)
            {
                double deltaX = x[i] - meanX;
                double deltaY = y[i] - meanY;

                covariance += deltaX * deltaY;
                varianceX += deltaX * deltaX;
                varianceY += deltaY * deltaY;
            }

            if (varianceX == 0 || varianceY == 0)
            {
                return 0;
            }

            return covariance / Math.Sqrt(varianceX * varianceY);
        }

        /// <summary>
        /// Calculate lead/lag relationship between two series (positive means sentiment leads price)
        /// </summary>
        private double CalculateLeadLagRelationship(List<double> sentimentSeries, List<double> priceSeries)
        {
            if (sentimentSeries.Count != priceSeries.Count || sentimentSeries.Count < 5)
            {
                return 0;
            }

            // Calculate correlations with different lags (-5 to +5 days)
            double maxCorrelation = double.MinValue;
            int bestLag = 0;

            for (int lag = -5; lag <= 5; lag++)
            {
                if (lag == 0) continue;

                var laggedCorrelation = CalculateLaggedCorrelation(sentimentSeries, priceSeries, lag);
                if (laggedCorrelation > maxCorrelation)
                {
                    maxCorrelation = laggedCorrelation;
                    bestLag = lag;
                }
            }

            return bestLag; // Positive means sentiment leads price
        }

        /// <summary>
        /// Calculate correlation with a specific lag
        /// </summary>
        private double CalculateLaggedCorrelation(List<double> series1, List<double> series2, int lag)
        {
            int n = series1.Count;
            if (Math.Abs(lag) >= n)
            {
                return 0;
            }

            var x = new List<double>();
            var y = new List<double>();

            if (lag > 0)
            {
                // series1 leads series2
                x.AddRange(series1.Take(n - lag));
                y.AddRange(series2.Skip(lag));
            }
            else
            {
                // series2 leads series1
                x.AddRange(series1.Skip(-lag));
                y.AddRange(series2.Take(n + lag));
            }

            return CalculatePearsonCorrelation(x, y);
        }

        /// <summary>
        /// Identify significant sentiment shift events and their impact on prices
        /// </summary>
        private List<SentimentShiftEvent> IdentifySentimentShiftEvents(
            Dictionary<string, List<double>> sentimentBySource,
            List<DateTime> dates,
            List<double> priceChanges)
        {
            var events = new List<SentimentShiftEvent>();

            if (dates.Count < 3 || priceChanges.Count < dates.Count - 1)
                return events;

            // For each sentiment source
            foreach (var source in sentimentBySource.Keys)
            {
                var sentiment = sentimentBySource[source];
                if (sentiment.Count < dates.Count)
                    continue;

                // Look for significant shifts in sentiment
                for (int i = 1; i < sentiment.Count; i++)
                {
                    // Calculate the shift
                    double shift = sentiment[i] - sentiment[i - 1];

                    // If the shift is significant
                    if (Math.Abs(shift) >= 0.2) // 0.2 is threshold for significant shift
                    {
                        // Check subsequent price changes
                        double subsequentPriceChange = 0;
                        bool priceFollowedSentiment = false;

                        // Look at next 3 days for price changes
                        int daysToExamine = Math.Min(3, priceChanges.Count - i);
                        for (int j = 0; j < daysToExamine; j++)
                        {
                            int priceIdx = i + j;
                            if (priceIdx < priceChanges.Count)
                            {
                                subsequentPriceChange += priceChanges[priceIdx];

                                // Check if price movement aligned with sentiment shift
                                if ((shift > 0 && priceChanges[priceIdx] > 0) ||
                                    (shift < 0 && priceChanges[priceIdx] < 0))
                                {
                                    priceFollowedSentiment = true;
                                }
                            }
                        }

                        // Create an event
                        events.Add(new SentimentShiftEvent
                        {
                            Date = dates[i],
                            Source = source,
                            SentimentShift = shift,
                            SubsequentPriceChange = subsequentPriceChange,
                            PriceFollowedSentiment = priceFollowedSentiment
                        });
                    }
                }
            }

            return events;
        }

        /// <summary>
        /// Calculate how accurately sentiment shifts predict price movements
        /// </summary>
        private double CalculatePredictiveAccuracy(List<SentimentShiftEvent> events)
        {
            if (events.Count == 0)
                return 0;

            int correctPredictions = events.Count(e => e.PriceFollowedSentiment);
            return (double)correctPredictions / events.Count;
        }

        /// <summary>
        /// Generates historical sector performance data based on current momentum values
        /// (in a real implementation, this would come from a database of historical data)
        /// </summary>
        private List<(DateTime Date, double Value)> GenerateHistoricalSectorPerformance(
            string sector,
            List<SectorMomentumModel> momentumModels,
            List<DateTime> dates)
        {
            var result = new List<(DateTime, double)>();

            // Calculate the average momentum for the sector
            double avgMomentum = momentumModels.Select(m => m.MomentumValue).Average();

            // Use the sector name as seed for consistent but varied results per sector
            var random = new Random(sector.GetHashCode());

            // Generate synthetic performance data for each date
            foreach (var date in dates)
            {
                // Base value on current average momentum but add random variations
                // and a trend that increases over time
                double dayFactor = (double)(date - dates.Min()).TotalDays / (dates.Max() - dates.Min()).TotalDays;
                double trendComponent = dayFactor * avgMomentum * 2; // Trend increases over time
                double randomComponent = (random.NextDouble() * 0.4) - 0.2; // -0.2 to +0.2 variation

                double performance = avgMomentum * 0.5 + trendComponent + randomComponent;

                // Ensure within reasonable bounds
                performance = Math.Max(-0.5, Math.Min(0.5, performance));

                result.Add((date, performance));
            }

            return result.OrderBy(r => r.Item1).ToList();
        }

        /// <summary>
        /// Aligns sector sentiment and performance data by date
        /// </summary>
        private (List<DateTime> Dates, List<double> SentimentValues, List<double> PerformanceValues)
            AlignSectorData(
                List<(DateTime Date, double Sentiment)> sectorSentiment,
                List<(DateTime Date, double Performance)> sectorPerformance)
        {
            // Create result containers
            var dates = new List<DateTime>();
            var sentimentValues = new List<double>();
            var performanceValues = new List<double>();

            // Use dates that exist in both datasets
            var sentimentDates = sectorSentiment.Select(s => s.Date.Date).ToHashSet();
            var performanceDates = sectorPerformance.Select(p => p.Date.Date).ToHashSet();
            var commonDates = sentimentDates.Intersect(performanceDates).OrderBy(d => d).ToList();

            foreach (var date in commonDates)
            {
                // Find matching entries
                var sentimentEntry = sectorSentiment.FirstOrDefault(s => s.Date.Date == date);
                var performanceEntry = sectorPerformance.FirstOrDefault(p => p.Date.Date == date);

                // Add if both exist
                if (sentimentEntry != default && performanceEntry != default)
                {
                    dates.Add(date);
                    sentimentValues.Add(sentimentEntry.Sentiment);
                    performanceValues.Add(performanceEntry.Performance);
                }
            }

            return (dates, sentimentValues, performanceValues);
        }

        /// <summary>
        /// Identifies significant sector sentiment shift events and their impact on sector performance
        /// </summary>
        private List<SentimentShiftEvent> IdentifySectorSentimentShiftEvents(
            List<DateTime> dates,
            List<double> sentimentValues,
            List<double> performanceValues)
        {
            var events = new List<SentimentShiftEvent>();

            if (dates.Count < 3 || sentimentValues.Count < dates.Count || performanceValues.Count < dates.Count)
                return events;

            // Look for significant shifts in sentiment
            for (int i = 1; i < sentimentValues.Count; i++)
            {
                // Calculate the shift
                double shift = sentimentValues[i] - sentimentValues[i - 1];

                // If the shift is significant
                if (Math.Abs(shift) >= 0.15) // 0.15 is threshold for significant shift (lower than stock-specific threshold)
                {
                    // Check subsequent performance changes
                    double subsequentPerformanceChange = 0;
                    bool performanceFollowedSentiment = false;

                    // Look at next 3 days for performance changes
                    int daysToExamine = Math.Min(3, performanceValues.Count - i);
                    for (int j = 0; j < daysToExamine; j++)
                    {
                        int perfIdx = i + j;
                        if (perfIdx < performanceValues.Count)
                        {
                            double perfChange = j > 0 ? performanceValues[perfIdx] - performanceValues[perfIdx - 1] : 0;
                            subsequentPerformanceChange += perfChange;

                            // Check if performance movement aligned with sentiment shift
                            if ((shift > 0 && perfChange > 0) || (shift < 0 && perfChange < 0))
                            {
                                performanceFollowedSentiment = true;
                            }
                        }
                    }

                    // Create an event
                    events.Add(new SentimentShiftEvent
                    {
                        Date = dates[i],
                        Source = "Sector", // Source is the sector itself
                        SentimentShift = shift,
                        SubsequentPriceChange = subsequentPerformanceChange,
                        PriceFollowedSentiment = performanceFollowedSentiment
                    });
                }
            }

            return events;
        }

        /// <summary>
        /// Gets formatted historical sentiment context for Market Chat integration (MarketChat story 6).
        /// Returns a summary of sentiment-price correlation data suitable for AI prompt enhancement.
        /// </summary>
        /// <param name="symbol">Stock symbol to analyze</param>
        /// <param name="days">Number of days to include in the analysis (default 30)</param>
        /// <returns>Formatted context string with correlation coefficients and sentiment shift summaries</returns>
        public async Task<string> GetHistoricalSentimentContext(string symbol, int days = 30)
        {
            try
            {
                // Get the correlation analysis
                var correlationResult = await AnalyzeSentimentPriceCorrelation(symbol, days);

                if (correlationResult == null || correlationResult.SourceCorrelations.Count == 0)
                {
                    return string.Empty;
                }

                var contextBuilder = new System.Text.StringBuilder();
                contextBuilder.AppendLine($"Sentiment-Price Correlation Analysis for {symbol} ({days} days):");
                contextBuilder.AppendLine();

                // Overall correlation interpretation
                contextBuilder.AppendLine($"- Overall Sentiment-Price Correlation: {correlationResult.OverallCorrelation:+0.00;-0.00}");
                contextBuilder.Append("  Interpretation: ");
                if (Math.Abs(correlationResult.OverallCorrelation) >= 0.7)
                {
                    contextBuilder.AppendLine(correlationResult.OverallCorrelation > 0
                        ? "Strong positive correlation - sentiment shifts tend to align with price movements"
                        : "Strong negative correlation - sentiment and price often move inversely");
                }
                else if (Math.Abs(correlationResult.OverallCorrelation) >= 0.4)
                {
                    contextBuilder.AppendLine(correlationResult.OverallCorrelation > 0
                        ? "Moderate positive correlation - sentiment provides useful predictive signal"
                        : "Moderate negative correlation - contrarian sentiment indicator may apply");
                }
                else
                {
                    contextBuilder.AppendLine("Weak correlation - sentiment alone is not a strong predictor for this symbol");
                }
                contextBuilder.AppendLine();

                // Source-specific correlations
                contextBuilder.AppendLine("Source-Specific Correlations:");
                foreach (var source in correlationResult.SourceCorrelations.OrderByDescending(x => Math.Abs(x.Value)))
                {
                    string interpretation = GetCorrelationInterpretation(source.Value);
                    contextBuilder.AppendLine($"  • {source.Key}: {source.Value:+0.00;-0.00} ({interpretation})");
                }
                contextBuilder.AppendLine();

                // Lead/lag relationship
                if (correlationResult.LeadLagRelationship != 0)
                {
                    string leadLagText = correlationResult.LeadLagRelationship > 0
                        ? $"Sentiment leads price by approximately {Math.Abs(correlationResult.LeadLagRelationship):F1} day(s)"
                        : $"Price leads sentiment by approximately {Math.Abs(correlationResult.LeadLagRelationship):F1} day(s)";
                    contextBuilder.AppendLine($"- Lead/Lag Relationship: {leadLagText}");
                }

                // Predictive accuracy
                contextBuilder.AppendLine($"- Historical Predictive Accuracy: {correlationResult.PredictiveAccuracy:P1}");
                contextBuilder.Append("  ");
                if (correlationResult.PredictiveAccuracy >= 0.7)
                {
                    contextBuilder.AppendLine("This is high - sentiment shifts have reliably predicted subsequent price movements.");
                }
                else if (correlationResult.PredictiveAccuracy >= 0.5)
                {
                    contextBuilder.AppendLine("This is moderate - sentiment provides some predictive value but should be combined with other indicators.");
                }
                else
                {
                    contextBuilder.AppendLine("This is low - sentiment alone has not been a reliable predictor for this symbol.");
                }
                contextBuilder.AppendLine();

                // Recent sentiment shift events
                if (correlationResult.SentimentShiftEvents != null && correlationResult.SentimentShiftEvents.Count > 0)
                {
                    var recentEvents = correlationResult.SentimentShiftEvents
                        .OrderByDescending(e => e.Date)
                        .Take(5)
                        .ToList();

                    contextBuilder.AppendLine("Recent Sentiment Shift Events:");
                    foreach (var evt in recentEvents)
                    {
                        string shiftDirection = evt.SentimentShift > 0 ? "positive" : "negative";
                        string priceOutcome = evt.PriceFollowedSentiment ? "confirmed" : "not confirmed";
                        contextBuilder.AppendLine($"  • {evt.Date:MMM dd}: {evt.Source} shifted {shiftDirection} ({evt.SentimentShift:+0.00;-0.00}), " +
                            $"price moved {evt.SubsequentPriceChange:+0.0;-0.0}% - {priceOutcome}");
                    }

                    // Calculate recent accuracy
                    int correctPredictions = recentEvents.Count(e => e.PriceFollowedSentiment);
                    contextBuilder.AppendLine($"  Recent accuracy: {correctPredictions}/{recentEvents.Count} predictions aligned with price movement");
                }

                return contextBuilder.ToString();
            }
            catch (Exception ex)
            {
                _loggingService?.LogWarning($"Failed to get historical sentiment context for {symbol}: {ex.Message}");
                return string.Empty;
            }
        }

        /// <summary>
        /// Gets a human-readable interpretation of a correlation coefficient
        /// </summary>
        private string GetCorrelationInterpretation(double correlation)
        {
            double absCorrelation = Math.Abs(correlation);
            string direction = correlation >= 0 ? "positive" : "negative";

            if (absCorrelation >= 0.7)
                return $"strong {direction}";
            else if (absCorrelation >= 0.4)
                return $"moderate {direction}";
            else if (absCorrelation >= 0.2)
                return $"weak {direction}";
            else
                return "negligible";
        }

        #endregion
    }

    #region Model Classes

    /// <summary>
    /// Results of sentiment-price correlation analysis
    /// </summary>
    public class SentimentPriceCorrelationResult
    {
        /// <summary>
        /// Symbol being analyzed
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Overall correlation between combined sentiment and price changes
        /// </summary>
        public double OverallCorrelation { get; set; }

        /// <summary>
        /// Correlations for each sentiment source
        /// </summary>
        public Dictionary<string, double> SourceCorrelations { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Lead/lag relationship in days (positive means sentiment leads price)
        /// </summary>
        public double LeadLagRelationship { get; set; }

        /// <summary>
        /// Accuracy of sentiment shifts in predicting price movements
        /// </summary>
        public double PredictiveAccuracy { get; set; }

        /// <summary>
        /// Significant sentiment shift events
        /// </summary>
        public List<SentimentShiftEvent> SentimentShiftEvents { get; set; } = new List<SentimentShiftEvent>();

        /// <summary>
        /// Aligned data used for analysis
        /// </summary>
        public SentimentPriceAlignedData AlignedData { get; set; }
    }

    /// <summary>
    /// Represents a significant shift in sentiment and its impact on price
    /// </summary>
    public class SentimentShiftEvent
    {
        /// <summary>
        /// Date of the sentiment shift
        /// </summary>
        public DateTime Date { get; set; }

        /// <summary>
        /// Source of sentiment (News, Twitter, etc.)
        /// </summary>
        public string Source { get; set; }

        /// <summary>
        /// Magnitude of sentiment shift
        /// </summary>
        public double SentimentShift { get; set; }

        /// <summary>
        /// Subsequent change in price (%)
        /// </summary>
        public double SubsequentPriceChange { get; set; }

        /// <summary>
        /// Whether the price movement aligned with sentiment shift
        /// </summary>
        public bool PriceFollowedSentiment { get; set; }
    }

    /// <summary>
    /// Aligned sentiment and price data for analysis
    /// </summary>
    public class SentimentPriceAlignedData
    {
        /// <summary>
        /// Dates for the time series
        /// </summary>
        public List<DateTime> Dates { get; set; } = new List<DateTime>();

        /// <summary>
        /// Price data
        /// </summary>
        public List<double> Prices { get; set; } = new List<double>();

        /// <summary>
        /// Daily price changes (%)
        /// </summary>
        public List<double> PriceChanges { get; set; } = new List<double>();

        /// <summary>
        /// Sentiment values for each source
        /// </summary>
        public Dictionary<string, List<double>> SentimentBySource { get; set; } =
            new Dictionary<string, List<double>>();

        /// <summary>
        /// Combined sentiment across all sources
        /// </summary>
        public List<double> CombinedSentiment { get; set; } = new List<double>();
    }

    /// <summary>
    /// Data for visualizing sentiment-price correlations
    /// </summary>
    public class SentimentPriceVisualData
    {
        /// <summary>
        /// Symbol being analyzed
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Dates for the time series
        /// </summary>
        public List<DateTime> Dates { get; set; } = new List<DateTime>();

        /// <summary>
        /// Price data
        /// </summary>
        public List<double> Prices { get; set; } = new List<double>();

        /// <summary>
        /// Daily price changes (%)
        /// </summary>
        public List<double> PriceChanges { get; set; } = new List<double>();

        /// <summary>
        /// Sentiment values for each source
        /// </summary>
        public Dictionary<string, List<double>> SentimentBySource { get; set; } =
            new Dictionary<string, List<double>>();

        /// <summary>
        /// Combined sentiment across all sources
        /// </summary>
        public List<double> CombinedSentiment { get; set; } = new List<double>();

        /// <summary>
        /// Overall correlation between combined sentiment and price changes
        /// </summary>
        public double OverallCorrelation { get; set; }

        /// <summary>
        /// Correlations for each sentiment source
        /// </summary>
        public Dictionary<string, double> SourceCorrelations { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Lead/lag relationship in days (positive means sentiment leads price)
        /// </summary>
        public double LeadLagRelationship { get; set; }

        /// <summary>
        /// Accuracy of sentiment shifts in predicting price movements
        /// </summary>
        public double PredictiveAccuracy { get; set; }

        /// <summary>
        /// Significant sentiment shift events
        /// </summary>
        public List<SentimentShiftEvent> SentimentShiftEvents { get; set; } = new List<SentimentShiftEvent>();
    }

    /// <summary>
    /// Results of sector sentiment correlation analysis
    /// </summary>
    public class SectorSentimentCorrelationResult
    {
        /// <summary>
        /// Sector being analyzed
        /// </summary>
        public string Sector { get; set; }

        /// <summary>
        /// Overall correlation between sector sentiment and performance
        /// </summary>
        public double OverallCorrelation { get; set; }

        /// <summary>
        /// The sector sentiment time series data
        /// </summary>
        public List<(DateTime Date, double Sentiment)> SectorSentiment { get; set; } = new List<(DateTime, double)>();

        /// <summary>
        /// The sector performance time series data
        /// </summary>
        public List<(DateTime Date, double Value)> SectorPerformance { get; set; } = new List<(DateTime, double)>();

        /// <summary>
        /// Lead/lag relationship in days (positive means sentiment leads performance)
        /// </summary>
        public double LeadLagRelationship { get; set; }

        /// <summary>
        /// Significant sentiment shift events in the sector
        /// </summary>
        public List<SentimentShiftEvent> SentimentShiftEvents { get; set; } = new List<SentimentShiftEvent>();

        /// <summary>
        /// Dates where both sentiment and performance data are aligned
        /// </summary>
        public List<DateTime> AlignedDates { get; set; } = new List<DateTime>();
    }

    #endregion
}