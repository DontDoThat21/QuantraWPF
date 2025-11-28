using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for generating LiveCharts-compatible chart data for historical prices
    /// and ML projections. Integrates with MarketChat to display charts inline (MarketChat story 8).
    /// </summary>
    public class ChartGenerationService : IChartGenerationService
    {
        private readonly IHistoricalDataService _historicalDataService;
        private readonly IPredictionDataService _predictionDataService;
        private readonly ILogger<ChartGenerationService> _logger;

        // Regex patterns for chart request detection
        private static readonly Regex ChartKeywordPattern = new Regex(
            @"\b(chart|graph|plot|visuali[sz]e|show|display|draw)\b.*\b(price|prediction|forecast|projection|trend|history|historical)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private static readonly Regex AlternativeChartPattern = new Regex(
            @"\b(price|prediction|forecast|projection)\b.*\b(chart|graph|plot|visual)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private static readonly Regex SymbolPattern = new Regex(
            @"\b([A-Z]{1,5})\b",
            RegexOptions.Compiled);

        private static readonly Regex DaysPattern = new Regex(
            @"(\d+)\s*(?:day|week|month)s?",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        // Common words to exclude from symbol detection
        private static readonly HashSet<string> ExcludedWords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "I", "A", "AN", "THE", "IN", "ON", "AT", "TO", "FOR", "OF", "AND", "OR", "IS", "IT",
            "BE", "AS", "BY", "IF", "DO", "GO", "SO", "NO", "UP", "MY", "ME", "WE", "US", "AM",
            "CAN", "ALL", "NEW", "ONE", "TWO", "NOW", "HOW", "WHY", "WHAT", "WHEN", "WHO",
            "CHART", "GRAPH", "PLOT", "SHOW", "DRAW", "RSI", "EMA", "SMA", "MACD", "ATR", "ADX"
        };

        /// <summary>
        /// Constructor for ChartGenerationService
        /// </summary>
        public ChartGenerationService(
            IHistoricalDataService historicalDataService = null,
            IPredictionDataService predictionDataService = null,
            ILogger<ChartGenerationService> logger = null)
        {
            _historicalDataService = historicalDataService;
            _predictionDataService = predictionDataService;
            _logger = logger;
        }

        /// <summary>
        /// Generates projection chart data combining historical prices and ML predictions.
        /// </summary>
        public async Task<ProjectionChartData> GenerateProjectionChartAsync(string symbol, DateTime startDate, int forecastDays)
        {
            var chartData = new ProjectionChartData
            {
                Symbol = symbol?.ToUpperInvariant(),
                ChartTitle = $"{symbol?.ToUpperInvariant()} Price Chart with {forecastDays}-Day Projection"
            };

            try
            {
                _logger?.LogInformation("Generating projection chart for {Symbol} from {StartDate} with {ForecastDays} day forecast",
                    symbol, startDate, forecastDays);

                // Get historical data
                var historicalPrices = await GetHistoricalDataAsync(symbol, startDate);
                if (historicalPrices == null || historicalPrices.Count == 0)
                {
                    chartData.ErrorMessage = $"No historical data available for {symbol}";
                    return chartData;
                }

                // Populate historical data
                chartData.HistoricalPrices = historicalPrices.Select(p => p.Close).ToList();
                chartData.HistoricalDates = historicalPrices.Select(p => p.Date).ToList();
                chartData.CurrentPrice = historicalPrices.Last().Close;

                // Calculate Bollinger Bands
                CalculateBollingerBands(chartData, historicalPrices);

                // Calculate support and resistance levels
                CalculateSupportResistanceLevels(chartData, historicalPrices);

                // Get ML prediction data
                await PopulatePredictionDataAsync(chartData, symbol, forecastDays);

                // Build combined series for continuous display
                BuildCombinedSeries(chartData);

                // Generate X-axis labels
                GenerateXAxisLabels(chartData);

                _logger?.LogInformation("Successfully generated chart data for {Symbol} with {HistoricalCount} historical points and {PredictionCount} prediction points",
                    symbol, chartData.HistoricalPrices.Count, chartData.PredictionPrices.Count);

                return chartData;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating projection chart for {Symbol}", symbol);
                chartData.ErrorMessage = $"Error generating chart: {ex.Message}";
                return chartData;
            }
        }

        /// <summary>
        /// Generates chart data from existing historical data and prediction result.
        /// </summary>
        public ProjectionChartData GenerateChartFromData(List<HistoricalPrice> historicalData, PredictionResult prediction, string symbol)
        {
            var chartData = new ProjectionChartData
            {
                Symbol = symbol?.ToUpperInvariant(),
                ChartTitle = $"{symbol?.ToUpperInvariant()} Price Chart with Projection"
            };

            try
            {
                if (historicalData == null || historicalData.Count == 0)
                {
                    chartData.ErrorMessage = "No historical data provided";
                    return chartData;
                }

                // Populate historical data
                var sortedData = historicalData.OrderBy(p => p.Date).ToList();
                chartData.HistoricalPrices = sortedData.Select(p => p.Close).ToList();
                chartData.HistoricalDates = sortedData.Select(p => p.Date).ToList();
                chartData.CurrentPrice = sortedData.Last().Close;

                // Calculate Bollinger Bands
                CalculateBollingerBands(chartData, sortedData);

                // Calculate support and resistance levels
                CalculateSupportResistanceLevels(chartData, sortedData);

                // Populate prediction data from provided result
                if (prediction != null)
                {
                    PopulatePredictionFromResult(chartData, prediction);
                }

                // Build combined series
                BuildCombinedSeries(chartData);

                // Generate X-axis labels
                GenerateXAxisLabels(chartData);

                return chartData;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating chart from provided data for {Symbol}", symbol);
                chartData.ErrorMessage = $"Error generating chart: {ex.Message}";
                return chartData;
            }
        }

        /// <summary>
        /// Determines if a user message is requesting a chart or visualization.
        /// </summary>
        public bool IsChartRequest(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return false;
            }

            // Check for chart-related keywords
            return ChartKeywordPattern.IsMatch(message) || AlternativeChartPattern.IsMatch(message);
        }

        /// <summary>
        /// Extracts chart parameters from a user's message.
        /// </summary>
        public ChartRequestParameters ExtractChartParameters(string message)
        {
            var parameters = new ChartRequestParameters();

            if (string.IsNullOrWhiteSpace(message))
            {
                return parameters;
            }

            // Extract symbols
            var symbolMatches = SymbolPattern.Matches(message);
            foreach (Match match in symbolMatches)
            {
                var symbol = match.Groups[1].Value;
                if (!ExcludedWords.Contains(symbol) && !parameters.Symbols.Contains(symbol))
                {
                    parameters.Symbols.Add(symbol);
                }
            }

            // Extract forecast days
            var daysMatch = DaysPattern.Match(message);
            if (daysMatch.Success)
            {
                if (int.TryParse(daysMatch.Groups[1].Value, out int days))
                {
                    var unit = daysMatch.Value.ToLower();
                    if (unit.Contains("week"))
                    {
                        days *= 7;
                    }
                    else if (unit.Contains("month"))
                    {
                        days *= 30;
                    }
                    parameters.ForecastDays = Math.Min(days, 90); // Cap at 90 days
                }
            }

            // Check for specific indicators
            var lowerMessage = message.ToLowerInvariant();
            parameters.IncludeBollingerBands = !lowerMessage.Contains("no bollinger") && !lowerMessage.Contains("without bollinger");
            parameters.IncludeSupportResistance = !lowerMessage.Contains("no support") && !lowerMessage.Contains("without support");

            return parameters;
        }

        #region Private Helper Methods

        /// <summary>
        /// Gets historical data for a symbol starting from the specified date.
        /// </summary>
        private async Task<List<HistoricalPrice>> GetHistoricalDataAsync(string symbol, DateTime startDate)
        {
            if (_historicalDataService == null)
            {
                _logger?.LogWarning("Historical data service is not configured");
                return GenerateMockHistoricalData(symbol, startDate);
            }

            try
            {
                var allData = await _historicalDataService.GetHistoricalPrices(symbol, "max", "daily");
                return allData?.Where(p => p.Date >= startDate).OrderBy(p => p.Date).ToList() ?? new List<HistoricalPrice>();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error fetching historical data for {Symbol}", symbol);
                return new List<HistoricalPrice>();
            }
        }

        /// <summary>
        /// Generates mock historical data for testing when service is unavailable.
        /// </summary>
        private List<HistoricalPrice> GenerateMockHistoricalData(string symbol, DateTime startDate)
        {
            var data = new List<HistoricalPrice>();
            var random = new Random(symbol?.GetHashCode() ?? 42);
            double basePrice = 100 + random.NextDouble() * 200;
            double currentPrice = basePrice;

            for (int i = 0; i < 60; i++)
            {
                var date = startDate.AddDays(i);
                if (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday)
                {
                    continue;
                }

                double change = (random.NextDouble() - 0.48) * 3; // Slight upward bias
                currentPrice = Math.Max(currentPrice + change, basePrice * 0.5);

                double high = currentPrice + random.NextDouble() * 2;
                double low = currentPrice - random.NextDouble() * 2;
                double open = currentPrice + (random.NextDouble() - 0.5) * 2;

                data.Add(new HistoricalPrice
                {
                    Date = date,
                    Open = Math.Round(open, 2),
                    High = Math.Round(high, 2),
                    Low = Math.Round(low, 2),
                    Close = Math.Round(currentPrice, 2),
                    Volume = random.Next(1000000, 10000000),
                    AdjClose = Math.Round(currentPrice, 2)
                });
            }

            return data;
        }

        /// <summary>
        /// Calculates Bollinger Bands for the chart data.
        /// </summary>
        private void CalculateBollingerBands(ProjectionChartData chartData, List<HistoricalPrice> prices)
        {
            const int period = 20;
            const double stdDevMultiplier = 2.0;

            var closes = prices.Select(p => p.Close).ToList();

            chartData.BollingerUpper = new List<double>();
            chartData.BollingerMiddle = new List<double>();
            chartData.BollingerLower = new List<double>();

            // Add NaN for initial periods where we can't calculate
            for (int i = 0; i < Math.Min(period - 1, closes.Count); i++)
            {
                chartData.BollingerUpper.Add(double.NaN);
                chartData.BollingerMiddle.Add(double.NaN);
                chartData.BollingerLower.Add(double.NaN);
            }

            // Calculate bands for remaining periods
            for (int i = period - 1; i < closes.Count; i++)
            {
                var periodPrices = closes.Skip(i - period + 1).Take(period).ToList();
                var average = periodPrices.Average();
                var variance = periodPrices.Sum(p => Math.Pow(p - average, 2)) / period;
                var stdDev = Math.Sqrt(variance);

                chartData.BollingerMiddle.Add(Math.Round(average, 2));
                chartData.BollingerUpper.Add(Math.Round(average + stdDevMultiplier * stdDev, 2));
                chartData.BollingerLower.Add(Math.Round(average - stdDevMultiplier * stdDev, 2));
            }
        }

        /// <summary>
        /// Calculates support and resistance levels from historical data.
        /// Uses a simple pivot point approach with local minima/maxima.
        /// </summary>
        private void CalculateSupportResistanceLevels(ProjectionChartData chartData, List<HistoricalPrice> prices)
        {
            if (prices.Count < 5)
            {
                return;
            }

            var supports = new List<double>();
            var resistances = new List<double>();

            // Find local minima (support) and maxima (resistance)
            for (int i = 2; i < prices.Count - 2; i++)
            {
                var current = prices[i].Low;
                var prev1 = prices[i - 1].Low;
                var prev2 = prices[i - 2].Low;
                var next1 = prices[i + 1].Low;
                var next2 = prices[i + 2].Low;

                // Local minimum (support)
                if (current <= prev1 && current <= prev2 && current <= next1 && current <= next2)
                {
                    supports.Add(current);
                }

                current = prices[i].High;
                prev1 = prices[i - 1].High;
                prev2 = prices[i - 2].High;
                next1 = prices[i + 1].High;
                next2 = prices[i + 2].High;

                // Local maximum (resistance)
                if (current >= prev1 && current >= prev2 && current >= next1 && current >= next2)
                {
                    resistances.Add(current);
                }
            }

            // Cluster nearby levels and take the most significant ones
            chartData.SupportLevels = ClusterLevels(supports, 3).Select(l => Math.Round(l, 2)).ToList();
            chartData.ResistanceLevels = ClusterLevels(resistances, 3).Select(l => Math.Round(l, 2)).ToList();
        }

        /// <summary>
        /// Clusters nearby price levels and returns the most significant ones.
        /// </summary>
        private List<double> ClusterLevels(List<double> levels, int maxLevels)
        {
            if (levels.Count == 0)
            {
                return new List<double>();
            }

            // Sort and group nearby levels (within 2% of each other)
            var sorted = levels.OrderBy(l => l).ToList();
            var clusters = new List<List<double>>();
            var currentCluster = new List<double> { sorted[0] };

            for (int i = 1; i < sorted.Count; i++)
            {
                if (sorted[i] <= currentCluster.Average() * 1.02)
                {
                    currentCluster.Add(sorted[i]);
                }
                else
                {
                    clusters.Add(currentCluster);
                    currentCluster = new List<double> { sorted[i] };
                }
            }
            clusters.Add(currentCluster);

            // Return the average of each cluster, taking the largest clusters
            return clusters
                .OrderByDescending(c => c.Count)
                .Take(maxLevels)
                .Select(c => c.Average())
                .OrderBy(l => l)
                .ToList();
        }

        /// <summary>
        /// Populates prediction data from the prediction service.
        /// </summary>
        private async Task PopulatePredictionDataAsync(ProjectionChartData chartData, string symbol, int forecastDays)
        {
            if (_predictionDataService == null)
            {
                _logger?.LogWarning("Prediction data service is not configured, generating mock predictions");
                GenerateMockPredictions(chartData, forecastDays);
                return;
            }

            try
            {
                var predictionContext = await _predictionDataService.GetPredictionContextWithCacheAsync(symbol);
                if (predictionContext?.Prediction != null)
                {
                    PopulatePredictionFromResult(chartData, predictionContext.Prediction);
                }
                else
                {
                    GenerateMockPredictions(chartData, forecastDays);
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error fetching prediction data for {Symbol}", symbol);
                GenerateMockPredictions(chartData, forecastDays);
            }
        }

        /// <summary>
        /// Populates chart data from a prediction result.
        /// </summary>
        private void PopulatePredictionFromResult(ProjectionChartData chartData, PredictionResult prediction)
        {
            chartData.PredictedAction = prediction.Action ?? prediction.PredictedAction;
            chartData.TargetPrice = prediction.TargetPrice;
            chartData.Confidence = prediction.Confidence;

            if (prediction.TimeSeries?.PricePredictions?.Count > 0)
            {
                chartData.PredictionPrices = prediction.TimeSeries.PricePredictions;
                chartData.PredictionDates = prediction.TimeSeries.TimePoints;
            }
            else
            {
                // Generate synthetic prediction curve from current to target
                GenerateSyntheticPredictions(chartData, 30);
            }
        }

        /// <summary>
        /// Generates mock predictions when no prediction service is available.
        /// </summary>
        private void GenerateMockPredictions(ProjectionChartData chartData, int forecastDays)
        {
            GenerateSyntheticPredictions(chartData, forecastDays);
            chartData.PredictedAction = chartData.TargetPrice > chartData.CurrentPrice ? "BUY" : "SELL";
            chartData.Confidence = 0.75;
        }

        /// <summary>
        /// Generates a synthetic prediction curve from current price to target.
        /// </summary>
        private void GenerateSyntheticPredictions(ProjectionChartData chartData, int forecastDays)
        {
            if (chartData.CurrentPrice <= 0)
            {
                return;
            }

            // Default target: 5% change from current price
            if (chartData.TargetPrice <= 0)
            {
                chartData.TargetPrice = Math.Round(chartData.CurrentPrice * 1.05, 2);
            }

            var random = new Random();
            var lastDate = chartData.HistoricalDates.LastOrDefault();
            if (lastDate == default)
            {
                lastDate = DateTime.Now;
            }

            chartData.PredictionPrices = new List<double>();
            chartData.PredictionDates = new List<DateTime>();

            double priceChange = (chartData.TargetPrice - chartData.CurrentPrice) / forecastDays;
            double currentPrediction = chartData.CurrentPrice;

            for (int i = 1; i <= forecastDays; i++)
            {
                var date = lastDate.AddDays(i);
                if (date.DayOfWeek == DayOfWeek.Saturday)
                {
                    date = date.AddDays(2);
                }
                else if (date.DayOfWeek == DayOfWeek.Sunday)
                {
                    date = date.AddDays(1);
                }

                // Add some noise to make the prediction curve more realistic
                double noise = (random.NextDouble() - 0.5) * Math.Abs(priceChange) * 0.5;
                currentPrediction += priceChange + noise;

                chartData.PredictionPrices.Add(Math.Round(currentPrediction, 2));
                chartData.PredictionDates.Add(date);
            }

            // Ensure the last prediction matches the target
            if (chartData.PredictionPrices.Count > 0)
            {
                chartData.PredictionPrices[chartData.PredictionPrices.Count - 1] = chartData.TargetPrice;
            }
        }

        /// <summary>
        /// Builds the combined series for continuous chart display.
        /// </summary>
        private void BuildCombinedSeries(ProjectionChartData chartData)
        {
            chartData.CombinedPrices = new List<double>(chartData.HistoricalPrices);
            chartData.CombinedDates = new List<DateTime>(chartData.HistoricalDates);
            chartData.PredictionStartIndex = chartData.HistoricalPrices.Count;

            if (chartData.PredictionPrices?.Count > 0)
            {
                chartData.CombinedPrices.AddRange(chartData.PredictionPrices);
                chartData.CombinedDates.AddRange(chartData.PredictionDates);
            }
        }

        /// <summary>
        /// Generates formatted X-axis labels for the chart.
        /// </summary>
        private void GenerateXAxisLabels(ProjectionChartData chartData)
        {
            chartData.XAxisLabels = chartData.CombinedDates
                .Select(d => d.ToString("MM/dd"))
                .ToList();
        }

        #endregion
    }
}
