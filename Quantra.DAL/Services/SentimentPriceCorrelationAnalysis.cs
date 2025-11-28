using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for analyzing correlation between sentiment and price/performance
    /// Used by SectorSentimentController for sector-specific analyses
    /// </summary>
    public class SentimentPriceCorrelationAnalysis
    {
        /// <summary>
        /// Initializes a new instance of the SentimentPriceCorrelationAnalysis service
        /// </summary>
        public SentimentPriceCorrelationAnalysis()
        {
        }

        /// <summary>
        /// Analyzes correlation between sentiment changes and price movements for a specific symbol
        /// Placeholder implementation to unblock callers in DAL.
        /// </summary>
        public async Task<SentimentPriceCorrelationResult> AnalyzeSentimentPriceCorrelation(
            string symbol,
            int lookbackDays = 30,
            List<string> sentimentSources = null)
        {
            // Minimal placeholder that returns empty aligned data so callers can work.
            await Task.Delay(1);
            return new SentimentPriceCorrelationResult
            {
                Symbol = symbol,
                OverallCorrelation = 0,
                SourceCorrelations = new Dictionary<string, double>(),
                LeadLagRelationship = 0,
                PredictiveAccuracy = 0,
                SentimentShiftEvents = new List<SentimentShiftEvent>(),
                AlignedData = new SentimentPriceAlignedData
                {
                    Dates = new List<DateTime>(),
                    Prices = new List<double>(),
                    PriceChanges = new List<double>(),
                    SentimentBySource = new Dictionary<string, List<double>>(),
                    CombinedSentiment = new List<double>()
                }
            };
        }

        /// <summary>
        /// Analyzes correlation between sector sentiment and sector price/performance
        /// </summary>
        /// <param name="sector">Market sector to analyze</param>
        /// <returns>Correlation analysis results</returns>
        public async Task<CorrelationResult> AnalyzeSectorSentimentCorrelation(string sector)
        {
            // TODO: Implement correlation analysis logic
            // For now, return a basic result structure
            await Task.Delay(1); // Placeholder for async operation

            return new CorrelationResult
            {
                OverallCorrelation = 0.0,
                LeadLagRelationship = 0.0,
                SentimentShiftEvents = new List<SentimentShiftEvent>()
            };
        }

        /// <summary>
        /// Gets formatted historical sentiment context for Market Chat integration (MarketChat story 6).
        /// Returns a summary of sentiment-price correlation data suitable for AI prompt enhancement.
        /// Placeholder implementation for DAL - full implementation is in Quantra.Modules.
        /// </summary>
        /// <param name="symbol">Stock symbol to analyze</param>
        /// <param name="days">Number of days to include in the analysis (default 30)</param>
        /// <returns>Formatted context string with correlation coefficients and sentiment shift summaries</returns>
        public async Task<string> GetHistoricalSentimentContext(string symbol, int days = 30)
        {
            // Validate input parameters
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return string.Empty;
            }

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
                if (correlationResult.SourceCorrelations.Count > 0)
                {
                    contextBuilder.AppendLine("Source-Specific Correlations:");
                    foreach (var source in correlationResult.SourceCorrelations.OrderByDescending(x => Math.Abs(x.Value)))
                    {
                        string interpretation = GetCorrelationInterpretation(source.Value);
                        contextBuilder.AppendLine($"  • {source.Key}: {source.Value:+0.00;-0.00} ({interpretation})");
                    }
                    contextBuilder.AppendLine();
                }

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
            catch (Exception)
            {
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
    }

    /// <summary>
    /// Result of correlation analysis between sentiment and price/performance
    /// </summary>
    public class CorrelationResult
    {
        /// <summary>
        /// Overall correlation between sentiment and price/performance
        /// </summary>
        public double OverallCorrelation { get; set; }

        /// <summary>
        /// Lead/lag relationship in days (positive means sentiment leads price/performance)
        /// </summary>
        public double LeadLagRelationship { get; set; }

        /// <summary>
        /// Significant sentiment shift events
        /// </summary>
        public List<SentimentShiftEvent> SentimentShiftEvents { get; set; } = new List<SentimentShiftEvent>();
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

    // Lightweight result types to satisfy DAL usage without depending on Helpers
    public class SentimentPriceCorrelationResult
    {
        public string Symbol { get; set; }
        public double OverallCorrelation { get; set; }
        public Dictionary<string, double> SourceCorrelations { get; set; } = new Dictionary<string, double>();
        public double LeadLagRelationship { get; set; }
        public double PredictiveAccuracy { get; set; }
        public List<SentimentShiftEvent> SentimentShiftEvents { get; set; } = new List<SentimentShiftEvent>();
        public SentimentPriceAlignedData AlignedData { get; set; } = new SentimentPriceAlignedData();
    }

    public class SentimentPriceAlignedData
    {
        public List<DateTime> Dates { get; set; } = new List<DateTime>();
        public List<double> Prices { get; set; } = new List<double>();
        public List<double> PriceChanges { get; set; } = new List<double>();
        public Dictionary<string, List<double>> SentimentBySource { get; set; } = new Dictionary<string, List<double>>();
        public List<double> CombinedSentiment { get; set; } = new List<double>();
    }
}