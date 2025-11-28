using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for multi-symbol comparative analysis.
    /// Batches database lookups for predictions and historical data,
    /// calculates comparison scores, and generates formatted comparison tables.
    /// (MarketChat story 7)
    /// </summary>
    public class MultiSymbolAnalyzer : IMultiSymbolAnalyzer
    {
        private readonly ILogger<MultiSymbolAnalyzer> _logger;
        private readonly IPredictionDataService _predictionDataService;
        private readonly IMarketDataEnrichmentService _marketDataEnrichmentService;
        private readonly ITechnicalIndicatorService _technicalIndicatorService;

        // Scoring weights for composite score calculation
        private const double WeightConfidence = 0.30;
        private const double WeightRisk = 0.25;
        private const double WeightMomentum = 0.25;
        private const double WeightPotentialReturn = 0.20;

        // Batch processing configuration
        private const int MaxSymbolsPerBatch = 10;
        private const int BatchDelayMs = 100;

        public MultiSymbolAnalyzer(
            ILogger<MultiSymbolAnalyzer> logger,
            IPredictionDataService predictionDataService = null,
            IMarketDataEnrichmentService marketDataEnrichmentService = null,
            ITechnicalIndicatorService technicalIndicatorService = null)
        {
            _logger = logger;
            _predictionDataService = predictionDataService;
            _marketDataEnrichmentService = marketDataEnrichmentService;
            _technicalIndicatorService = technicalIndicatorService;
        }

        /// <inheritdoc/>
        public async Task<MultiSymbolComparisonResult> CompareSymbolsAsync(IEnumerable<string> symbols, bool includeHistoricalContext = true)
        {
            var result = new MultiSymbolComparisonResult
            {
                GeneratedAt = DateTime.UtcNow
            };

            if (symbols == null || !symbols.Any())
            {
                result.Errors.Add("No symbols provided for comparison");
                return result;
            }

            var symbolList = symbols
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .Select(s => s.ToUpperInvariant().Trim())
                .Distinct()
                .Take(MaxSymbolsPerBatch)
                .ToList();

            result.Symbols = symbolList;
            _logger?.LogInformation("Starting multi-symbol comparison for {Count} symbols: {Symbols}", 
                symbolList.Count, string.Join(", ", symbolList));

            bool usedCachedData = false;

            // Batch process symbols
            foreach (var symbol in symbolList)
            {
                try
                {
                    var symbolData = await FetchSymbolDataAsync(symbol, includeHistoricalContext);
                    
                    if (symbolData.IsCached)
                    {
                        usedCachedData = true;
                    }

                    result.SymbolData[symbol] = symbolData;

                    // Small delay between symbol lookups to avoid overwhelming services
                    if (symbolList.Count > 3)
                    {
                        await Task.Delay(BatchDelayMs);
                    }
                }
                catch (Exception ex)
                {
                    _logger?.LogError(ex, "Error fetching data for symbol {Symbol}", symbol);
                    result.SymbolData[symbol] = new SymbolAnalysisData
                    {
                        Symbol = symbol,
                        Error = ex.Message
                    };
                    result.Errors.Add($"Error fetching {symbol}: {ex.Message}");
                }
            }

            result.UsedCachedData = usedCachedData;

            // Calculate composite scores for all symbols with data
            result.CompositeScores = CalculateCompositeScores(result);

            // Update individual symbol scores
            foreach (var kvp in result.CompositeScores)
            {
                if (result.SymbolData.TryGetValue(kvp.Key, out var data))
                {
                    data.CompositeScore = kvp.Value;
                }
            }

            // Identify signal highlights
            result.Highlights = IdentifySignalHighlights(result);

            _logger?.LogInformation("Multi-symbol comparison completed. {SuccessCount}/{TotalCount} symbols successful", 
                result.SymbolData.Count(x => x.Value.HasData), symbolList.Count);

            return result;
        }

        /// <summary>
        /// Fetches all analysis data for a single symbol.
        /// </summary>
        private async Task<SymbolAnalysisData> FetchSymbolDataAsync(string symbol, bool includeHistoricalContext)
        {
            var data = new SymbolAnalysisData
            {
                Symbol = symbol
            };

            // Fetch prediction data
            if (_predictionDataService != null)
            {
                try
                {
                    var predictionResult = await _predictionDataService.GetPredictionContextWithCacheAsync(symbol);
                    if (predictionResult != null && !string.IsNullOrEmpty(predictionResult.Context))
                    {
                        ParsePredictionContext(predictionResult.Context, data);
                        data.IsCached = predictionResult.IsCached;
                        data.PredictionTimestamp = predictionResult.PredictionTimestamp;
                    }
                }
                catch (Exception ex)
                {
                    _logger?.LogWarning(ex, "Failed to fetch prediction for {Symbol}", symbol);
                }
            }

            // Fetch technical indicators
            if (_technicalIndicatorService != null)
            {
                try
                {
                    var indicators = await _technicalIndicatorService.GetIndicatorsForPrediction(symbol, "1day");
                    if (indicators != null && indicators.Count > 0)
                    {
                        data.Indicators = indicators;
                        CalculateRiskMetrics(data, indicators);
                    }
                }
                catch (Exception ex)
                {
                    _logger?.LogWarning(ex, "Failed to fetch indicators for {Symbol}", symbol);
                }
            }

            // Fetch historical context
            if (includeHistoricalContext && _marketDataEnrichmentService != null)
            {
                try
                {
                    var historicalContext = await _marketDataEnrichmentService.GetHistoricalContextAsync(symbol, 60);
                    if (!string.IsNullOrEmpty(historicalContext))
                    {
                        ParseHistoricalContext(historicalContext, data);
                    }
                }
                catch (Exception ex)
                {
                    _logger?.LogWarning(ex, "Failed to fetch historical context for {Symbol}", symbol);
                }
            }

            return data;
        }

        /// <summary>
        /// Parses prediction context string to extract key data.
        /// </summary>
        private void ParsePredictionContext(string context, SymbolAnalysisData data)
        {
            if (string.IsNullOrEmpty(context))
            {
                return;
            }

            var lines = context.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            foreach (var line in lines)
            {
                var trimmedLine = line.Trim().TrimStart('-', ' ');
                
                if (trimmedLine.StartsWith("Predicted Action:", StringComparison.OrdinalIgnoreCase))
                {
                    data.PredictedAction = ExtractValue(trimmedLine, "Predicted Action:").Trim();
                }
                else if (trimmedLine.StartsWith("Confidence:", StringComparison.OrdinalIgnoreCase))
                {
                    var confStr = ExtractValue(trimmedLine, "Confidence:");
                    if (TryParsePercentage(confStr, out double conf))
                    {
                        data.Confidence = conf;
                    }
                }
                else if (trimmedLine.StartsWith("Target Price:", StringComparison.OrdinalIgnoreCase))
                {
                    var priceStr = ExtractValue(trimmedLine, "Target Price:");
                    if (TryParsePrice(priceStr, out double price))
                    {
                        data.TargetPrice = price;
                    }
                }
                else if (trimmedLine.StartsWith("Current Price:", StringComparison.OrdinalIgnoreCase))
                {
                    var priceStr = ExtractValue(trimmedLine, "Current Price:");
                    if (TryParsePrice(priceStr, out double price))
                    {
                        data.CurrentPrice = price;
                    }
                }
                else if (trimmedLine.StartsWith("Potential Return:", StringComparison.OrdinalIgnoreCase))
                {
                    var returnStr = ExtractValue(trimmedLine, "Potential Return:");
                    if (TryParsePercentage(returnStr, out double ret))
                    {
                        data.PotentialReturn = ret;
                    }
                }
            }

            // Calculate potential return if not provided but we have current and target prices
            if (data.PotentialReturn == 0 && data.CurrentPrice > 0 && data.TargetPrice > 0)
            {
                data.PotentialReturn = (data.TargetPrice - data.CurrentPrice) / data.CurrentPrice;
            }
        }

        /// <summary>
        /// Parses historical context string to extract summary data.
        /// </summary>
        private void ParseHistoricalContext(string context, SymbolAnalysisData data)
        {
            if (string.IsNullOrEmpty(context))
            {
                return;
            }

            data.HistoricalContext = new HistoricalContextSummary();

            var lines = context.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            foreach (var line in lines)
            {
                var trimmedLine = line.Trim().TrimStart('-', ' ');

                // Parse various historical metrics from the context
                if (trimmedLine.Contains("5-day", StringComparison.OrdinalIgnoreCase) && trimmedLine.Contains("change", StringComparison.OrdinalIgnoreCase))
                {
                    if (TryParsePercentageFromText(trimmedLine, out double change))
                    {
                        data.HistoricalContext.FiveDayChange = change;
                    }
                }
                else if (trimmedLine.Contains("20-day", StringComparison.OrdinalIgnoreCase) && trimmedLine.Contains("change", StringComparison.OrdinalIgnoreCase))
                {
                    if (TryParsePercentageFromText(trimmedLine, out double change))
                    {
                        data.HistoricalContext.TwentyDayChange = change;
                    }
                }
                else if (trimmedLine.Contains("5-day MA", StringComparison.OrdinalIgnoreCase) || trimmedLine.Contains("SMA 5", StringComparison.OrdinalIgnoreCase))
                {
                    if (TryParsePriceFromText(trimmedLine, out double ma))
                    {
                        data.HistoricalContext.FiveDayMA = ma;
                    }
                }
                else if (trimmedLine.Contains("20-day MA", StringComparison.OrdinalIgnoreCase) || trimmedLine.Contains("SMA 20", StringComparison.OrdinalIgnoreCase))
                {
                    if (TryParsePriceFromText(trimmedLine, out double ma))
                    {
                        data.HistoricalContext.TwentyDayMA = ma;
                    }
                }
                else if (trimmedLine.Contains("50-day MA", StringComparison.OrdinalIgnoreCase) || trimmedLine.Contains("SMA 50", StringComparison.OrdinalIgnoreCase))
                {
                    if (TryParsePriceFromText(trimmedLine, out double ma))
                    {
                        data.HistoricalContext.FiftyDayMA = ma;
                    }
                }
                else if (trimmedLine.Contains("bullish", StringComparison.OrdinalIgnoreCase))
                {
                    data.HistoricalContext.TrendDirection = "bullish";
                }
                else if (trimmedLine.Contains("bearish", StringComparison.OrdinalIgnoreCase))
                {
                    data.HistoricalContext.TrendDirection = "bearish";
                }
            }

            // Calculate momentum score based on available data
            data.HistoricalContext.MomentumScore = CalculateMomentumScore(data);

            // Determine price vs MA
            if (data.CurrentPrice > 0 && data.HistoricalContext.TwentyDayMA > 0)
            {
                if (data.CurrentPrice > data.HistoricalContext.TwentyDayMA * 1.02)
                {
                    data.HistoricalContext.PriceVsMA = "above";
                }
                else if (data.CurrentPrice < data.HistoricalContext.TwentyDayMA * 0.98)
                {
                    data.HistoricalContext.PriceVsMA = "below";
                }
                else
                {
                    data.HistoricalContext.PriceVsMA = "at";
                }
            }
        }

        /// <summary>
        /// Calculates risk metrics from technical indicators.
        /// </summary>
        private void CalculateRiskMetrics(SymbolAnalysisData data, Dictionary<string, double> indicators)
        {
            data.RiskMetrics = new SymbolRiskMetrics();

            // Extract ATR if available
            if (indicators.TryGetValue("ATR", out double atr))
            {
                data.RiskMetrics.ATR = atr;
            }

            // Calculate risk score based on indicators
            double riskScore = 50; // Start at neutral

            // RSI extremes increase risk
            if (indicators.TryGetValue("RSI", out double rsi))
            {
                if (rsi > 70 || rsi < 30)
                {
                    riskScore += 15;
                }
                else if (rsi > 60 || rsi < 40)
                {
                    riskScore += 5;
                }
            }

            // High ATR increases risk
            if (atr > 0 && data.CurrentPrice > 0)
            {
                double atrPercent = atr / data.CurrentPrice * 100;
                if (atrPercent > 5)
                {
                    riskScore += 20;
                }
                else if (atrPercent > 3)
                {
                    riskScore += 10;
                }
            }

            // ADX indicates trend strength
            if (indicators.TryGetValue("ADX", out double adx))
            {
                if (adx > 40)
                {
                    riskScore += 10; // Strong trends can be riskier
                }
            }

            // Bound risk score
            data.RiskMetrics.RiskScore = Math.Max(0, Math.Min(100, riskScore));

            // Calculate volatility from ATR if possible
            if (atr > 0 && data.CurrentPrice > 0)
            {
                data.RiskMetrics.Volatility = atr / data.CurrentPrice * 100;
            }
        }

        /// <summary>
        /// Calculates momentum score from available data.
        /// </summary>
        private double CalculateMomentumScore(SymbolAnalysisData data)
        {
            double score = 0;

            if (data.HistoricalContext == null)
            {
                return score;
            }

            // Add points for positive price changes
            if (data.HistoricalContext.FiveDayChange > 0)
            {
                score += Math.Min(50, data.HistoricalContext.FiveDayChange * 100 * 10);
            }
            else
            {
                score += Math.Max(-50, data.HistoricalContext.FiveDayChange * 100 * 10);
            }

            // Add points for price above moving averages
            if (data.CurrentPrice > 0)
            {
                if (data.HistoricalContext.TwentyDayMA > 0 && data.CurrentPrice > data.HistoricalContext.TwentyDayMA)
                {
                    score += 15;
                }
                if (data.HistoricalContext.FiftyDayMA > 0 && data.CurrentPrice > data.HistoricalContext.FiftyDayMA)
                {
                    score += 10;
                }
            }

            // Consider RSI momentum
            if (data.Indicators != null && data.Indicators.TryGetValue("RSI", out double rsi))
            {
                if (rsi > 50 && rsi < 70)
                {
                    score += (rsi - 50) / 2; // Bullish momentum
                }
                else if (rsi < 50 && rsi > 30)
                {
                    score -= (50 - rsi) / 2; // Bearish momentum
                }
            }

            return Math.Max(-100, Math.Min(100, score));
        }

        /// <inheritdoc/>
        public Dictionary<string, double> CalculateCompositeScores(MultiSymbolComparisonResult comparisonResult)
        {
            var scores = new Dictionary<string, double>();

            if (comparisonResult?.SymbolData == null || comparisonResult.SymbolData.Count == 0)
            {
                return scores;
            }

            // Get max/min values for normalization
            var symbolsWithData = comparisonResult.SymbolData.Values.Where(s => s.HasData).ToList();
            if (symbolsWithData.Count == 0)
            {
                return scores;
            }

            double maxConfidence = symbolsWithData.Max(s => s.Confidence);
            double minConfidence = symbolsWithData.Min(s => s.Confidence);
            double maxRisk = symbolsWithData.Max(s => s.RiskMetrics?.RiskScore ?? 50);
            double minRisk = symbolsWithData.Min(s => s.RiskMetrics?.RiskScore ?? 50);
            double maxMomentum = symbolsWithData.Max(s => s.HistoricalContext?.MomentumScore ?? 0);
            double minMomentum = symbolsWithData.Min(s => s.HistoricalContext?.MomentumScore ?? 0);
            double maxReturn = symbolsWithData.Max(s => Math.Abs(s.PotentialReturn));

            foreach (var kvp in comparisonResult.SymbolData)
            {
                var data = kvp.Value;
                if (!data.HasData)
                {
                    continue;
                }

                // Normalize confidence (0-100)
                double normalizedConfidence = maxConfidence == minConfidence ? 50 : 
                    ((data.Confidence - minConfidence) / (maxConfidence - minConfidence)) * 100;

                // Normalize risk (inverted, lower risk = higher score)
                double riskScore = data.RiskMetrics?.RiskScore ?? 50;
                double normalizedRisk = maxRisk == minRisk ? 50 : 
                    100 - ((riskScore - minRisk) / (maxRisk - minRisk)) * 100;

                // Normalize momentum
                double momentum = data.HistoricalContext?.MomentumScore ?? 0;
                double normalizedMomentum = maxMomentum == minMomentum ? 50 : 
                    ((momentum - minMomentum) / (maxMomentum - minMomentum)) * 100;

                // Normalize potential return (consider direction)
                double normalizedReturn = maxReturn == 0 ? 50 : 
                    50 + (data.PotentialReturn / maxReturn) * 50;

                // Calculate weighted composite score
                double compositeScore = 
                    (normalizedConfidence * WeightConfidence) +
                    (normalizedRisk * WeightRisk) +
                    (normalizedMomentum * WeightMomentum) +
                    (normalizedReturn * WeightPotentialReturn);

                // Bonus for BUY signals, penalty for SELL
                if (data.PredictedAction?.ToUpperInvariant() == "BUY")
                {
                    compositeScore += 5;
                }
                else if (data.PredictedAction?.ToUpperInvariant() == "SELL")
                {
                    compositeScore -= 5;
                }

                scores[kvp.Key] = Math.Max(0, Math.Min(100, compositeScore));
            }

            return scores;
        }

        /// <inheritdoc/>
        public SignalHighlights IdentifySignalHighlights(MultiSymbolComparisonResult comparisonResult)
        {
            var highlights = new SignalHighlights();

            if (comparisonResult?.SymbolData == null || comparisonResult.SymbolData.Count == 0)
            {
                return highlights;
            }

            var symbolsWithData = comparisonResult.SymbolData.Values.Where(s => s.HasData).ToList();
            if (symbolsWithData.Count == 0)
            {
                return highlights;
            }

            // Find strongest bullish (BUY with highest confidence)
            var bullishSignals = symbolsWithData
                .Where(s => s.PredictedAction?.ToUpperInvariant() == "BUY")
                .OrderByDescending(s => s.Confidence)
                .FirstOrDefault();

            if (bullishSignals != null)
            {
                highlights.StrongestBullish = bullishSignals.Symbol;
                highlights.StrongestBullishReason = $"BUY signal with {bullishSignals.Confidence:P0} confidence, {bullishSignals.PotentialReturn:P1} potential return";
            }

            // Find strongest bearish (SELL with highest confidence)
            var bearishSignals = symbolsWithData
                .Where(s => s.PredictedAction?.ToUpperInvariant() == "SELL")
                .OrderByDescending(s => s.Confidence)
                .FirstOrDefault();

            if (bearishSignals != null)
            {
                highlights.StrongestBearish = bearishSignals.Symbol;
                highlights.StrongestBearishReason = $"SELL signal with {bearishSignals.Confidence:P0} confidence, {Math.Abs(bearishSignals.PotentialReturn):P1} potential downside";
            }

            // Highest confidence
            var highestConfidence = symbolsWithData.OrderByDescending(s => s.Confidence).First();
            highlights.HighestConfidence = highestConfidence.Symbol;
            highlights.HighestConfidenceValue = highestConfidence.Confidence;

            // Lowest/highest risk
            var byRisk = symbolsWithData
                .Where(s => s.RiskMetrics != null)
                .OrderBy(s => s.RiskMetrics.RiskScore)
                .ToList();

            if (byRisk.Count > 0)
            {
                highlights.LowestRisk = byRisk.First().Symbol;
                highlights.LowestRiskValue = byRisk.First().RiskMetrics.RiskScore;
                highlights.HighestRisk = byRisk.Last().Symbol;
                highlights.HighestRiskValue = byRisk.Last().RiskMetrics.RiskScore;
            }

            // Highest momentum
            var byMomentum = symbolsWithData
                .Where(s => s.HistoricalContext != null)
                .OrderByDescending(s => s.HistoricalContext.MomentumScore)
                .ToList();

            if (byMomentum.Count > 0)
            {
                highlights.HighestMomentum = byMomentum.First().Symbol;
                highlights.HighestMomentumValue = byMomentum.First().HistoricalContext.MomentumScore;
            }

            // Highest potential return
            var highestReturn = symbolsWithData.OrderByDescending(s => s.PotentialReturn).First();
            highlights.HighestPotentialReturn = highestReturn.Symbol;
            highlights.HighestPotentialReturnValue = highestReturn.PotentialReturn;

            // Recommended pick (highest composite score among BUY signals, or highest overall if no BUY)
            var recommended = comparisonResult.CompositeScores.OrderByDescending(kvp => kvp.Value).First();
            highlights.RecommendedPick = recommended.Key;
            
            var recommendedData = comparisonResult.SymbolData[recommended.Key];
            highlights.RecommendedPickReason = $"Composite score: {recommended.Value:F0}/100 ({recommendedData.PredictedAction}, {recommendedData.Confidence:P0} confidence)";

            return highlights;
        }

        /// <inheritdoc/>
        public string FormatComparisonAsMarkdown(MultiSymbolComparisonResult comparisonResult)
        {
            if (comparisonResult == null || !comparisonResult.IsSuccessful)
            {
                return "No comparison data available.";
            }

            var sb = new StringBuilder();

            // Header
            sb.AppendLine($"## Multi-Symbol Comparison Analysis");
            sb.AppendLine($"*Generated at {comparisonResult.GeneratedAt:yyyy-MM-dd HH:mm} UTC*");
            sb.AppendLine();

            // Main comparison table
            sb.AppendLine("### Prediction Comparison");
            sb.AppendLine("| Symbol | Action | Confidence | Target Price | Current Price | Potential Return | Score |");
            sb.AppendLine("|--------|--------|------------|--------------|---------------|------------------|-------|");

            foreach (var symbol in comparisonResult.Symbols)
            {
                if (comparisonResult.SymbolData.TryGetValue(symbol, out var data) && data.HasData)
                {
                    sb.AppendLine($"| {symbol} | {data.PredictedAction ?? "N/A"} | {data.Confidence:P0} | ${data.TargetPrice:F2} | ${data.CurrentPrice:F2} | {data.PotentialReturn:P1} | {data.CompositeScore:F0} |");
                }
                else
                {
                    sb.AppendLine($"| {symbol} | N/A | - | - | - | - | - |");
                }
            }
            sb.AppendLine();

            // Technical indicators table
            sb.AppendLine("### Technical Indicators");
            sb.AppendLine("| Symbol | RSI | MACD | ADX | ATR | Momentum |");
            sb.AppendLine("|--------|-----|------|-----|-----|----------|");

            foreach (var symbol in comparisonResult.Symbols)
            {
                if (comparisonResult.SymbolData.TryGetValue(symbol, out var data) && data.HasData)
                {
                    string rsi = data.Indicators.TryGetValue("RSI", out var r) ? r.ToString("F1") : "-";
                    string macd = data.Indicators.TryGetValue("MACD", out var m) ? m.ToString("F3") : "-";
                    string adx = data.Indicators.TryGetValue("ADX", out var a) ? a.ToString("F1") : "-";
                    string atr = data.RiskMetrics?.ATR > 0 ? data.RiskMetrics.ATR.ToString("F2") : "-";
                    string momentum = data.HistoricalContext?.MomentumScore != 0 ? data.HistoricalContext?.MomentumScore.ToString("F0") ?? "-" : "-";
                    
                    sb.AppendLine($"| {symbol} | {rsi} | {macd} | {adx} | {atr} | {momentum} |");
                }
                else
                {
                    sb.AppendLine($"| {symbol} | - | - | - | - | - |");
                }
            }
            sb.AppendLine();

            // Risk comparison table
            sb.AppendLine("### Risk Metrics");
            sb.AppendLine("| Symbol | Risk Score | Volatility | Trend |");
            sb.AppendLine("|--------|------------|------------|-------|");

            foreach (var symbol in comparisonResult.Symbols)
            {
                if (comparisonResult.SymbolData.TryGetValue(symbol, out var data) && data.HasData)
                {
                    string riskScore = data.RiskMetrics?.RiskScore > 0 ? data.RiskMetrics.RiskScore.ToString("F0") : "-";
                    string volatility = data.RiskMetrics?.Volatility > 0 ? $"{data.RiskMetrics.Volatility:F1}%" : "-";
                    string trend = data.HistoricalContext?.TrendDirection ?? "-";
                    
                    sb.AppendLine($"| {symbol} | {riskScore} | {volatility} | {trend} |");
                }
                else
                {
                    sb.AppendLine($"| {symbol} | - | - | - |");
                }
            }
            sb.AppendLine();

            // Signal highlights
            if (comparisonResult.Highlights != null)
            {
                sb.AppendLine("### Signal Highlights");

                if (!string.IsNullOrEmpty(comparisonResult.Highlights.StrongestBullish))
                {
                    sb.AppendLine($"🟢 **Strongest Bullish**: {comparisonResult.Highlights.StrongestBullish} - {comparisonResult.Highlights.StrongestBullishReason}");
                }

                if (!string.IsNullOrEmpty(comparisonResult.Highlights.StrongestBearish))
                {
                    sb.AppendLine($"🔴 **Strongest Bearish**: {comparisonResult.Highlights.StrongestBearish} - {comparisonResult.Highlights.StrongestBearishReason}");
                }

                sb.AppendLine($"📊 **Highest Confidence**: {comparisonResult.Highlights.HighestConfidence} ({comparisonResult.Highlights.HighestConfidenceValue:P0})");
                
                if (!string.IsNullOrEmpty(comparisonResult.Highlights.LowestRisk))
                {
                    sb.AppendLine($"🛡️ **Lowest Risk**: {comparisonResult.Highlights.LowestRisk} (score: {comparisonResult.Highlights.LowestRiskValue:F0})");
                }
                
                if (!string.IsNullOrEmpty(comparisonResult.Highlights.HighestMomentum))
                {
                    sb.AppendLine($"🚀 **Highest Momentum**: {comparisonResult.Highlights.HighestMomentum} ({comparisonResult.Highlights.HighestMomentumValue:F0})");
                }

                sb.AppendLine();
                sb.AppendLine($"⭐ **Recommended Pick**: {comparisonResult.Highlights.RecommendedPick}");
                sb.AppendLine($"   *{comparisonResult.Highlights.RecommendedPickReason}*");
            }

            return sb.ToString();
        }

        /// <inheritdoc/>
        public string GenerateAllocationRecommendations(MultiSymbolComparisonResult comparisonResult, string riskProfile = "moderate")
        {
            if (comparisonResult == null || !comparisonResult.IsSuccessful)
            {
                return "Insufficient data for allocation recommendations.";
            }

            var sb = new StringBuilder();
            sb.AppendLine("### Portfolio Allocation Suggestions");
            sb.AppendLine($"*Risk Profile: {riskProfile}*");
            sb.AppendLine();

            var symbolsWithData = comparisonResult.SymbolData.Values
                .Where(s => s.HasData && comparisonResult.CompositeScores.ContainsKey(s.Symbol))
                .OrderByDescending(s => comparisonResult.CompositeScores[s.Symbol])
                .ToList();

            if (symbolsWithData.Count == 0)
            {
                return "No symbols with sufficient data for allocation recommendations.";
            }

            // Calculate suggested allocations based on composite scores and risk profile
            double totalScore = symbolsWithData.Sum(s => comparisonResult.CompositeScores[s.Symbol]);
            
            // Adjust allocations based on risk profile
            double maxSingleAllocation = riskProfile.ToLowerInvariant() switch
            {
                "conservative" => 0.25,
                "aggressive" => 0.50,
                _ => 0.35
            };

            double minAllocation = riskProfile.ToLowerInvariant() switch
            {
                "conservative" => 0.10,
                "aggressive" => 0.05,
                _ => 0.10
            };

            sb.AppendLine("| Symbol | Suggested Allocation | Rationale |");
            sb.AppendLine("|--------|---------------------|-----------|");

            double runningTotal = 0;
            foreach (var data in symbolsWithData)
            {
                double rawAllocation = comparisonResult.CompositeScores[data.Symbol] / totalScore;
                double adjustedAllocation = Math.Max(minAllocation, Math.Min(maxSingleAllocation, rawAllocation));
                
                // Ensure we don't exceed 100%
                if (runningTotal + adjustedAllocation > 1.0)
                {
                    adjustedAllocation = 1.0 - runningTotal;
                }
                
                if (adjustedAllocation <= 0)
                {
                    continue;
                }

                runningTotal += adjustedAllocation;

                string rationale = BuildAllocationRationale(data, riskProfile);
                sb.AppendLine($"| {data.Symbol} | {adjustedAllocation:P0} | {rationale} |");
            }

            sb.AppendLine();

            // Add risk considerations
            sb.AppendLine("**Risk Considerations:**");
            
            if (!string.IsNullOrEmpty(comparisonResult.Highlights?.HighestRisk))
            {
                sb.AppendLine($"- {comparisonResult.Highlights.HighestRisk} has the highest risk score ({comparisonResult.Highlights.HighestRiskValue:F0}) - consider smaller position");
            }

            if (!string.IsNullOrEmpty(comparisonResult.Highlights?.StrongestBearish))
            {
                sb.AppendLine($"- {comparisonResult.Highlights.StrongestBearish} shows bearish signals - exercise caution or consider hedging");
            }

            sb.AppendLine("- These suggestions are based on ML predictions and should not be considered investment advice");
            sb.AppendLine("- Always conduct your own research and consider your personal financial situation");

            return sb.ToString();
        }

        /// <summary>
        /// Builds allocation rationale for a symbol.
        /// </summary>
        private string BuildAllocationRationale(SymbolAnalysisData data, string riskProfile)
        {
            var reasons = new List<string>();

            if (data.PredictedAction?.ToUpperInvariant() == "BUY" && data.Confidence > 0.7)
            {
                reasons.Add("Strong BUY signal");
            }
            else if (data.PredictedAction?.ToUpperInvariant() == "BUY")
            {
                reasons.Add("BUY signal");
            }

            if (data.HistoricalContext?.MomentumScore > 30)
            {
                reasons.Add("positive momentum");
            }

            if (data.RiskMetrics?.RiskScore < 40)
            {
                reasons.Add("low risk");
            }
            else if (data.RiskMetrics?.RiskScore > 70 && riskProfile.ToLowerInvariant() == "conservative")
            {
                reasons.Add("higher risk (reduced allocation)");
            }

            if (data.PotentialReturn > 0.05)
            {
                reasons.Add($"{data.PotentialReturn:P0} upside");
            }

            return reasons.Count > 0 ? string.Join(", ", reasons) : "Balanced exposure";
        }

        /// <inheritdoc/>
        public string BuildComparisonContext(MultiSymbolComparisonResult comparisonResult)
        {
            if (comparisonResult == null || !comparisonResult.IsSuccessful)
            {
                return string.Empty;
            }

            var sb = new StringBuilder();
            sb.AppendLine("Multi-Symbol Comparison Data:");
            sb.AppendLine();

            foreach (var symbol in comparisonResult.Symbols)
            {
                if (comparisonResult.SymbolData.TryGetValue(symbol, out var data) && data.HasData)
                {
                    sb.AppendLine($"{symbol}:");
                    sb.AppendLine($"  - Predicted Action: {data.PredictedAction}");
                    sb.AppendLine($"  - Confidence: {data.Confidence:P0}");
                    sb.AppendLine($"  - Target Price: ${data.TargetPrice:F2}");
                    sb.AppendLine($"  - Potential Return: {data.PotentialReturn:P1}");
                    sb.AppendLine($"  - Composite Score: {data.CompositeScore:F0}/100");

                    if (data.RiskMetrics != null)
                    {
                        sb.AppendLine($"  - Risk Score: {data.RiskMetrics.RiskScore:F0}/100");
                    }

                    if (data.HistoricalContext != null)
                    {
                        sb.AppendLine($"  - Momentum: {data.HistoricalContext.MomentumScore:F0}");
                        if (!string.IsNullOrEmpty(data.HistoricalContext.TrendDirection))
                        {
                            sb.AppendLine($"  - Trend: {data.HistoricalContext.TrendDirection}");
                        }
                    }

                    // Key indicators
                    if (data.Indicators != null && data.Indicators.Count > 0)
                    {
                        if (data.Indicators.TryGetValue("RSI", out var rsi))
                        {
                            sb.AppendLine($"  - RSI: {rsi:F1}");
                        }
                    }

                    sb.AppendLine();
                }
            }

            // Add highlights
            if (comparisonResult.Highlights != null)
            {
                sb.AppendLine("Signal Highlights:");
                if (!string.IsNullOrEmpty(comparisonResult.Highlights.RecommendedPick))
                {
                    sb.AppendLine($"  - Recommended Pick: {comparisonResult.Highlights.RecommendedPick} ({comparisonResult.Highlights.RecommendedPickReason})");
                }
                if (!string.IsNullOrEmpty(comparisonResult.Highlights.StrongestBullish))
                {
                    sb.AppendLine($"  - Strongest Bullish: {comparisonResult.Highlights.StrongestBullish}");
                }
                if (!string.IsNullOrEmpty(comparisonResult.Highlights.StrongestBearish))
                {
                    sb.AppendLine($"  - Strongest Bearish: {comparisonResult.Highlights.StrongestBearish}");
                }
            }

            return sb.ToString();
        }

        #region Helper Methods

        private static string ExtractValue(string line, string prefix)
        {
            int startIndex = line.IndexOf(prefix, StringComparison.OrdinalIgnoreCase);
            if (startIndex < 0)
            {
                return string.Empty;
            }

            return line.Substring(startIndex + prefix.Length).Trim();
        }

        private static bool TryParsePercentage(string value, out double result)
        {
            result = 0;
            if (string.IsNullOrWhiteSpace(value))
            {
                return false;
            }

            value = value.Trim().TrimEnd('%');
            if (double.TryParse(value, out result))
            {
                // If the value looks like it's already in percentage form (e.g., "75" for 75%)
                if (result > 1)
                {
                    result /= 100;
                }
                return true;
            }
            return false;
        }

        private static bool TryParsePrice(string value, out double result)
        {
            result = 0;
            if (string.IsNullOrWhiteSpace(value))
            {
                return false;
            }

            value = value.Trim().TrimStart('$').Replace(",", "");
            return double.TryParse(value, out result);
        }

        private static bool TryParsePercentageFromText(string text, out double result)
        {
            result = 0;
            
            // Look for patterns like "+5.2%" or "-3.1%" or "5.2%"
            var match = System.Text.RegularExpressions.Regex.Match(text, @"([+-]?\d+\.?\d*)%");
            if (match.Success)
            {
                if (double.TryParse(match.Groups[1].Value, out result))
                {
                    result /= 100; // Convert to decimal
                    return true;
                }
            }
            return false;
        }

        private static bool TryParsePriceFromText(string text, out double result)
        {
            result = 0;
            
            // Look for patterns like "$150.23" or "150.23"
            var match = System.Text.RegularExpressions.Regex.Match(text, @"\$?(\d+\.?\d*)");
            if (match.Success)
            {
                return double.TryParse(match.Groups[1].Value, out result);
            }
            return false;
        }

        #endregion
    }
}
