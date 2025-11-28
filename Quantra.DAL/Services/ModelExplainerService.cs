using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for generating plain English explanations of ML model predictions.
    /// Parses feature weights from Python models and translates technical indicators
    /// into language traders can understand without reading code (MarketChat story 4).
    /// </summary>
    public class ModelExplainerService : IModelExplainerService
    {
        private readonly ILogger<ModelExplainerService> _logger;

        // Technical indicator descriptions for plain English translation
        private static readonly Dictionary<string, IndicatorDescription> IndicatorDescriptions = new Dictionary<string, IndicatorDescription>(StringComparer.OrdinalIgnoreCase)
        {
            ["rsi"] = new IndicatorDescription(
                "Relative Strength Index (RSI)",
                "momentum indicator measuring speed and magnitude of price changes",
                "Below 30 suggests oversold (potential buying opportunity), above 70 suggests overbought (potential selling opportunity)"),
            ["macd"] = new IndicatorDescription(
                "Moving Average Convergence Divergence (MACD)",
                "trend-following momentum indicator showing relationship between two moving averages",
                "Positive values indicate bullish momentum, negative values indicate bearish momentum"),
            ["macd_signal"] = new IndicatorDescription(
                "MACD Signal Line",
                "9-day EMA of the MACD line used for trading signals",
                "When MACD crosses above the signal line, it suggests a bullish signal"),
            ["macd_histogram"] = new IndicatorDescription(
                "MACD Histogram",
                "difference between MACD and its signal line",
                "Growing histogram indicates strengthening momentum in the current direction"),
            ["vwap"] = new IndicatorDescription(
                "Volume-Weighted Average Price (VWAP)",
                "average price weighted by volume, representing the true average price",
                "Price above VWAP suggests bullish sentiment, below suggests bearish sentiment"),
            ["sma"] = new IndicatorDescription(
                "Simple Moving Average (SMA)",
                "average price over a specified period",
                "Price above SMA indicates uptrend, below indicates downtrend"),
            ["ema"] = new IndicatorDescription(
                "Exponential Moving Average (EMA)",
                "weighted moving average giving more weight to recent prices",
                "More responsive to recent price changes than SMA"),
            ["bb_upper"] = new IndicatorDescription(
                "Bollinger Band Upper",
                "upper volatility band 2 standard deviations above the 20-day SMA",
                "Price touching upper band may indicate overbought conditions"),
            ["bb_lower"] = new IndicatorDescription(
                "Bollinger Band Lower",
                "lower volatility band 2 standard deviations below the 20-day SMA",
                "Price touching lower band may indicate oversold conditions"),
            ["bb_width"] = new IndicatorDescription(
                "Bollinger Band Width",
                "measure of price volatility (distance between upper and lower bands)",
                "Narrow bands indicate low volatility, wide bands indicate high volatility"),
            ["atr"] = new IndicatorDescription(
                "Average True Range (ATR)",
                "measure of price volatility over a period",
                "Higher ATR indicates more volatile price action"),
            ["adx"] = new IndicatorDescription(
                "Average Directional Index (ADX)",
                "measures trend strength regardless of direction",
                "Above 25 indicates strong trend, below 20 indicates weak or no trend"),
            ["cci"] = new IndicatorDescription(
                "Commodity Channel Index (CCI)",
                "measures deviation from statistical mean",
                "Above +100 suggests overbought, below -100 suggests oversold"),
            ["mfi"] = new IndicatorDescription(
                "Money Flow Index (MFI)",
                "volume-weighted RSI measuring buying and selling pressure",
                "Above 80 indicates overbought, below 20 indicates oversold"),
            ["stoch_k"] = new IndicatorDescription(
                "Stochastic Oscillator %K",
                "compares closing price to price range over a period",
                "Above 80 indicates overbought, below 20 indicates oversold"),
            ["stoch_d"] = new IndicatorDescription(
                "Stochastic Oscillator %D",
                "3-day SMA of %K, used for signal confirmation",
                "Crossovers of %K above %D suggest bullish momentum"),
            ["obv"] = new IndicatorDescription(
                "On-Balance Volume (OBV)",
                "cumulative volume indicator showing buying/selling pressure",
                "Rising OBV confirms uptrend, falling OBV confirms downtrend"),
            ["volume"] = new IndicatorDescription(
                "Trading Volume",
                "number of shares traded",
                "High volume confirms price movements, low volume suggests weak conviction"),
            ["volatility"] = new IndicatorDescription(
                "Price Volatility",
                "measure of price fluctuation over time",
                "Higher volatility indicates greater risk and potential reward"),
            ["momentum"] = new IndicatorDescription(
                "Price Momentum",
                "rate of price change over a specified period",
                "Positive momentum indicates upward price pressure, negative indicates downward"),
            ["roc"] = new IndicatorDescription(
                "Rate of Change (ROC)",
                "percentage change in price over a period",
                "Positive ROC indicates uptrend, negative indicates downtrend"),
            ["returns"] = new IndicatorDescription(
                "Price Returns",
                "percentage change in price",
                "Recent returns influence model expectations for future price movement")
        };

        /// <summary>
        /// Constructor for ModelExplainerService
        /// </summary>
        public ModelExplainerService(ILogger<ModelExplainerService> logger = null)
        {
            _logger = logger;
        }

        /// <inheritdoc/>
        public string ExplainPredictionFactors(PredictionResult prediction)
        {
            if (prediction == null)
            {
                return "No prediction data available to explain.";
            }

            var builder = new StringBuilder();
            builder.AppendLine($"**Prediction Explanation for {prediction.Symbol ?? "Unknown Symbol"}**");
            builder.AppendLine();

            // Action and target price explanation
            builder.AppendLine(ExplainActionAndTarget(prediction));
            builder.AppendLine();

            // Confidence explanation
            builder.AppendLine(ExplainConfidenceScore(prediction.Confidence, prediction.ModelType));
            builder.AppendLine();

            // Top influential factors
            if (prediction.FeatureWeights != null && prediction.FeatureWeights.Count > 0)
            {
                builder.AppendLine("**Key Factors Driving This Prediction:**");
                builder.AppendLine(GetTopInfluentialFactors(prediction.FeatureWeights, 5));
                builder.AppendLine();
            }

            // Risk metrics explanation
            if (prediction.RiskMetrics != null)
            {
                builder.AppendLine("**Risk Assessment:**");
                builder.AppendLine(ExplainRiskMetrics(prediction.RiskMetrics));
            }

            _logger?.LogInformation("Generated explanation for {Symbol} prediction with {FactorCount} factors",
                prediction.Symbol, prediction.FeatureWeights?.Count ?? 0);

            return builder.ToString();
        }

        /// <inheritdoc/>
        public string ExplainConfidenceScore(double confidence, string modelType)
        {
            var confidencePercent = confidence * 100;
            var builder = new StringBuilder();

            // Determine confidence level category
            string confidenceLevel;
            string interpretation;

            if (confidence >= 0.85)
            {
                confidenceLevel = "very high";
                interpretation = "The model has strong conviction in this prediction based on multiple confirming signals.";
            }
            else if (confidence >= 0.70)
            {
                confidenceLevel = "high";
                interpretation = "The model shows solid confidence with most indicators aligning.";
            }
            else if (confidence >= 0.55)
            {
                confidenceLevel = "moderate";
                interpretation = "The signals are mixed, suggesting some uncertainty in the prediction.";
            }
            else if (confidence >= 0.40)
            {
                confidenceLevel = "low";
                interpretation = "The model sees conflicting signals and has limited conviction.";
            }
            else
            {
                confidenceLevel = "very low";
                interpretation = "The signals are highly uncertain or contradictory.";
            }

            builder.AppendLine($"**Confidence Score: {confidencePercent:F1}% ({confidenceLevel})**");
            builder.AppendLine(interpretation);

            // Add model-specific context
            if (!string.IsNullOrEmpty(modelType))
            {
                builder.AppendLine();
                builder.AppendLine(GetModelTypeExplanation(modelType));
            }

            return builder.ToString();
        }

        /// <inheritdoc/>
        public string ExplainRiskMetrics(Quantra.Models.RiskMetrics riskMetrics)
        {
            if (riskMetrics == null)
            {
                return "No risk metrics available.";
            }

            var builder = new StringBuilder();

            // Value at Risk (VaR)
            builder.AppendLine($"• **Value at Risk (VaR):** ${riskMetrics.ValueAtRisk:F2}");
            builder.AppendLine($"  This is the maximum expected loss at 95% confidence - there's a 5% chance losses could exceed this amount.");
            builder.AppendLine();

            // Maximum Drawdown
            // Note: MaxDrawdown is expected to be positive (Python model applies abs() before sending)
            builder.AppendLine($"• **Maximum Drawdown:** ${riskMetrics.MaxDrawdown:F2}");
            builder.AppendLine($"  This represents the worst-case decline from peak to trough that could occur.");
            builder.AppendLine();

            // Sharpe Ratio
            builder.AppendLine($"• **Sharpe Ratio:** {riskMetrics.SharpeRatio:F2}");
            string sharpeInterpretation = riskMetrics.SharpeRatio switch
            {
                >= 2.0 => "Excellent risk-adjusted returns - significantly outperforming on a risk-adjusted basis.",
                >= 1.0 => "Good risk-adjusted returns - acceptable compensation for risk taken.",
                >= 0.5 => "Moderate risk-adjusted returns - returns are reasonable given the risk.",
                >= 0 => "Low risk-adjusted returns - risk may not be adequately compensated.",
                _ => "Negative risk-adjusted returns - the expected return does not justify the risk."
            };
            builder.AppendLine($"  {sharpeInterpretation}");
            builder.AppendLine();

            // Overall Risk Score
            builder.AppendLine($"• **Overall Risk Score:** {riskMetrics.RiskScore:P0}");
            string riskInterpretation = riskMetrics.RiskScore switch
            {
                >= 0.8 => "Very high risk - only suitable for aggressive traders with significant risk tolerance.",
                >= 0.6 => "High risk - proceed with caution and use proper position sizing.",
                >= 0.4 => "Moderate risk - reasonable risk level for most trading strategies.",
                >= 0.2 => "Low risk - relatively safe with limited downside exposure.",
                _ => "Very low risk - minimal risk exposure expected."
            };
            builder.AppendLine($"  {riskInterpretation}");

            return builder.ToString();
        }

        /// <inheritdoc/>
        public string TranslateIndicatorToPlainEnglish(string indicatorName, double weight, double? value = null)
        {
            if (string.IsNullOrWhiteSpace(indicatorName))
            {
                return string.Empty;
            }

            // Normalize indicator name for lookup
            var normalizedName = NormalizeIndicatorName(indicatorName);

            // Get weight classification
            string weightLevel;
            if (weight >= 0.30)
                weightLevel = "very significant";
            else if (weight >= 0.20)
                weightLevel = "significant";
            else if (weight >= 0.10)
                weightLevel = "moderate";
            else if (weight >= 0.05)
                weightLevel = "minor";
            else
                weightLevel = "minimal";

            var builder = new StringBuilder();

            if (IndicatorDescriptions.TryGetValue(normalizedName, out var description))
            {
                builder.Append($"**{description.Name}** (weight: {weight:P0}): ");
                builder.Append($"A {description.Description}. ");
                
                if (value.HasValue)
                {
                    builder.Append($"Current value: {value.Value:F2}. ");
                }
                
                builder.Append($"{description.Interpretation} ");
                builder.Append($"This indicator has a {weightLevel} influence on the prediction.");
            }
            else
            {
                // Generic description for unknown indicators
                builder.Append($"**{indicatorName}** (weight: {weight:P0}): ");
                
                if (value.HasValue)
                {
                    builder.Append($"Current value: {value.Value:F2}. ");
                }
                
                builder.Append($"This feature has a {weightLevel} influence on the model's prediction.");
            }

            return builder.ToString();
        }

        /// <inheritdoc/>
        public string CompareModelPredictions(Dictionary<string, PredictionResult> predictions)
        {
            if (predictions == null || predictions.Count == 0)
            {
                return "No model predictions available for comparison.";
            }

            if (predictions.Count == 1)
            {
                return $"Only one model prediction available ({predictions.Keys.First()}). No comparison possible.";
            }

            var builder = new StringBuilder();
            builder.AppendLine("**Model Comparison:**");
            builder.AppendLine();

            // Summarize each model's prediction
            foreach (var kvp in predictions.OrderByDescending(p => p.Value?.Confidence ?? 0))
            {
                var modelType = kvp.Key;
                var prediction = kvp.Value;

                if (prediction == null) continue;

                builder.AppendLine($"• **{GetModelDisplayName(modelType)}:**");
                builder.AppendLine($"  - Action: {prediction.Action}");
                builder.AppendLine($"  - Confidence: {prediction.Confidence:P0}");
                builder.AppendLine($"  - Target Price: ${prediction.TargetPrice:F2}");
                builder.AppendLine();
            }

            // Check for consensus
            var actions = predictions.Values.Where(p => p != null).Select(p => p.Action).Distinct().ToList();
            if (actions.Count == 1)
            {
                builder.AppendLine($"✓ **Consensus:** All models agree on a **{actions.First()}** signal.");
            }
            else
            {
                builder.AppendLine($"⚠ **Mixed Signals:** Models disagree ({string.Join(", ", actions)}). Consider waiting for clearer consensus.");
            }

            // Identify the highest confidence model
            var highestConfidence = predictions.OrderByDescending(p => p.Value?.Confidence ?? 0).First();
            builder.AppendLine();
            builder.AppendLine($"The **{GetModelDisplayName(highestConfidence.Key)}** model has the highest confidence ({highestConfidence.Value?.Confidence:P0}).");

            return builder.ToString();
        }

        /// <inheritdoc/>
        public string GetTopInfluentialFactors(Dictionary<string, double> featureWeights, int topN = 3)
        {
            if (featureWeights == null || featureWeights.Count == 0)
            {
                return "No feature weight data available.";
            }

            var builder = new StringBuilder();
            var topFactors = featureWeights
                .OrderByDescending(f => Math.Abs(f.Value))
                .Take(topN)
                .ToList();

            int rank = 1;
            foreach (var factor in topFactors)
            {
                builder.AppendLine($"{rank}. {TranslateIndicatorToPlainEnglish(factor.Key, factor.Value)}");
                builder.AppendLine();
                rank++;
            }

            return builder.ToString();
        }

        /// <summary>
        /// Generates an explanation of the predicted action and target price.
        /// </summary>
        private string ExplainActionAndTarget(PredictionResult prediction)
        {
            var builder = new StringBuilder();
            var action = prediction.Action?.ToUpperInvariant() ?? "HOLD";

            switch (action)
            {
                case "BUY":
                    builder.AppendLine($"The model recommends a **BUY** position.");
                    if (prediction.TargetPrice > 0 && prediction.CurrentPrice > 0)
                    {
                        var potentialReturn = (prediction.TargetPrice - prediction.CurrentPrice) / prediction.CurrentPrice;
                        builder.AppendLine($"Target price: **${prediction.TargetPrice:F2}** (potential gain of {potentialReturn:P1} from current ${prediction.CurrentPrice:F2})");
                    }
                    else if (prediction.TargetPrice > 0)
                    {
                        builder.AppendLine($"Target price: **${prediction.TargetPrice:F2}**");
                    }
                    break;

                case "SELL":
                    builder.AppendLine($"The model recommends a **SELL** position.");
                    if (prediction.TargetPrice > 0 && prediction.CurrentPrice > 0)
                    {
                        var potentialDecline = (prediction.CurrentPrice - prediction.TargetPrice) / prediction.CurrentPrice;
                        builder.AppendLine($"Target price: **${prediction.TargetPrice:F2}** (expected decline of {potentialDecline:P1} from current ${prediction.CurrentPrice:F2})");
                    }
                    else if (prediction.TargetPrice > 0)
                    {
                        builder.AppendLine($"Target price: **${prediction.TargetPrice:F2}**");
                    }
                    break;

                default:
                    builder.AppendLine($"The model recommends **HOLD** - no clear trading signal at this time.");
                    builder.AppendLine("The indicators are not strongly aligned in either direction, suggesting it's best to wait for clearer conditions.");
                    break;
            }

            return builder.ToString();
        }

        /// <summary>
        /// Gets a human-readable explanation for the model type.
        /// </summary>
        private string GetModelTypeExplanation(string modelType)
        {
            return modelType?.ToLowerInvariant() switch
            {
                "pytorch" or "pytorch lstm" or "pytorch gru" or "pytorch transformer" =>
                    "This prediction comes from a PyTorch deep learning model that uses neural networks to identify complex patterns in historical price data and technical indicators.",
                
                "tensorflow" or "tensorflow lstm" or "tensorflow gru" or "tensorflow transformer" =>
                    "This prediction comes from a TensorFlow deep learning model that leverages advanced neural network architectures to forecast price movements.",
                
                "random_forest" or "randomforest" =>
                    "This prediction comes from a Random Forest ensemble model that combines multiple decision trees to create robust predictions less prone to overfitting.",
                
                "ensemble" =>
                    "This prediction combines multiple model types (neural networks and ensemble methods) to provide a more balanced and robust forecast.",
                
                _ => $"This prediction comes from the {modelType ?? "default"} model."
            };
        }

        /// <summary>
        /// Gets a display-friendly name for the model type.
        /// </summary>
        private string GetModelDisplayName(string modelType)
        {
            return modelType?.ToLowerInvariant() switch
            {
                "pytorch" => "PyTorch Neural Network",
                "pytorch lstm" => "PyTorch LSTM",
                "pytorch gru" => "PyTorch GRU",
                "pytorch transformer" => "PyTorch Transformer",
                "tensorflow" => "TensorFlow Neural Network",
                "tensorflow lstm" => "TensorFlow LSTM",
                "tensorflow gru" => "TensorFlow GRU",
                "tensorflow transformer" => "TensorFlow Transformer",
                "random_forest" or "randomforest" => "Random Forest",
                "ensemble" => "Ensemble Model",
                _ => modelType ?? "Unknown Model"
            };
        }

        /// <summary>
        /// Normalizes indicator names for dictionary lookup.
        /// </summary>
        private string NormalizeIndicatorName(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return string.Empty;

            // Remove common prefixes and suffixes
            var normalized = name.ToLowerInvariant()
                .Replace("feature_", "")
                .Replace("_value", "")
                .Replace("indicator_", "")
                .Trim();

            // Map common variations to standard names
            return normalized switch
            {
                "sma_5" or "sma_20" or "sma_50" or "sma_200" => "sma",
                "ema_5" or "ema_20" or "ema_50" or "ema_200" => "ema",
                "rsi_14" or "rsi14" => "rsi",
                "macd_line" or "macd_value" => "macd",
                "signal_line" or "macdsignal" => "macd_signal",
                "histogram" or "macdhistogram" => "macd_histogram",
                "stochastic_k" or "stochk" or "%k" => "stoch_k",
                "stochastic_d" or "stochd" or "%d" => "stoch_d",
                "bollinger_upper" or "bbupper" => "bb_upper",
                "bollinger_lower" or "bblower" => "bb_lower",
                "bollinger_width" or "bbwidth" => "bb_width",
                "volume_ma5" or "volume_ma20" or "volume_ratio" => "volume",
                _ => normalized
            };
        }

        /// <summary>
        /// Represents a description of a technical indicator for plain English translation.
        /// </summary>
        private class IndicatorDescription
        {
            public string Name { get; }
            public string Description { get; }
            public string Interpretation { get; }

            public IndicatorDescription(string name, string description, string interpretation)
            {
                Name = name;
                Description = description;
                Interpretation = interpretation;
            }
        }
    }
}
