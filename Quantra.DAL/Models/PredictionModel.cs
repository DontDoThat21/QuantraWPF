using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Media;
using Quantra.DAL.Models;

namespace Quantra.Models
{
    /// <summary>
    /// Standard prediction model used throughout the application for stock prediction data
    /// </summary>
    public class PredictionModel
    {
        // Basic prediction info
        public string Symbol { get; set; }
        public string PredictedAction { get; set; } // "BUY", "SELL", "HOLD"
        public double Confidence { get; set; } // 0.0 to 1.0
        public double CurrentPrice { get; set; }
        public double TargetPrice { get; set; }
        public DateTime PredictionDate { get; set; } = DateTime.Now;

        // Time series prediction data
        public List<double> PricePredictions { get; set; } = new(); // Future price points
        public List<DateTime> PredictionTimePoints { get; set; } = new(); // Corresponding time points

        // Risk metrics
        public double RiskScore { get; set; } // 0.0 to 1.0, higher means riskier
        public double ValueAtRisk { get; set; } // 95% VaR
        public double MaxDrawdown { get; set; } // Maximum expected drawdown
        public double SharpeRatio { get; set; } // Risk-adjusted return metric
        public double? Volatility { get; set; } // Price volatility as a percentage

        // Prediction quality metrics
        public double PredictionAccuracy { get; set; } // Historical accuracy for this symbol
        public double ErrorMargin { get; set; } // Average error margin on price predictions
        public DateTime LastVerifiedDate { get; set; } // Last time prediction was verified
        public int ConsecutiveCorrectPredictions { get; set; } // Streak of correct predictions

        // Potential return property made read-write to support UI scenarios
        private double? _potentialReturn;
        public double PotentialReturn
        {
            get
            {
                if (_potentialReturn.HasValue)
                    return _potentialReturn.Value;
                if (CurrentPrice != 0)
                    return (TargetPrice - CurrentPrice) / CurrentPrice;
                return 0;
            }
            set => _potentialReturn = value;
        }

        // Additional context information
        public string Notes { get; set; }

        // Analysis details for UI and reporting
        public string AnalysisDetails { get; set; } = string.Empty;

        // OpenAI integration properties
        public double OpenAISentiment { get; set; } = 0;
        public bool UsesOpenAI { get; set; } = false;
        public string OpenAIExplanation { get; set; } = string.Empty;

        // Trading rule association
        public string TradingRule { get; set; }

        // Technical indicators used for the prediction
        public Dictionary<string, double> Indicators { get; set; } = new Dictionary<string, double>();

        // Technical patterns detected
        public List<TechnicalPattern> DetectedPatterns { get; set; } = new();

        // Aggregation metadata
        public string AggregationMethod { get; set; }
        public Dictionary<string, double> ModelWeights { get; set; } = new(); // Weights of different models in aggregation

        // Market conditions during analysis
        public MarketConditions MarketContext { get; set; } = new MarketConditions();

        // Normalized signal strength (across all algorithms)
        public double NormalizedSignalStrength { get; set; }

        // Helper properties for UI binding
        public Brush SignalColor => PredictedAction switch
        {
            "BUY" => new SolidColorBrush(Color.FromRgb(32, 192, 64)), // Green
            "SELL" => new SolidColorBrush(Color.FromRgb(192, 32, 32)), // Red
            _ => new SolidColorBrush(Color.FromRgb(192, 192, 32)) // Yellow
        };

        public string FormattedCurrentPrice => $"${CurrentPrice:F2}";
        public string SignalSummary => $"{Symbol} - {PredictedAction} @ {Confidence:P0}";
        public string RiskSummary => $"Risk: {RiskScore:P0} | VaR: ${ValueAtRisk:F2} | Sharpe: {SharpeRatio:F2}";

        /// <summary>
        /// Calculates potential return based on target price vs current price.
        /// Positive return indicates price expected to rise (BUY signal).
        /// Negative return indicates price expected to fall (SELL signal).
        /// </summary>
        public void CalculatePotentialReturn()
        {
            if (CurrentPrice != 0)
            {
                PotentialReturn = (TargetPrice - CurrentPrice) / CurrentPrice;
            }
            else
            {
                PotentialReturn = 0;
            }
        }

        // Feature importances/weights from ML model (Python interop)
        public Dictionary<string, double> FeatureWeights { get; set; } = new Dictionary<string, double>();

        // Real-time inference specific properties
        public string RequestId { get; set; } // Request ID for tracking real-time predictions
        public double InferenceTimeMs { get; set; } // Time taken for ML inference in milliseconds
        public string ModelType { get; set; } // Type of ML model used (e.g., "ensemble", "random_forest")
        public bool IsRealTime { get; set; } // Flag indicating if this is a real-time prediction
        public string Error { get; set; } // Any error that occurred during prediction
        public DateTime InferenceTimestamp { get; set; } = DateTime.Now; // When the inference was made
    }

    public class TechnicalPattern
    {
        public string PatternName { get; set; } // e.g., "Double Bottom", "Head and Shoulders"
        public double PatternStrength { get; set; } // 0.0 to 1.0
        public string ExpectedOutcome { get; set; } // "Bullish" or "Bearish"
        public DateTime DetectionDate { get; set; }
        public double HistoricalAccuracy { get; set; } // Historical accuracy of this pattern
    }

    /// <summary>
    /// Extension methods for the PredictionModel class
    /// </summary>
    public static class PredictionModelExtensions
    {
        /// <summary>
        /// Calculates predicted trend direction based on indicator values and technical patterns
        /// </summary>
        /// <param name="model">The prediction model to analyze</param>
        /// <returns>Trend direction as string: "Up", "Down", or "Neutral"</returns>
        public static string CalculateTrendDirection(this PredictionModel model)
        {
            if (model?.Indicators == null || model.Indicators.Count == 0)
                return "Neutral";

            int bullishSignals = 0;
            int bearishSignals = 0;

            // Technical pattern analysis
            if (model.DetectedPatterns?.Any() == true)
            {
                foreach (var pattern in model.DetectedPatterns)
                {
                    if (pattern.ExpectedOutcome == "Bullish" && pattern.PatternStrength > 0.6)
                        bullishSignals++;
                    else if (pattern.ExpectedOutcome == "Bearish" && pattern.PatternStrength > 0.6)
                        bearishSignals++;
                }
            }

            // RSI rules
            if (model.Indicators.TryGetValue("RSI", out double rsi))
            {
                if (rsi < 30) bullishSignals++;
                else if (rsi > 70) bearishSignals++;
            }

            // MACD rules
            if (model.Indicators.TryGetValue("MACD", out double macd) &&
                model.Indicators.TryGetValue("MACDSignal", out double macdSignal))
            {
                if (macd > macdSignal) bullishSignals++;
                else if (macd < macdSignal) bearishSignals++;
            }

            // Price vs VWAP
            if (model.Indicators.TryGetValue("VWAP", out double vwap))
            {
                if (model.CurrentPrice > vwap) bullishSignals++;
                else if (model.CurrentPrice < vwap) bearishSignals++;
            }

            // ADX trend strength
            if (model.Indicators.TryGetValue("ADX", out double adx))
            {
                if (adx > 25)
                {
                    if (model.PredictedAction == "BUY") bullishSignals++;
                    else if (model.PredictedAction == "SELL") bearishSignals++;
                }
            }

            // Market context analysis
            if (model.MarketContext != null)
            {
                // Use double-based MarketTrend: >0.3 bullish, <-0.3 bearish
                if (model.MarketContext.MarketTrend > 0.3)
                    bullishSignals++;
                else if (model.MarketContext.MarketTrend < -0.3)
                    bearishSignals++;
            }

            // Compare signals with stronger threshold for market conditions
            if (bullishSignals > bearishSignals + 2) return "Up";
            if (bearishSignals > bullishSignals + 2) return "Down";
            return "Neutral";
        }

        /// <summary>
        /// Estimates the signal strength based on available indicators, patterns, and market context
        /// </summary>
        /// <param name="model">The prediction model to analyze</param>
        /// <returns>Signal strength value from 0.0 to 1.0</returns>
        public static double EstimateSignalStrength(this PredictionModel model)
        {
            if (model?.Indicators == null || model.Indicators.Count == 0)
                return 0.5;

            double signalStrength = 0.5;
            double totalWeight = 0;

            // Technical indicator analysis
            if (model.Indicators.TryGetValue("RSI", out double rsi))
            {
                double rsiWeight = 0.2;
                if (model.PredictedAction == "BUY")
                {
                    if (rsi < 30)
                        signalStrength += (30 - rsi) / 30.0 * rsiWeight * 2;
                    else
                        signalStrength += (50 - rsi) / 50.0 * rsiWeight;
                }
                else if (model.PredictedAction == "SELL")
                {
                    if (rsi > 70)
                        signalStrength += (rsi - 70) / 30.0 * rsiWeight * 2;
                    else
                        signalStrength += (rsi - 50) / 50.0 * rsiWeight;
                }
                totalWeight += rsiWeight;
            }

            // Technical pattern analysis
            if (model.DetectedPatterns?.Any() == true)
            {
                double patternWeight = 0.3;
                double patternScore = model.DetectedPatterns
                    .Where(p => p.ExpectedOutcome == (model.PredictedAction == "BUY" ? "Bullish" : "Bearish"))
                    .Sum(p => p.PatternStrength * p.HistoricalAccuracy);

                signalStrength += patternScore * patternWeight;
                totalWeight += patternWeight;
            }

            // Market context analysis
            if (model.MarketContext != null)
            {
                double contextWeight = 0.25;
                double contextScore = 0.5; // Neutral base

                // Analyze market conditions
                if (model.PredictedAction == "BUY")
                {
                    if (model.MarketContext.MarketTrend > 0.3)
                        contextScore += 0.2;
                }
                else if (model.PredictedAction == "SELL")
                {
                    if (model.MarketContext.MarketTrend < -0.3)
                        contextScore += 0.2;
                }

                signalStrength += contextScore * contextWeight;
                totalWeight += contextWeight;
            }

            // Normalize the signal strength
            if (totalWeight > 0)
                signalStrength = signalStrength / totalWeight;

            // Consider prediction quality
            if (model.PredictionAccuracy > 0)
                signalStrength *= (0.5 + model.PredictionAccuracy * 0.5);

            return Math.Min(1.0, Math.Max(0.0, signalStrength));
        }

        /// <summary>
        /// Determines if the prediction model signals a strong algorithmic trading opportunity
        /// </summary>
        /// <param name="model">The prediction model to analyze</param>
        /// <param name="confidenceThreshold">Minimum confidence threshold (0.0-1.0)</param>
        /// <returns>True if the signal is strong enough for algorithmic trading</returns>
        public static bool IsAlgorithmicTradingSignal(this PredictionModel model, double confidenceThreshold = 0.75)
        {
            if (model == null)
                return false;

            // Enhanced confidence check including prediction accuracy
            double adjustedConfidence = model.Confidence * (0.7 + 0.3 * model.PredictionAccuracy);
            if (adjustedConfidence < confidenceThreshold)
                return false;

            // Risk assessment
            if (model.RiskScore > 0.8 || model.ValueAtRisk > model.PotentialReturn * 0.5)
                return false;

            // Technical analysis consensus
            string trend = CalculateTrendDirection(model);
            bool trendAligned = (model.PredictedAction == "BUY" && trend == "Up") ||
                              (model.PredictedAction == "SELL" && trend == "Down");

            if (!trendAligned)
                return false;

            // Market conditions check
            if (model.MarketContext != null)
            {
                // Avoid trading in extreme market conditions (use VolatilityIndex if available)
                if (model.MarketContext.VolatilityIndex > 35) // High VIX
                    return false;

                // Check if market trend aligns with prediction
                bool marketAligned = (model.PredictedAction == "BUY" && model.MarketContext.MarketTrend > 0.3) ||
                                     (model.PredictedAction == "SELL" && model.MarketContext.MarketTrend < -0.3);

                if (!marketAligned)
                    return false;
            }

            // Return potential and pattern confirmation
            if (Math.Abs(model.PotentialReturn) >= 0.10 && // 10% or greater potential
                model.DetectedPatterns?.Any(p => p.PatternStrength > 0.7 &&
                    p.ExpectedOutcome == (model.PredictedAction == "BUY" ? "Bullish" : "Bearish")) == true)
            {
                return true;
            }

            // Strong signal with good prediction history
            if (model.ConsecutiveCorrectPredictions >= 3 &&
                model.PredictionAccuracy > 0.7 &&
                Math.Abs(model.PotentialReturn) >= 0.05)
            {
                return true;
            }

            return false;
        }

        /// <summary>
        /// Analyzes prediction quality based on historical performance
        /// </summary>
        /// <param name="model">The prediction model to analyze</param>
        /// <returns>Quality score from 0.0 to 1.0</returns>
        public static double AnalyzePredictionQuality(this PredictionModel model)
        {
            if (model == null)
                return 0.0;

            double qualityScore = 0.5; // Base score
            int factorCount = 1;

            // Historical accuracy weight
            if (model.PredictionAccuracy > 0)
            {
                qualityScore += model.PredictionAccuracy * 0.3;
                factorCount++;
            }

            // Prediction streak consideration
            if (model.ConsecutiveCorrectPredictions > 0)
            {
                qualityScore += Math.Min(model.ConsecutiveCorrectPredictions * 0.1, 0.3);
                factorCount++;
            }

            // Error margin analysis
            if (model.ErrorMargin < 0.1) // Less than 10% error
            {
                qualityScore += (0.1 - model.ErrorMargin) / 0.1 * 0.2;
                factorCount++;
            }

            // Normalize the score
            return Math.Min(1.0, qualityScore / factorCount);
        }
    }

    public static class PythonStockPredictor
    {
        /// <summary>
        /// Calls a Python script to predict stock direction using a GPU-accelerated model (e.g., random forest, PyTorch, etc.).
        /// </summary>
        /// <param name="features">Dictionary of feature names and values (e.g., technical indicators)</param>
        /// <returns>Comprehensive prediction result with time series data and risk metrics</returns>
        public static async Task<PredictionResult> PredictAsync(Dictionary<string, double> features)
        {
            string pythonScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "stock_predictor.py");
            try
            {
                if (!File.Exists(pythonScript))
                {
                    throw new FileNotFoundException($"Python script not found at: {pythonScript}");
                }

                var requestData = new PredictionRequest
                {
                    Features = features,
                    RequireTimeSeries = true,
                    RequirePatternDetection = true,
                    RequireRiskMetrics = true
                };

                var json = System.Text.Json.JsonSerializer.Serialize(requestData);

                // Create temporary files for input/output
                string tempDir = Path.Combine(Path.GetTempPath(), "Quantra_Predictions");
                Directory.CreateDirectory(tempDir);
                
                string inputFile = Path.Combine(tempDir, $"input_{Guid.NewGuid()}.json");
                string outputFile = Path.Combine(tempDir, $"output_{Guid.NewGuid()}.json");

                try
                {
                    // Write input JSON to temp file
                    await File.WriteAllTextAsync(inputFile, json);

                    string pythonExe = "python";

                    var psi = new ProcessStartInfo
                    {
                        FileName = pythonExe,
                        Arguments = $"\"{pythonScript}\" \"{inputFile}\" \"{outputFile}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = Path.GetDirectoryName(pythonScript)
                    };

                    using (var process = Process.Start(psi))
                    {
                        if (process == null)
                            throw new Exception("Failed to start Python process");

                        var outputTask = process.StandardOutput.ReadToEndAsync();
                        var errorTask = process.StandardError.ReadToEndAsync();

                        await process.WaitForExitAsync();

                        string stdOut = await outputTask;
                        string stdErr = await errorTask;

                        if (process.ExitCode != 0)
                        {
                            throw new Exception($"Python prediction failed with exit code {process.ExitCode}: {stdErr}\nOutput: {stdOut}");
                        }

                        // Read the output file
                        if (!File.Exists(outputFile))
                        {
                            throw new Exception($"Python script did not create output file. StdErr: {stdErr}\nStdOut: {stdOut}");
                        }

                        stdOut = await File.ReadAllTextAsync(outputFile);
                        stdErr = string.Empty; // Reset since we successfully read output

                        // Check for empty or invalid output before deserializing
                        if (string.IsNullOrWhiteSpace(stdOut))
                        {
                            throw new Exception($"Python script did not return any output. StdErr: {stdErr}");
                        }

                        Debug.WriteLine($"Python output: {stdOut}");

                        try
                        {
                            // Parse comprehensive prediction result from stdout
                            var result = System.Text.Json.JsonSerializer.Deserialize<PythonPredictionResult>(stdOut);

                        if (result == null)
                            throw new Exception($"Failed to deserialize prediction result. Output: {stdOut}\nStdErr: {stdErr}");

                        // Create the full prediction result with all data
                        var predictionResult = new PredictionResult
                        {
                            Action = result.action,
                            Confidence = result.confidence,
                            TargetPrice = result.targetPrice,
                            FeatureWeights = result.weights ?? new Dictionary<string, double>()
                        };

                        // Map time series data if available
                        if (result.timeSeries != null)
                        {
                            predictionResult.TimeSeries = new TimeSeriesPrediction
                            {
                                PricePredictions = result.timeSeries.prices,
                                Confidence = result.timeSeries.confidence,
                                TimePoints = result.timeSeries.dates?.Select(date =>
                                    DateTime.TryParse(date, out DateTime dt) ? dt : DateTime.Now.AddDays(1)).ToList() ?? new List<DateTime>()
                            };
                        }

                        // Map risk metrics if available
                        if (result.risk != null)
                        {
                            predictionResult.RiskMetrics = new RiskMetrics
                            {
                                ValueAtRisk = result.risk.var,
                                MaxDrawdown = result.risk.maxDrawdown,
                                SharpeRatio = result.risk.sharpeRatio,
                                RiskScore = result.risk.riskScore
                            };
                        }

                        // Map detected patterns if available
                        if (result.patterns != null)
                        {
                            predictionResult.DetectedPatterns = result.patterns.Select(p => new TechnicalPattern
                            {
                                PatternName = p.name,
                                PatternStrength = p.strength,
                                ExpectedOutcome = p.outcome,
                                DetectionDate = DateTime.TryParse(p.detectionDate, out DateTime dt) ? dt : DateTime.Now,
                                HistoricalAccuracy = p.historicalAccuracy
                            }).ToList();
                        }

                            return predictionResult;
                        }
                        catch (Exception jsonEx)
                        {
                            throw new Exception($"JSON parsing error: {jsonEx.Message}\nRaw output: {stdOut}", jsonEx);
                        }
                    }
                }
                finally
                {
                    // Clean up temporary files
                    try
                    {
                        if (File.Exists(inputFile))
                            File.Delete(inputFile);
                        if (File.Exists(outputFile))
                            File.Delete(outputFile);
                    }
                    catch (Exception cleanupEx)
                    {
                        Debug.WriteLine($"Error cleaning up temp files: {cleanupEx.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error in PredictAsync: {ex}");
                throw;
            }
        }

        private class PredictionRequest
        {
            public Dictionary<string, double> Features { get; set; }
            public bool RequireTimeSeries { get; set; }
            public bool RequirePatternDetection { get; set; }
            public bool RequireRiskMetrics { get; set; }
        }

        private class PythonPredictionResult
        {
            public string action { get; set; }
            public double confidence { get; set; }
            public double targetPrice { get; set; }
            public Dictionary<string, double> weights { get; set; }
            public TimeSeriesData timeSeries { get; set; }
            public RiskData risk { get; set; }
            public List<PatternData> patterns { get; set; }
        }

        private class TimeSeriesData
        {
            public List<double> prices { get; set; }
            public List<string> dates { get; set; }
            public double confidence { get; set; }
        }

        private class RiskData
        {
            public double var { get; set; }
            public double maxDrawdown { get; set; }
            public double sharpeRatio { get; set; }
            public double riskScore { get; set; }
        }

        private class PatternData
        {
            public string name { get; set; }
            public double strength { get; set; }
            public string outcome { get; set; }
            public string detectionDate { get; set; }
            public double historicalAccuracy { get; set; }
        }
    }

    public class PredictionResult
    {
        public string Action { get; set; }

        /// <summary>
        /// Alias for Action property to maintain consistency with PredictionModel naming convention
        /// </summary>
        public string PredictedAction
        {
            get => Action;
            set => Action = value;
        }

        public double Confidence { get; set; }
        public double TargetPrice { get; set; }
        public Dictionary<string, double> FeatureWeights { get; set; } = new();
        public TimeSeriesPrediction TimeSeries { get; set; }
        public RiskMetrics RiskMetrics { get; set; }
        public List<TechnicalPattern> DetectedPatterns { get; set; } = new();

        // Real-time inference specific properties
        public string Symbol { get; set; }
        public string RequestId { get; set; }
        public double CurrentPrice { get; set; }
        public DateTime PredictionDate { get; set; } = DateTime.Now;
        public double InferenceTimeMs { get; set; }
        public string ModelType { get; set; }
        public bool IsRealTime { get; set; }
        public string Error { get; set; }

        // Additional real-time metrics
        public double RiskScore { get; set; }
        public double ValueAtRisk { get; set; }
        public double MaxDrawdown { get; set; }
        public double SharpeRatio { get; set; }

        /// <summary>
        /// Detailed timing information tracking the prediction lifecycle from request to completion.
        /// Captured when Python returns the prediction result.
        /// </summary>
        public PredictionTimestamp Timestamp { get; set; }

        // Helper properties
        public double PotentialReturn => CurrentPrice != 0 ? (TargetPrice - CurrentPrice) / CurrentPrice : 0;
        public bool HasError => !string.IsNullOrEmpty(Error);
        public string FormattedInferenceTime => $"{InferenceTimeMs:F2}ms";
    }

    public class TimeSeriesPrediction
    {
        public List<double> PricePredictions { get; set; }
        public List<DateTime> TimePoints { get; set; }
        public double Confidence { get; set; }
    }

    public class RiskMetrics
    {
        public double ValueAtRisk { get; set; }
        public double MaxDrawdown { get; set; }
        public double SharpeRatio { get; set; }
        public double RiskScore { get; set; }
    }
}
