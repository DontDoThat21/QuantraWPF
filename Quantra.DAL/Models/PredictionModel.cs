using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Media;
using Quantra.DAL.Models;
using Quantra.DAL.Services;

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

        // Database-related prediction properties
        public int Id { get; set; } // Database Id
        public DateTime? ExpectedFruitionDate { get; set; } // Date when the prediction is expected to come to fruition
        public string ArchitectureType { get; set; } // Neural network architecture (lstm, gru, transformer, tft)
        public int? TrainingHistoryId { get; set; } // Reference to the training history used
        public string UserQuery { get; set; } // Original user query that triggered this prediction
        public int? ChatHistoryId { get; set; } // Links prediction to a chat history record

        // Multi-horizon target prices from TFT predictions
        public double? Target5d { get; set; } // Target price for 5-day horizon
        public double? Target10d { get; set; } // Target price for 10-day horizon
        public double? Target20d { get; set; } // Target price for 20-day horizon
        public double? Target30d { get; set; } // Target price for 30-day horizon
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
        /// <param name="useFeatureEngineering">Whether to use advanced feature engineering pipeline (default: true)</param>
        /// <returns>Comprehensive prediction result with time series data and risk metrics</returns>
        public static async Task<PredictionResult> PredictAsync(Dictionary<string, double> features, bool useFeatureEngineering = true)
        {
            string pythonScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "stock_predictor.py");
            try
            {
                if (!File.Exists(pythonScript))
                {
                    throw new FileNotFoundException($"Python script not found at: {pythonScript}");
                }

                // CRITICAL FIX: Validate feature data quality BEFORE calling Python
                // Check if features dictionary contains at least some valid data
                int validFeatureCount = features.Count(kvp => kvp.Value != 0 && !double.IsNaN(kvp.Value) && !double.IsInfinity(kvp.Value));
                
                if (validFeatureCount < 2)
                {
                    // Insufficient valid data - most features are zero or invalid
                    string errorMsg = $"Insufficient valid market data for prediction. " +
                                    $"Only {validFeatureCount} out of {features.Count} features have valid non-zero values. " +
                                    $"This indicates API failure, rate limiting, or invalid symbol. " +
                                    $"Features: [{string.Join(", ", features.Select(kvp => $"{kvp.Key}={kvp.Value:F4}"))}]";
                    
                    return new PredictionResult
                    {
                        Action = "ERROR",
                        Confidence = 0.0,
                        TargetPrice = 0.0,
                        CurrentPrice = 0.0,
                        Error = errorMsg
                    };
                }

                // CRITICAL FIX: Ensure current_price is in the features dictionary and has a valid value
                // The Python script requires 'current_price', 'close', or 'price' to be present and non-zero
                if (!features.ContainsKey("current_price") && !features.ContainsKey("close") && !features.ContainsKey("price"))
                {
                    // Try to find a price-related key in the features dictionary
                    var priceKey = features.Keys.FirstOrDefault(k => k.ToLowerInvariant().Contains("price") || k.ToLowerInvariant() == "close");
                    
                    if (priceKey != null && features[priceKey] > 0)
                    {
                        // Add current_price using the found price key (only if it has a valid value)
                        features["current_price"] = features[priceKey];
                    }
                    else
                    {
                        // No valid price found - return error immediately to avoid Python script failure
                        string errorMsg = $"Features dictionary must include a valid 'current_price', 'close', or 'price' value. " +
                                        $"Please ensure the Features dictionary passed from C# includes the current stock price. " +
                                        $"Current features: [{string.Join(", ", features.Select(kvp => $"{kvp.Key}={kvp.Value:F4}"))}]";
                        
                        return new PredictionResult
                        {
                            Action = "ERROR",
                            Confidence = 0.0,
                            TargetPrice = 0.0,
                            CurrentPrice = 0.0,
                            Error = errorMsg
                        };
                    }
                }
                else
                {
                    // Ensure the current_price value is valid (not zero, not NaN, not infinity)
                    var currentPriceKey = features.ContainsKey("current_price") ? "current_price" : 
                                        features.ContainsKey("close") ? "close" : "price";
                    var currentPriceValue = features[currentPriceKey];
                    
                    if (currentPriceValue <= 0 || double.IsNaN(currentPriceValue) || double.IsInfinity(currentPriceValue))
                    {
                        string errorMsg = $"Invalid current_price value: {currentPriceValue}. " +
                                        $"The price must be a positive number. " +
                                        $"This indicates API failure or invalid market data. " +
                                        $"Features: [{string.Join(", ", features.Select(kvp => $"{kvp.Key}={kvp.Value:F4}"))}]";
                        
                        return new PredictionResult
                        {
                            Action = "ERROR",
                            Confidence = 0.0,
                            TargetPrice = 0.0,
                            CurrentPrice = currentPriceValue,
                            Error = errorMsg
                        };
                    }
                    
                    // Ensure current_price key exists with the valid value
                    if (!features.ContainsKey("current_price"))
                    {
                        features["current_price"] = currentPriceValue;
                    }
                }

                var requestData = new PredictionRequest
                {
                    Features = features,
                    RequireTimeSeries = true,
                    RequirePatternDetection = true,
                    RequireRiskMetrics = true,
                    UseFeatureEngineering = useFeatureEngineering
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

                        // Check for Python-side errors first
                        if (!string.IsNullOrEmpty(result.error))
                        {
                            if (result.needsRetraining)
                            {
                                throw new Exception($"MODEL NEEDS RETRAINING: {result.error}");
                            }
                            throw new Exception($"Python prediction error: {result.error}");
                        }

                        // Create the full prediction result with all data
                        var predictionResult = new PredictionResult
                        {
                            Action = result.action,
                            Confidence = result.confidence,
                            TargetPrice = result.targetPrice,
                            FeatureWeights = result.weights ?? new Dictionary<string, double>(),
                            Error = result.error
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

        /// <summary>
        /// Calls the Temporal Fusion Transformer Python script for multi-horizon stock price prediction.
        /// TFT provides superior forecasting with attention mechanisms, variable selection, and uncertainty quantification.
        /// </summary>
        /// <param name="indicators">Dictionary of technical indicators and features</param>
        /// <param name="symbol">Stock symbol for the prediction</param>
        /// <param name="historicalSequence">Optional historical price sequence for TFT temporal analysis</param>
        /// <param name="horizons">Prediction horizons in days (default: 1, 3, 5, 10)</param>
        /// <returns>TFT prediction result with multi-horizon forecasts and uncertainty intervals</returns>
        public static async Task<TFTPredictionResult> PredictWithTFTAsync(
            Dictionary<string, double> indicators,
            string symbol = null,
            List<Dictionary<string, double>> historicalSequence = null,
            List<int> horizons = null)
        {
            string pythonScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "temporal_fusion_transformer.py");
            string tftPredictScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "tft_predict.py");
            
            // Use tft_predict.py if it exists (preferred for production use)
            string scriptToUse = File.Exists(tftPredictScript) ? tftPredictScript : pythonScript;
            
            var result = new TFTPredictionResult
            {
                Symbol = symbol ?? "UNKNOWN",
                PredictionTimestamp = DateTime.Now
            };
            
            try
            {
                if (!File.Exists(scriptToUse))
                {
                    result.Error = $"TFT Python script not found at: {scriptToUse}";
                    result.Action = "HOLD";
                    result.Confidence = 0.0;
                    return result;
                }

                // Default horizons for multi-horizon TFT prediction
                if (horizons == null)
                {
                    horizons = new List<int> { 1, 3, 5, 10 };
                }

                // Prepare TFT request data
                var requestData = new TFTRequest
                {
                    Symbol = symbol ?? "UNKNOWN",
                    Indicators = indicators,
                    HistoricalSequence = historicalSequence,
                    Horizons = horizons,
                    LookbackDays = 60,
                    FutureHorizon = 30,
                    ForecastHorizons = new int[] { 5, 10, 20, 30 }
                };

                var json = System.Text.Json.JsonSerializer.Serialize(requestData);

                // Create temporary files for input/output
                string tempDir = Path.Combine(Path.GetTempPath(), "Quantra_TFT_Predictions");
                Directory.CreateDirectory(tempDir);
                
                string inputFile = Path.Combine(tempDir, $"tft_input_{Guid.NewGuid()}.json");
                string outputFile = Path.Combine(tempDir, $"tft_output_{Guid.NewGuid()}.json");

                var stopwatch = Stopwatch.StartNew();

                try
                {
                    // Write input JSON to temp file
                    await File.WriteAllTextAsync(inputFile, json);

                    string pythonExe = "python";

                    var psi = new ProcessStartInfo
                    {
                        FileName = pythonExe,
                        Arguments = $"\"{scriptToUse}\" \"{inputFile}\" \"{outputFile}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = Path.GetDirectoryName(scriptToUse)
                    };

                    using (var process = Process.Start(psi))
                    {
                        if (process == null)
                        {
                            result.Error = "Failed to start Python process for TFT prediction";
                            result.Action = "HOLD";
                            result.Confidence = 0.0;
                            return result;
                        }

                        var outputTask = process.StandardOutput.ReadToEndAsync();
                        var errorTask = process.StandardError.ReadToEndAsync();

                        // Wait up to 3 minutes for TFT prediction
                        if (!process.WaitForExit(180000))
                        {
                            try { process.Kill(); } catch { }
                            result.Error = "TFT prediction timed out after 3 minutes";
                            result.Action = "HOLD";
                            result.Confidence = 0.0;
                            return result;
                        }

                        string stdOut = await outputTask;
                        string stdErr = await errorTask;

                        stopwatch.Stop();
                        result.InferenceTimeMs = stopwatch.ElapsedMilliseconds;

                        if (process.ExitCode != 0)
                        {
                            result.Error = $"TFT prediction failed with exit code {process.ExitCode}: {stdErr}";
                            result.Action = "HOLD";
                            result.Confidence = 0.0;
                            return result;
                        }

                        // Read the output file
                        if (!File.Exists(outputFile))
                        {
                            result.Error = $"TFT script did not create output file. StdErr: {stdErr}";
                            result.Action = "HOLD";
                            result.Confidence = 0.0;
                            return result;
                        }

                        string outputJson = await File.ReadAllTextAsync(outputFile);

                        if (string.IsNullOrWhiteSpace(outputJson))
                        {
                            result.Error = "TFT script returned empty output";
                            result.Action = "HOLD";
                            result.Confidence = 0.0;
                            return result;
                        }

                        Debug.WriteLine($"TFT output: {outputJson}");

                        // Parse TFT result
                        var options = new System.Text.Json.JsonSerializerOptions
                        {
                            PropertyNameCaseInsensitive = true
                        };
                        
                        var tftResponse = System.Text.Json.JsonSerializer.Deserialize<TFTResponse>(outputJson, options);

                        if (tftResponse == null)
                        {
                            result.Error = $"Failed to deserialize TFT result. Output: {outputJson}";
                            result.Action = "HOLD";
                            result.Confidence = 0.0;
                            return result;
                        }

                        // Check for Python-side errors
                        if (!string.IsNullOrEmpty(tftResponse.Error))
                        {
                            result.Error = tftResponse.Error;
                            result.Action = "HOLD";
                            result.Confidence = 0.0;
                            return result;
                        }

                        // Map response to result
                        result.Action = tftResponse.Action ?? "HOLD";
                        result.Confidence = tftResponse.Confidence;
                        result.TargetPrice = tftResponse.TargetPrice;
                        result.CurrentPrice = tftResponse.CurrentPrice;
                        result.LowerBound = tftResponse.LowerBound;
                        result.UpperBound = tftResponse.UpperBound;
                        result.Uncertainty = tftResponse.Uncertainty;

                        // Map horizon predictions
                        if (tftResponse.Horizons != null)
                        {
                            result.Horizons = new Dictionary<string, HorizonPredictionData>();
                            foreach (var kvp in tftResponse.Horizons)
                            {
                                result.Horizons[kvp.Key] = new HorizonPredictionData
                                {
                                    MedianPrice = kvp.Value.Median,
                                    LowerBound = kvp.Value.Lower,
                                    UpperBound = kvp.Value.Upper,
                                    TargetPrice = kvp.Value.TargetPrice
                                };
                            }
                        }

                        // Map feature weights
                        if (tftResponse.FeatureWeights != null)
                        {
                            result.FeatureWeights = tftResponse.FeatureWeights;
                        }

                        // Map temporal attention
                        if (tftResponse.TemporalAttention != null)
                        {
                            result.TemporalAttention = new Dictionary<int, double>();
                            foreach (var kvp in tftResponse.TemporalAttention)
                            {
                                if (int.TryParse(kvp.Key, out int timeOffset))
                                {
                                    result.TemporalAttention[timeOffset] = kvp.Value;
                                }
                            }
                        }

                        return result;
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
                        Debug.WriteLine($"Error cleaning up TFT temp files: {cleanupEx.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error in PredictWithTFTAsync: {ex}");
                result.Error = ex.Message;
                result.Action = "HOLD";
                result.Confidence = 0.0;
                return result;
            }
        }

        private class TFTRequest
        {
            public string Symbol { get; set; }
            public Dictionary<string, double> Indicators { get; set; }
            public List<Dictionary<string, double>> HistoricalSequence { get; set; }
            public List<int> Horizons { get; set; }
            public int LookbackDays { get; set; }
            public int FutureHorizon { get; set; }
            public int[] ForecastHorizons { get; set; }
        }

        private class TFTResponse
        {
            public string Action { get; set; }
            public double Confidence { get; set; }
            public double TargetPrice { get; set; }
            public double CurrentPrice { get; set; }
            public double LowerBound { get; set; }
            public double UpperBound { get; set; }
            public double Uncertainty { get; set; }
            public Dictionary<string, TFTHorizonData> Horizons { get; set; }
            public Dictionary<string, double> FeatureWeights { get; set; }
            public Dictionary<string, double> TemporalAttention { get; set; }
            public string Error { get; set; }
        }

        private class TFTHorizonData
        {
            public double Lower { get; set; }
            public double Median { get; set; }
            public double Upper { get; set; }
            public double TargetPrice { get; set; }
        }

        private class PredictionRequest
        {
            public Dictionary<string, double> Features { get; set; }
            public bool RequireTimeSeries { get; set; }
            public bool RequirePatternDetection { get; set; }
            public bool RequireRiskMetrics { get; set; }
            public bool UseFeatureEngineering { get; set; }
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
            public string error { get; set; }
            public bool needsRetraining { get; set; }
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

        // TFT-specific properties for multi-horizon predictions and uncertainty quantification
        /// <summary>
        /// Multi-horizon predictions from TFT model (5, 10, 20, 30 days ahead).
        /// Each horizon includes median, lower, and upper bounds.
        /// </summary>
        public List<HorizonPrediction> TimeSeriesPredictions { get; set; } = new();

        /// <summary>
        /// Uncertainty measure from TFT model (difference between upper and lower bounds).
        /// Higher values indicate less certain predictions.
        /// </summary>
        public double PredictionUncertainty { get; set; }

        /// <summary>
        /// Confidence interval for the prediction [lower, upper].
        /// </summary>
        public double[] ConfidenceInterval { get; set; }

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
