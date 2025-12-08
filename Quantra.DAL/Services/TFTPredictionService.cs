using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for TFT (Temporal Fusion Transformer) predictions with confidence intervals and feature attention
    /// </summary>
    public class TFTPredictionService
    {
        private readonly LoggingService _loggingService;
        private readonly AlphaVantageService _alphaVantageService;
        private readonly string _pythonScriptPath;

        public TFTPredictionService(LoggingService loggingService, AlphaVantageService alphaVantageService = null)
        {
            _loggingService = loggingService;
            _alphaVantageService = alphaVantageService;
            _pythonScriptPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "python",
                "tft_predict.py"
            );
        }

        /// <summary>
        /// Get TFT predictions with confidence intervals and feature attention for a stock symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="historicalData">List of OHLC data points (at least 60 days)</param>
        /// <param name="lookbackDays">Number of historical days to use (default: 60)</param>
        /// <param name="futureHorizon">Number of days to predict ahead (default: 30)</param>
        /// <param name="forecastHorizons">Specific forecast horizons (default: 5, 10, 20, 30)</param>
        /// <returns>TFT prediction result with confidence bands and feature attention</returns>
        public async Task<TFTResult> GetTFTPredictionsAsync(
            string symbol,
            List<HistoricalPrice> historicalData,
            int lookbackDays = 60,
            int futureHorizon = 30,
            List<int> forecastHorizons = null)
        {
            try
            {
                if (string.IsNullOrEmpty(symbol))
                    throw new ArgumentNullException(nameof(symbol));

                if (historicalData == null || historicalData.Count < lookbackDays)
                    throw new ArgumentException($"Insufficient historical data. Need at least {lookbackDays} days, got {historicalData?.Count ?? 0}");

                if (!File.Exists(_pythonScriptPath))
                    throw new FileNotFoundException($"TFT prediction script not found: {_pythonScriptPath}");

                // Default forecast horizons if not provided
                forecastHorizons ??= new List<int> { 5, 10, 20, 30 };

                _loggingService?.Log("Info", $"Requesting TFT predictions for {symbol} with {historicalData.Count} days of historical data");

                // Create temporary input/output files
                string tempDir = Path.Combine(Path.GetTempPath(), "Quantra_TFT");
                Directory.CreateDirectory(tempDir);
                string inputFile = Path.Combine(tempDir, $"tft_input_{Guid.NewGuid()}.json");
                string outputFile = Path.Combine(tempDir, $"tft_output_{Guid.NewGuid()}.json");

                try
                {
                    // Prepare input data for Python script
                    var request = new TFTPredictionRequest
                    {
                        Symbol = symbol,
                        HistoricalSequence = historicalData.Select(d => new HistoricalDataPoint
                        {
                            Symbol = symbol,
                            Timestamp = d.Date.ToString("yyyy-MM-dd"),
                            Open = d.Open,
                            High = d.High,
                            Low = d.Low,
                            Close = d.Close,
                            Volume = d.Volume
                        }).ToList(),
                        CalendarFeatures = historicalData.Select(d => new CalendarFeature
                        {
                            DayOfWeek = (int)d.Date.DayOfWeek,
                            DayOfMonth = d.Date.Day,
                            Month = d.Date.Month,
                            Quarter = (d.Date.Month - 1) / 3 + 1,
                            IsMonthEnd = d.Date.Day >= DateTime.DaysInMonth(d.Date.Year, d.Date.Month) - 2,
                            IsQuarterEnd = (d.Date.Month % 3 == 0) &&
                                          d.Date.Day >= DateTime.DaysInMonth(d.Date.Year, d.Date.Month) - 2
                        }).ToList(),
                        LookbackDays = lookbackDays,
                        FutureHorizon = futureHorizon,
                        ForecastHorizons = forecastHorizons
                    };

                    // Write input JSON
                    var options = new JsonSerializerOptions
                    {
                        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                        WriteIndented = true
                    };
                    await File.WriteAllTextAsync(inputFile, JsonSerializer.Serialize(request, options));

                    // Execute Python script
                    var psi = new ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = $"\"{_pythonScriptPath}\" \"{inputFile}\" \"{outputFile}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = Path.GetDirectoryName(_pythonScriptPath)
                    };

                    using (var process = Process.Start(psi))
                    {
                        if (process == null)
                            throw new Exception("Failed to start Python process");

                        var errorOutput = new System.Text.StringBuilder();
                        var standardOutput = new System.Text.StringBuilder();

                        process.OutputDataReceived += (sender, e) =>
                        {
                            if (!string.IsNullOrEmpty(e.Data))
                            {
                                standardOutput.AppendLine(e.Data);
                                Debug.WriteLine($"TFT Python: {e.Data}");
                            }
                        };

                        process.ErrorDataReceived += (sender, e) =>
                        {
                            if (!string.IsNullOrEmpty(e.Data))
                            {
                                errorOutput.AppendLine(e.Data);
                                Debug.WriteLine($"TFT Python Error: {e.Data}");
                            }
                        };

                        process.BeginOutputReadLine();
                        process.BeginErrorReadLine();

                        await process.WaitForExitAsync();
                        process.WaitForExit(); // Ensure async readers complete

                        // Check for errors
                        if (!File.Exists(outputFile))
                        {
                            var errorMessage = $"TFT prediction failed - output file not created (exit code {process.ExitCode})\n" +
                                             $"Standard output:\n{standardOutput}\n" +
                                             $"Error output:\n{errorOutput}";
                            throw new Exception(errorMessage);
                        }

                        if (process.ExitCode != 0)
                        {
                            _loggingService?.Log("Warning", $"TFT Python process exited with code {process.ExitCode}");
                        }

                        // Read and parse results
                        var jsonResult = await File.ReadAllTextAsync(outputFile);
                        
                        // Log the JSON for debugging if deserialization fails
                        _loggingService?.Log("Debug", $"TFT JSON response length: {jsonResult.Length} chars");
                        
                        var readOptions = new JsonSerializerOptions
                        {
                            PropertyNameCaseInsensitive = true,
                            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                        };
                        
                        TFTPythonResponse pythonResult;
                        try
                        {
                            pythonResult = JsonSerializer.Deserialize<TFTPythonResponse>(jsonResult, readOptions);
                        }
                        catch (JsonException jsonEx)
                        {
                            _loggingService?.Log("Error", $"Failed to deserialize TFT response. First 500 chars: {jsonResult.Substring(0, Math.Min(500, jsonResult.Length))}");
                            throw new Exception($"Failed to parse TFT prediction result: {jsonEx.Message}", jsonEx);
                        }

                        if (pythonResult == null)
                            throw new Exception("Failed to parse TFT prediction result");

                        if (!string.IsNullOrEmpty(pythonResult.Error))
                        {
                            _loggingService?.Log("Warning", $"TFT prediction returned error: {pythonResult.Error}");
                        }

                        // Convert Python response to C# result format
                        var result = ConvertPythonResponseToResult(pythonResult, historicalData);

                        _loggingService?.Log("Info", $"TFT prediction complete for {symbol}: {pythonResult.Action} with {pythonResult.Confidence:P0} confidence");

                        return result;
                    }
                }
                finally
                {
                    // Cleanup temp files
                    try
                    {
                        if (File.Exists(inputFile)) File.Delete(inputFile);
                        if (File.Exists(outputFile)) File.Delete(outputFile);
                    }
                    catch { /* Ignore cleanup errors */ }
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"TFT prediction failed for {symbol}");

                // Return empty result on error
                return new TFTResult
                {
                    Predictions = new List<TFTForecast>(),
                    FeatureAttention = new Dictionary<string, double>(),
                    Error = ex.Message
                };
            }
        }

        private TFTResult ConvertPythonResponseToResult(TFTPythonResponse pythonResult, List<HistoricalPrice> historicalData)
        {
            var result = new TFTResult
            {
                Predictions = new List<TFTForecast>(),
                FeatureAttention = new Dictionary<string, double>(),
                Action = pythonResult.Action,
                Confidence = pythonResult.Confidence,
                CurrentPrice = pythonResult.CurrentPrice,
                TargetPrice = pythonResult.TargetPrice,
                Error = pythonResult.Error
            };

            // Get the last date from historical data
            var lastDate = historicalData.OrderByDescending(d => d.Date).First().Date;

            // Convert horizons to prediction points
            if (pythonResult.Horizons != null && pythonResult.Horizons.Count > 0)
            {
                _loggingService?.Log("Debug", $"Processing {pythonResult.Horizons.Count} horizon predictions");
                
                foreach (var horizon in pythonResult.Horizons)
                {
                    var predictedPrice = pythonResult.CurrentPrice * (1 + horizon.PredictedChange);
                    var upperPrice = pythonResult.CurrentPrice * (1 + horizon.UpperBound);
                    var lowerPrice = pythonResult.CurrentPrice * (1 + horizon.LowerBound);
                    
                    result.Predictions.Add(new TFTForecast
                    {
                        Timestamp = lastDate.AddDays(horizon.DaysAhead),
                        PredictedPrice = predictedPrice,
                        UpperConfidence = upperPrice,
                        LowerConfidence = lowerPrice
                    });
                    
                    _loggingService?.Log("Debug", $"Horizon {horizon.DaysAhead}d: Change={horizon.PredictedChange:P2}, Price=${predictedPrice:F2}");
                }
            }
            else
            {
                _loggingService?.Log("Warning", "No horizon data in Python response, using fallback prediction");
                
                // Fallback: use main prediction values
                result.Predictions.Add(new TFTForecast
                {
                    Timestamp = lastDate.AddDays(30),
                    PredictedPrice = pythonResult.TargetPrice,
                    UpperConfidence = pythonResult.UpperBound,
                    LowerConfidence = pythonResult.LowerBound
                });
            }

            // Convert feature importance to attention dictionary
            if (pythonResult.FeatureImportance != null && pythonResult.FeatureImportance.Count > 0)
            {
                // Assume feature importance is an array of values corresponding to features
                // For now, create generic feature names
                for (int i = 0; i < pythonResult.FeatureImportance.Count && i < 50; i++)
                {
                    string featureName = $"Feature_{i + 1}";

                    // Map to known feature names if possible
                    if (i < FeatureNames.Count)
                        featureName = FeatureNames[i];

                    result.FeatureAttention[featureName] = pythonResult.FeatureImportance[i];
                }
            }

            return result;
        }

        // Common feature names used in TFT models
        private static readonly List<string> FeatureNames = new List<string>
        {
            "Close_Price", "Open_Price", "High_Price", "Low_Price", "Volume",
            "Returns", "Log_Returns", "Price_Change",
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width",
            "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_200",
            "EMA_5", "EMA_10", "EMA_20", "EMA_50",
            "Stochastic_K", "Stochastic_D",
            "ATR", "ADX", "CCI", "Williams_R",
            "OBV", "MFI", "VWAP",
            "Day_Of_Week", "Day_Of_Month", "Month", "Quarter",
            "Is_Month_End", "Is_Quarter_End",
            "Volatility_5", "Volatility_10", "Volatility_20",
            "Volume_SMA_5", "Volume_SMA_20",
            "Price_To_SMA_20", "Price_To_SMA_50",
            "Volume_Ratio", "Price_Momentum_5", "Price_Momentum_10"
        };
    }

    #region Request/Response Models

    public class TFTPredictionRequest
    {
        [JsonPropertyName("symbol")]
        public string Symbol { get; set; }

        [JsonPropertyName("historical_sequence")]
        public List<HistoricalDataPoint> HistoricalSequence { get; set; }

        [JsonPropertyName("calendar_features")]
        public List<CalendarFeature> CalendarFeatures { get; set; }

        [JsonPropertyName("lookback_days")]
        public int LookbackDays { get; set; }

        [JsonPropertyName("future_horizon")]
        public int FutureHorizon { get; set; }

        [JsonPropertyName("forecast_horizons")]
        public List<int> ForecastHorizons { get; set; }
    }

    public class HistoricalDataPoint
    {
        [JsonPropertyName("symbol")]
        public string Symbol { get; set; }

        [JsonPropertyName("timestamp")]
        public string Timestamp { get; set; }

        [JsonPropertyName("open")]
        public double Open { get; set; }

        [JsonPropertyName("high")]
        public double High { get; set; }

        [JsonPropertyName("low")]
        public double Low { get; set; }

        [JsonPropertyName("close")]
        public double Close { get; set; }

        [JsonPropertyName("volume")]
        public long Volume { get; set; }
    }

    public class CalendarFeature
    {
        [JsonPropertyName("day_of_week")]
        public int DayOfWeek { get; set; }

        [JsonPropertyName("day_of_month")]
        public int DayOfMonth { get; set; }

        [JsonPropertyName("month")]
        public int Month { get; set; }

        [JsonPropertyName("quarter")]
        public int Quarter { get; set; }

        [JsonPropertyName("is_month_end")]
        public bool IsMonthEnd { get; set; }

        [JsonPropertyName("is_quarter_end")]
        public bool IsQuarterEnd { get; set; }
    }

    public class TFTPythonResponse
    {
        [JsonPropertyName("symbol")]
        public string Symbol { get; set; }

        [JsonPropertyName("action")]
        public string Action { get; set; }

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }

        [JsonPropertyName("currentPrice")]
        public double CurrentPrice { get; set; }

        [JsonPropertyName("targetPrice")]
        public double TargetPrice { get; set; }

        [JsonPropertyName("medianPrediction")]
        public double MedianPrediction { get; set; }

        [JsonPropertyName("lowerBound")]
        public double LowerBound { get; set; }

        [JsonPropertyName("upperBound")]
        public double UpperBound { get; set; }

        [JsonPropertyName("horizons")]
        public List<TFTHorizonData> Horizons { get; set; }

        [JsonPropertyName("modelType")]
        public string ModelType { get; set; }

        [JsonPropertyName("uncertainty")]
        public double Uncertainty { get; set; }

        [JsonPropertyName("featureImportance")]
        public List<double> FeatureImportance { get; set; }

        [JsonPropertyName("error")]
        public string Error { get; set; }

        [JsonPropertyName("success")]
        public bool Success { get; set; }
    }

    /// <summary>
    /// Horizon prediction data from Python TFT response
    /// </summary>
    public class TFTHorizonData
    {
        [JsonPropertyName("days_ahead")]
        public int DaysAhead { get; set; }

        [JsonPropertyName("predicted_change")]
        public double PredictedChange { get; set; }

        [JsonPropertyName("lower_bound")]
        public double LowerBound { get; set; }

        [JsonPropertyName("upper_bound")]
        public double UpperBound { get; set; }
    }

    #endregion

    #region Service Result Models

    /// <summary>
    /// TFT prediction result for service layer
    /// </summary>
    public class TFTResult
    {
        public List<TFTForecast> Predictions { get; set; }
        public Dictionary<string, double> FeatureAttention { get; set; }
        public string Action { get; set; }
        public double Confidence { get; set; }
        public double CurrentPrice { get; set; }
        public double TargetPrice { get; set; }
        public string Error { get; set; }
    }

    /// <summary>
    /// Single TFT forecast point with confidence interval
    /// </summary>
    public class TFTForecast
    {
        public DateTime Timestamp { get; set; }
        public double PredictedPrice { get; set; }
        public double UpperConfidence { get; set; }
        public double LowerConfidence { get; set; }
    }

    #endregion
}
