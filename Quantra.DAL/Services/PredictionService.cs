using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using System.Text.Json;
using System.Diagnostics;
using System.Threading;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Models;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public class PredictionService
    {
        private const string PythonScript = "python/stock_predictor.py";
        private const string PythonExecutable = "python";

        private readonly QuantraDbContext _dbContext;

        public PredictionService()
        {
        }

        public PredictionService(QuantraDbContext dbContext)
        {
            _dbContext = dbContext;
        }

        /// <summary>
        /// Predicts future stock price movement using random forest model
        /// </summary>
        public async Task<PredictionModel> PredictStockMovement(string symbol, List<StockDataPoint> historicalData)
        {
            try
            {
                // Prepare input data
                var inputData = historicalData.Select(h => new
                {
                    date = h.Date.ToString("yyyy-MM-dd"),
                    open = h.Open,
                    high = h.High,
                    low = h.Low,
                    close = h.Close,
                    volume = h.Volume
                }).ToList();

                // Create temporary files for input/output
                string tempInput = Path.GetTempFileName();
                string tempOutput = Path.GetTempFileName();

                try
                {
                    // Write input data to temp file
                    await File.WriteAllTextAsync(tempInput, JsonSerializer.Serialize(inputData));

                    // Create process to run Python script
                    var startInfo = new ProcessStartInfo
                    {
                        FileName = PythonExecutable,
                        Arguments = $"\"{PythonScript}\" \"{tempInput}\" \"{tempOutput}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };

                    // Run prediction script
                    using var process = Process.Start(startInfo);
                    string output = await process.StandardOutput.ReadToEndAsync();
                    string error = await process.StandardError.ReadToEndAsync();
                    await process.WaitForExitAsync();

                    if (process.ExitCode != 0)
                    {
                        throw new Exception($"Python prediction failed: {error}");
                    }

                    // Read prediction results
                    var jsonResult = await File.ReadAllTextAsync(tempOutput);
                    var result = JsonSerializer.Deserialize<PredictionResult>(jsonResult);

                    // Convert to PredictionModel
                    return new PredictionModel
                    {
                        Symbol = symbol,
                        PredictedAction = result.action,
                        Confidence = result.confidence,
                        CurrentPrice = result.currentPrice,
                        TargetPrice = result.targetPrice,
                        PredictionDate = DateTime.Now,
                        PotentialReturn = (result.targetPrice - result.currentPrice) / result.currentPrice,
                        // Add feature importances to indicators
                        Indicators = result.featureWeights.ToDictionary(
                            kv => kv.Key,
                            kv => (double)kv.Value)
                    };
                }
                finally
                {
                    // Cleanup temp files
                    try
                    {
                        File.Delete(tempInput);
                        File.Delete(tempOutput);
                    }
                    catch { /* Ignore cleanup errors */ }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to predict stock movement for {symbol}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Saves a TFT (Temporal Fusion Transformer) prediction with all related multi-horizon data.
        /// Stores the main prediction, horizon-specific predictions, feature importance, and temporal attention.
        /// </summary>
        /// <param name="prediction">The main prediction model to save</param>
        /// <param name="tftResult">The TFT prediction result with multi-horizon data</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>The ID of the saved prediction</returns>
        public async Task<int> SaveTFTPredictionAsync(
            PredictionModel prediction,
            TFTPredictionResult tftResult,
            CancellationToken cancellationToken = default)
        {
            if (_dbContext == null)
            {
                throw new InvalidOperationException("Database context is not available. Use the constructor that accepts QuantraDbContext.");
            }

            // Create the main prediction entity
            var predictionEntity = new StockPredictionEntity
            {
                Symbol = prediction.Symbol,
                PredictedAction = tftResult.Action ?? prediction.PredictedAction ?? "HOLD",
                Confidence = tftResult.Confidence,
                CurrentPrice = tftResult.CurrentPrice > 0 ? tftResult.CurrentPrice : prediction.CurrentPrice,
                TargetPrice = tftResult.TargetPrice > 0 ? tftResult.TargetPrice : prediction.TargetPrice,
                PotentialReturn = tftResult.PotentialReturn,
                CreatedDate = DateTime.Now,
                ModelType = "tft",
                ArchitectureType = "tft",
                TrainingHistoryId = prediction.TrainingHistoryId,
                TradingRule = prediction.TradingRule,
                UserQuery = prediction.UserQuery,
                ChatHistoryId = prediction.ChatHistoryId
            };

            // Add main prediction to database
            await _dbContext.StockPredictions.AddAsync(predictionEntity, cancellationToken);
            await _dbContext.SaveChangesAsync(cancellationToken);

            int predictionId = predictionEntity.Id;

            // Save multi-horizon predictions
            if (tftResult.Horizons != null && tftResult.Horizons.Count > 0)
            {
                foreach (var kvp in tftResult.Horizons)
                {
                    // Parse horizon from key (e.g., "5d" -> 5)
                    int horizon = ParseHorizonFromKey(kvp.Key);
                    var horizonData = kvp.Value;

                    var horizonEntity = new StockPredictionHorizonEntity
                    {
                        PredictionId = predictionId,
                        Horizon = horizon,
                        TargetPrice = horizonData.TargetPrice > 0 ? horizonData.TargetPrice : horizonData.MedianPrice,
                        LowerBound = horizonData.LowerBound,
                        UpperBound = horizonData.UpperBound,
                        Confidence = horizonData.Confidence > 0 ? horizonData.Confidence : tftResult.Confidence,
                        ExpectedFruitionDate = DateTime.Now.AddDays(horizon)
                    };

                    await _dbContext.StockPredictionHorizons.AddAsync(horizonEntity, cancellationToken);
                }
            }

            // Save feature importance weights
            if (tftResult.FeatureWeights != null && tftResult.FeatureWeights.Count > 0)
            {
                foreach (var kvp in tftResult.FeatureWeights)
                {
                    var featureEntity = new PredictionFeatureImportanceEntity
                    {
                        PredictionId = predictionId,
                        FeatureName = kvp.Key,
                        ImportanceScore = kvp.Value,
                        FeatureType = DetermineFeatureType(kvp.Key)
                    };

                    await _dbContext.PredictionFeatureImportances.AddAsync(featureEntity, cancellationToken);
                }
            }

            // Save temporal attention weights
            if (tftResult.TemporalAttention != null && tftResult.TemporalAttention.Count > 0)
            {
                foreach (var kvp in tftResult.TemporalAttention)
                {
                    var attentionEntity = new PredictionTemporalAttentionEntity
                    {
                        PredictionId = predictionId,
                        TimeStep = kvp.Key,
                        AttentionWeight = kvp.Value
                    };

                    await _dbContext.PredictionTemporalAttentions.AddAsync(attentionEntity, cancellationToken);
                }
            }

            await _dbContext.SaveChangesAsync(cancellationToken);

            return predictionId;
        }

        /// <summary>
        /// Retrieves a prediction with all its multi-horizon data.
        /// </summary>
        /// <param name="predictionId">The prediction ID to retrieve</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>The prediction entity with all related data, or null if not found</returns>
        public async Task<StockPredictionEntity> GetPredictionWithHorizonsAsync(
            int predictionId,
            CancellationToken cancellationToken = default)
        {
            if (_dbContext == null)
            {
                throw new InvalidOperationException("Database context is not available. Use the constructor that accepts QuantraDbContext.");
            }

            return await _dbContext.StockPredictions
                .Include(p => p.PredictionHorizons)
                .Include(p => p.FeatureImportances)
                .Include(p => p.TemporalAttentions)
                .Include(p => p.Indicators)
                .FirstOrDefaultAsync(p => p.Id == predictionId, cancellationToken);
        }

        /// <summary>
        /// Retrieves all TFT predictions for a symbol with their multi-horizon data.
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="limit">Maximum number of predictions to return (default: 10)</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>List of prediction entities with all related data</returns>
        public async Task<List<StockPredictionEntity>> GetTFTPredictionsForSymbolAsync(
            string symbol,
            int limit = 10,
            CancellationToken cancellationToken = default)
        {
            if (_dbContext == null)
            {
                throw new InvalidOperationException("Database context is not available. Use the constructor that accepts QuantraDbContext.");
            }

            return await _dbContext.StockPredictions
                .Where(p => p.Symbol == symbol && p.ModelType == "tft")
                .OrderByDescending(p => p.CreatedDate)
                .Take(limit)
                .Include(p => p.PredictionHorizons)
                .Include(p => p.FeatureImportances)
                .Include(p => p.TemporalAttentions)
                .ToListAsync(cancellationToken);
        }

        /// <summary>
        /// Updates actual prices for horizon predictions that have reached their fruition date.
        /// Used for model evaluation and accuracy tracking.
        /// </summary>
        /// <param name="actualPrices">Dictionary mapping (PredictionId, Horizon) to actual prices</param>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task UpdateActualPricesAsync(
            Dictionary<(int PredictionId, int Horizon), double> actualPrices,
            CancellationToken cancellationToken = default)
        {
            if (_dbContext == null)
            {
                throw new InvalidOperationException("Database context is not available. Use the constructor that accepts QuantraDbContext.");
            }

            foreach (var kvp in actualPrices)
            {
                var horizonEntity = await _dbContext.StockPredictionHorizons
                    .FirstOrDefaultAsync(h => h.PredictionId == kvp.Key.PredictionId && h.Horizon == kvp.Key.Horizon, cancellationToken);

                if (horizonEntity != null)
                {
                    var prediction = await _dbContext.StockPredictions
                        .FirstOrDefaultAsync(p => p.Id == kvp.Key.PredictionId, cancellationToken);

                    horizonEntity.ActualPrice = kvp.Value;

                    if (prediction != null && prediction.CurrentPrice > 0)
                    {
                        horizonEntity.ActualReturn = (kvp.Value - prediction.CurrentPrice) / prediction.CurrentPrice;
                    }

                    if (horizonEntity.TargetPrice > 0)
                    {
                        horizonEntity.ErrorPct = (kvp.Value - horizonEntity.TargetPrice) / horizonEntity.TargetPrice;
                    }
                }
            }

            await _dbContext.SaveChangesAsync(cancellationToken);
        }

        /// <summary>
        /// Gets horizon predictions that are due for actual price updates.
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>List of horizons that have passed their expected fruition date but don't have actual prices</returns>
        public async Task<List<StockPredictionHorizonEntity>> GetHorizonsPendingActualPriceAsync(
            CancellationToken cancellationToken = default)
        {
            if (_dbContext == null)
            {
                throw new InvalidOperationException("Database context is not available. Use the constructor that accepts QuantraDbContext.");
            }

            return await _dbContext.StockPredictionHorizons
                .Include(h => h.Prediction)
                .Where(h => h.ExpectedFruitionDate <= DateTime.Now && h.ActualPrice == null)
                .OrderBy(h => h.ExpectedFruitionDate)
                .ToListAsync(cancellationToken);
        }

        /// <summary>
        /// Determines the TFT feature type based on feature name.
        /// </summary>
        private static string DetermineFeatureType(string featureName)
        {
            // Static features: time-invariant characteristics
            var staticFeatures = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "Sector", "Industry", "MarketCap", "MarketCapCategory", "Exchange", 
                "Country", "AssetClass", "DividendYield", "PERatio", "Beta"
            };

            // Known future features: features known in advance
            var knownFeatures = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "DayOfWeek", "DayOfMonth", "Month", "Quarter", "Year", 
                "IsHoliday", "IsWeekend", "DaysToEarnings", "DaysToDividend",
                "TradingDaysInWeek", "WeekNumber", "IsMonthEnd", "IsQuarterEnd"
            };

            if (staticFeatures.Contains(featureName))
            {
                return "static";
            }

            if (knownFeatures.Contains(featureName))
            {
                return "known";
            }

            // Default to observed (time-varying observed inputs like price, volume, indicators)
            return "observed";
        }

        /// <summary>
        /// Parses horizon in days from string key (e.g., "5d" -> 5, "10" -> 10).
        /// </summary>
        private static int ParseHorizonFromKey(string key)
        {
            if (string.IsNullOrEmpty(key))
            {
                return 1;
            }

            // Remove common suffixes
            var cleaned = key.ToLowerInvariant()
                .Replace("d", "")
                .Replace("days", "")
                .Replace("day", "")
                .Trim();

            if (int.TryParse(cleaned, out int horizon))
            {
                return horizon;
            }

            return 1; // Default to 1 day
        }

        private class PredictionResult
        {
            public string action { get; set; }
            public double confidence { get; set; }
            public double targetPrice { get; set; }
            public double currentPrice { get; set; }
            public double predictedPrice { get; set; }
            public double priceChangePct { get; set; }
            public Dictionary<string, double> featureWeights { get; set; }
        }
    }

    public class StockDataPoint
    {
        public DateTime Date { get; set; }
        public double Open { get; set; }
        public double High { get; set; }
        public double Low { get; set; }
        public double Close { get; set; }
        public double Volume { get; set; }
    }
}