using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.Models;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing prediction analysis operations using Entity Framework
    /// </summary>
    public class PredictionAnalysisService
    {
        private readonly Func<QuantraDbContext> _contextFactory;
        private readonly LoggingService _loggingService;

        public PredictionAnalysisService(Func<QuantraDbContext> contextFactory, LoggingService loggingService)
        {
            _contextFactory = contextFactory ?? throw new ArgumentNullException(nameof(contextFactory));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        // Parameterless constructor for backward compatibility
        public PredictionAnalysisService()
        {
            // Create a factory that generates new DbContext instances for each operation
            _contextFactory = () =>
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString, sqlServerOptions =>
                {
                    sqlServerOptions.CommandTimeout(30);
                });
                return new QuantraDbContext(optionsBuilder.Options);
            };
            
            _loggingService = new LoggingService();
        }

        /// <summary>
        /// Gets the latest predictions from the database for each symbol
        /// </summary>
        /// <returns>List of the most recent prediction for each symbol, ordered by confidence descending</returns>
        public async Task<List<PredictionModel>> GetLatestPredictionsAsync()
        {
            try
            {
                using var context = _contextFactory();
                
                // Get all symbols first
                var symbols = await context.StockPredictions
                    .AsNoTracking()
                    .Select(p => p.Symbol)
                    .Distinct()
                    .ToListAsync()
                    .ConfigureAwait(false);

                var result = new List<PredictionModel>();

                // For each symbol, get the most recent prediction
                foreach (var symbol in symbols)
                {
                    var prediction = await context.StockPredictions
                        .AsNoTracking()
                        .Where(p => p.Symbol == symbol)
                        .OrderByDescending(p => p.CreatedDate)
                        .FirstOrDefaultAsync()
                        .ConfigureAwait(false);

                    if (prediction != null)
                    {
                        var model = new PredictionModel
                        {
                            Symbol = prediction.Symbol,
                            PredictedAction = prediction.PredictedAction,
                            Confidence = prediction.Confidence,
                            CurrentPrice = prediction.CurrentPrice,
                            TargetPrice = prediction.TargetPrice,
                            PotentialReturn = prediction.PotentialReturn,
                            PredictionDate = prediction.CreatedDate,
                            TradingRule = null,
                            Indicators = new Dictionary<string, double>()
                        };

                        // Load indicators for this prediction using EF Core
                        var indicators = await context.PredictionIndicators
                            .AsNoTracking()
                            .Where(i => i.PredictionId == prediction.Id)
                            .ToListAsync()
                            .ConfigureAwait(false);

                        foreach (var indicator in indicators)
                        {
                            model.Indicators[indicator.IndicatorName] = indicator.IndicatorValue;
                        }

                        result.Add(model);
                    }
                }

                // Sort by confidence descending in memory
                return result.OrderByDescending(p => p.Confidence).ToList();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to retrieve latest predictions from database", ex.ToString());
                return new List<PredictionModel>();
            }
        }

        /// <summary>
        /// Synchronous version of GetLatestPredictionsAsync for backward compatibility
        /// </summary>
        public List<PredictionModel> GetLatestPredictions()
        {
            return GetLatestPredictionsAsync().GetAwaiter().GetResult();
        }

        /// <summary>
        /// Gets predictions for a specific symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="count">Maximum number of predictions to return</param>
        /// <returns>List of predictions for the symbol, ordered by date descending</returns>
        public async Task<List<PredictionModel>> GetPredictionsForSymbolAsync(string symbol, int count = 10)
        {
            try
            {
                using var context = _contextFactory();
                
                var predictions = await context.StockPredictions
                    .AsNoTracking()
                    .Where(p => p.Symbol == symbol)
                    .OrderByDescending(p => p.CreatedDate)
                    .Take(count)
                    .ToListAsync()
                    .ConfigureAwait(false);

                var result = new List<PredictionModel>();

                foreach (var prediction in predictions)
                {
                    var model = new PredictionModel
                    {
                        Symbol = prediction.Symbol,
                        PredictedAction = prediction.PredictedAction,
                        Confidence = prediction.Confidence,
                        CurrentPrice = prediction.CurrentPrice,
                        TargetPrice = prediction.TargetPrice,
                        PotentialReturn = prediction.PotentialReturn,
                        PredictionDate = prediction.CreatedDate,
                        TradingRule = null, // TradingRule not stored in entity
                        Indicators = new Dictionary<string, double>()
                    };

                    // Load indicators
                    var indicators = await context.PredictionIndicators
                        .AsNoTracking()
                        .Where(i => i.PredictionId == prediction.Id)
                        .ToListAsync()
                        .ConfigureAwait(false);

                    foreach (var indicator in indicators)
                    {
                        model.Indicators[indicator.IndicatorName] = indicator.IndicatorValue;
                    }

                    result.Add(model);
                }

                return result;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to retrieve predictions for {symbol}", ex.ToString());
                return new List<PredictionModel>();
            }
        }

        /// <summary>
        /// Gets predictions based on action type (BUY, SELL, HOLD)
        /// </summary>
        /// <param name="action">Predicted action</param>
        /// <param name="minConfidence">Minimum confidence threshold</param>
        /// <returns>List of predictions matching the criteria</returns>
        public async Task<List<PredictionModel>> GetPredictionsByActionAsync(string action, double minConfidence = 0.0)
        {
            try
            {
                using var context = _contextFactory();
                
                // Get all symbols with the specified action and confidence
                var symbols = await context.StockPredictions
                    .AsNoTracking()
                    .Where(p => p.PredictedAction == action && p.Confidence >= minConfidence)
                    .Select(p => p.Symbol)
                    .Distinct()
                    .ToListAsync()
                    .ConfigureAwait(false);

                var result = new List<PredictionModel>();

                // For each symbol, get the most recent prediction with the specified action
                foreach (var symbol in symbols)
                {
                    var prediction = await context.StockPredictions
                        .AsNoTracking()
                        .Where(p => p.Symbol == symbol && p.PredictedAction == action && p.Confidence >= minConfidence)
                        .OrderByDescending(p => p.CreatedDate)
                        .FirstOrDefaultAsync()
                        .ConfigureAwait(false);

                    if (prediction != null)
                    {
                        var model = new PredictionModel
                        {
                            Symbol = prediction.Symbol,
                            PredictedAction = prediction.PredictedAction,
                            Confidence = prediction.Confidence,
                            CurrentPrice = prediction.CurrentPrice,
                            TargetPrice = prediction.TargetPrice,
                            PotentialReturn = prediction.PotentialReturn,
                            PredictionDate = prediction.CreatedDate,
                            TradingRule = null,
                            Indicators = new Dictionary<string, double>()
                        };

                        // Load indicators
                        var indicators = await context.PredictionIndicators
                            .AsNoTracking()
                            .Where(i => i.PredictionId == prediction.Id)
                            .ToListAsync()
                            .ConfigureAwait(false);

                        foreach (var indicator in indicators)
                        {
                            model.Indicators[indicator.IndicatorName] = indicator.IndicatorValue;
                        }

                        result.Add(model);
                    }
                }

                // Sort by confidence descending in memory
                return result.OrderByDescending(p => p.Confidence).ToList();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to retrieve predictions by action {action}", ex.ToString());
                return new List<PredictionModel>();
            }
        }

        /// <summary>
        /// Saves a prediction to the database using Entity Framework
        /// </summary>
        /// <param name="prediction">The prediction model to save</param>
        /// <param name="cancellationToken">Optional cancellation token for timeout control</param>
        /// <param name="expectedFruitionDate">Optional date when the prediction is expected to come to fruition</param>
        /// <param name="modelType">Optional model type used for the prediction</param>
        /// <param name="architectureType">Optional architecture type used for the prediction</param>
        /// <param name="trainingHistoryId">Optional training history ID reference</param>
        /// <returns>The ID of the saved prediction</returns>
        public async Task<int> SavePredictionAsync(
            PredictionModel prediction, 
            CancellationToken cancellationToken = default,
            DateTime? expectedFruitionDate = null,
            string modelType = null,
            string architectureType = null,
            int? trainingHistoryId = null)
        {
            if (prediction == null)
                throw new ArgumentNullException(nameof(prediction));

            // Validate required fields
            if (string.IsNullOrWhiteSpace(prediction.Symbol) ||
                string.IsNullOrWhiteSpace(prediction.PredictedAction) ||
                double.IsNaN(prediction.Confidence) || double.IsInfinity(prediction.Confidence) ||
                double.IsNaN(prediction.CurrentPrice) || double.IsInfinity(prediction.CurrentPrice) ||
                double.IsNaN(prediction.TargetPrice) || double.IsInfinity(prediction.TargetPrice) ||
                double.IsNaN(prediction.PotentialReturn) || double.IsInfinity(prediction.PotentialReturn))
            {
                throw new ArgumentException("Invalid prediction data. Required fields are missing or contain invalid values.");
            }

            try
            {
                _loggingService.Log("Info", $"SavePredictionAsync started for {prediction.Symbol}");

                // Create a new DbContext instance for this operation to avoid concurrency issues
                using var context = _contextFactory();
                
                // Ensure the stock symbol exists in the database
                var stockSymbol = await context.StockSymbols
                    .AsNoTracking()
                    .FirstOrDefaultAsync(s => s.Symbol == prediction.Symbol, cancellationToken)
                    .ConfigureAwait(false);

                if (stockSymbol == null)
                {
                    _loggingService.Log("Info", $"Creating new stock symbol entry for {prediction.Symbol}");
                    stockSymbol = new StockSymbolEntity
                    {
                        Symbol = prediction.Symbol,
                        LastUpdated = DateTime.Now
                    };
                    context.StockSymbols.Add(stockSymbol);
                    await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
                    _loggingService.Log("Info", $"Stock symbol {prediction.Symbol} created successfully");
                }
                else
                {
                    _loggingService.Log("Info", $"Stock symbol {prediction.Symbol} already exists");
                }

                // Create prediction entity with new fields
                var predictionEntity = new StockPredictionEntity
                {
                    Symbol = prediction.Symbol,
                    PredictedAction = prediction.PredictedAction,
                    Confidence = prediction.Confidence,
                    CurrentPrice = prediction.CurrentPrice,
                    TargetPrice = prediction.TargetPrice,
                    PotentialReturn = prediction.PotentialReturn,
                    CreatedDate = DateTime.Now,
                    ExpectedFruitionDate = expectedFruitionDate,
                    ModelType = modelType,
                    ArchitectureType = architectureType,
                    TrainingHistoryId = trainingHistoryId,
                    TradingRule = prediction.TradingRule
                };

                context.StockPredictions.Add(predictionEntity);
                _loggingService.Log("Info", $"Saving prediction entity for {prediction.Symbol}");
                await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
                _loggingService.Log("Info", $"Prediction entity saved with ID {predictionEntity.Id}");

                // Save indicators if any
                if (prediction.Indicators != null && prediction.Indicators.Any())
                {
                    _loggingService.Log("Info", $"Saving {prediction.Indicators.Count} indicators for prediction {predictionEntity.Id}");
                    foreach (var indicator in prediction.Indicators)
                    {
                        var indicatorEntity = new PredictionIndicatorEntity
                        {
                            PredictionId = predictionEntity.Id,
                            IndicatorName = indicator.Key,
                            IndicatorValue = indicator.Value
                        };
                        context.PredictionIndicators.Add(indicatorEntity);
                    }
                    await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
                    _loggingService.Log("Info", $"Indicators saved successfully for prediction {predictionEntity.Id}");
                }
                else
                {
                    _loggingService.Log("Warning", $"No indicators to save for prediction {predictionEntity.Id}");
                }

                _loggingService.Log("Info", $"SavePredictionAsync completed successfully for {prediction.Symbol}, ID: {predictionEntity.Id}");
                return predictionEntity.Id;
            }
            catch (OperationCanceledException)
            {
                _loggingService.Log("Warning", $"Save operation timed out for {prediction.Symbol}", "Operation was cancelled or timed out");
                throw;
            }
            catch (DbUpdateException dbEx)
            {
                _loggingService.Log("Error", $"Database error saving prediction for {prediction.Symbol}", dbEx.ToString());
                throw new InvalidOperationException($"Database error: {dbEx.Message}", dbEx);
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to save prediction for {prediction.Symbol}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Deletes old predictions older than specified date
        /// </summary>
        /// <param name="olderThan">Delete predictions older than this date</param>
        /// <returns>Number of predictions deleted</returns>
        public async Task<int> DeleteOldPredictionsAsync(DateTime olderThan)
        {
            try
            {
                using var context = _contextFactory();
                
                var oldPredictions = await context.StockPredictions
                    .Where(p => p.CreatedDate < olderThan)
                    .ToListAsync()
                    .ConfigureAwait(false);

                if (oldPredictions.Any())
                {
                    // Delete associated indicators first
                    var predictionIds = oldPredictions.Select(p => p.Id).ToList();
                    var indicators = await context.PredictionIndicators
                        .Where(i => predictionIds.Contains(i.PredictionId))
                        .ToListAsync()
                        .ConfigureAwait(false);

                    context.PredictionIndicators.RemoveRange(indicators);
                    context.StockPredictions.RemoveRange(oldPredictions);

                    await context.SaveChangesAsync().ConfigureAwait(false);

                    _loggingService.Log("Info", $"Deleted {oldPredictions.Count} old predictions");
                    return oldPredictions.Count;
                }

                return 0;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to delete old predictions", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Gets all predictions from the database with full entity data including model type and expected fruition date
        /// </summary>
        /// <param name="count">Maximum number of predictions to return (default 1000)</param>
        /// <returns>List of all predictions, ordered by creation date descending</returns>
        public async Task<List<PredictionModel>> GetAllPredictionsAsync(int count = 1000)
        {
            try
            {
                using var context = _contextFactory();
                
                var predictions = await context.StockPredictions
                    .AsNoTracking()
                    .OrderByDescending(p => p.CreatedDate)
                    .Take(count)
                    .ToListAsync()
                    .ConfigureAwait(false);

                var result = new List<PredictionModel>();

                foreach (var prediction in predictions)
                {
                    var model = new PredictionModel
                    {
                        Id = prediction.Id,
                        Symbol = prediction.Symbol,
                        PredictedAction = prediction.PredictedAction,
                        Confidence = prediction.Confidence,
                        CurrentPrice = prediction.CurrentPrice,
                        TargetPrice = prediction.TargetPrice,
                        PotentialReturn = prediction.PotentialReturn,
                        PredictionDate = prediction.CreatedDate,
                        TradingRule = prediction.TradingRule,
                        UserQuery = prediction.UserQuery,
                        ChatHistoryId = prediction.ChatHistoryId,
                        ExpectedFruitionDate = prediction.ExpectedFruitionDate,
                        ModelType = prediction.ModelType,
                        ArchitectureType = prediction.ArchitectureType,
                        TrainingHistoryId = prediction.TrainingHistoryId,
                        Indicators = new Dictionary<string, double>()
                    };

                    result.Add(model);
                }

                return result;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to retrieve all predictions from database", ex.ToString());
                return new List<PredictionModel>();
            }
        }

        /// <summary>
        /// Synchronous version of GetAllPredictionsAsync for backward compatibility
        /// </summary>
        public List<PredictionModel> GetAllPredictions(int count = 1000)
        {
            return GetAllPredictionsAsync(count).GetAwaiter().GetResult();
        }
    }
}
