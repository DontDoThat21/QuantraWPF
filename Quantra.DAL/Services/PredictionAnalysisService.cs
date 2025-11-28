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
        private readonly QuantraDbContext _context;
        private LoggingService _loggingService;

        public PredictionAnalysisService(QuantraDbContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _loggingService = new LoggingService();
        }

        // Parameterless constructor for backward compatibility
        public PredictionAnalysisService()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();

            // Use SQL Server with QuantraRelational database via ConnectionHelper
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString, sqlServerOptions =>
            {
                sqlServerOptions.CommandTimeout(30);
            });

            _context = new QuantraDbContext(optionsBuilder.Options);
        }

        /// <summary>
        /// Gets the latest predictions from the database for each symbol
        /// </summary>
        /// <returns>List of the most recent prediction for each symbol, ordered by confidence descending</returns>
        public async Task<List<PredictionModel>> GetLatestPredictionsAsync()
        {
            try
            {
                // Get all symbols first
                var symbols = await _context.StockPredictions
                    .AsNoTracking()
                    .Select(p => p.Symbol)
                    .Distinct()
                    .ToListAsync();

                var result = new List<PredictionModel>();

                // For each symbol, get the most recent prediction
                foreach (var symbol in symbols)
                {
                    var prediction = await _context.StockPredictions
                        .AsNoTracking()
                        .Where(p => p.Symbol == symbol)
                        .OrderByDescending(p => p.CreatedDate)
                        .FirstOrDefaultAsync();

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
                        var indicators = await _context.PredictionIndicators
                            .AsNoTracking()
                            .Where(i => i.PredictionId == prediction.Id)
                            .ToListAsync();

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
                var predictions = await _context.StockPredictions
                   .AsNoTracking()
                   .Where(p => p.Symbol == symbol)
                    .OrderByDescending(p => p.CreatedDate)
             .Take(count)
                        .ToListAsync();

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
                    var indicators = await _context.PredictionIndicators
                   .AsNoTracking()
                .Where(i => i.PredictionId == prediction.Id)
                          .ToListAsync();

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
                // Get all symbols with the specified action and confidence
                var symbols = await _context.StockPredictions
                    .AsNoTracking()
                    .Where(p => p.PredictedAction == action && p.Confidence >= minConfidence)
                    .Select(p => p.Symbol)
                    .Distinct()
                    .ToListAsync();

                var result = new List<PredictionModel>();

                // For each symbol, get the most recent prediction with the specified action
                foreach (var symbol in symbols)
                {
                    var prediction = await _context.StockPredictions
                        .AsNoTracking()
                        .Where(p => p.Symbol == symbol && p.PredictedAction == action && p.Confidence >= minConfidence)
                        .OrderByDescending(p => p.CreatedDate)
                        .FirstOrDefaultAsync();

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
                        var indicators = await _context.PredictionIndicators
                            .AsNoTracking()
                            .Where(i => i.PredictionId == prediction.Id)
                            .ToListAsync();

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
        /// <returns>The ID of the saved prediction</returns>
        public async Task<int> SavePredictionAsync(PredictionModel prediction)
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
                // Ensure the stock symbol exists in the database
                var stockSymbol = await _context.StockSymbols
                 .FirstOrDefaultAsync(s => s.Symbol == prediction.Symbol);

                if (stockSymbol == null)
                {
                    stockSymbol = new StockSymbolEntity
                    {
                        Symbol = prediction.Symbol,
                        LastUpdated = DateTime.Now
                    };
                    _context.StockSymbols.Add(stockSymbol);
                    await _context.SaveChangesAsync();
                }

                // Create prediction entity (TradingRule not saved since it's not in the entity)
                var predictionEntity = new StockPredictionEntity
                {
                    Symbol = prediction.Symbol,
                    PredictedAction = prediction.PredictedAction,
                    Confidence = prediction.Confidence,
                    CurrentPrice = prediction.CurrentPrice,
                    TargetPrice = prediction.TargetPrice,
                    PotentialReturn = prediction.PotentialReturn,
                    CreatedDate = DateTime.Now
                };

                _context.StockPredictions.Add(predictionEntity);
                await _context.SaveChangesAsync();

                // Save indicators if any
                if (prediction.Indicators != null && prediction.Indicators.Any())
                {
                    foreach (var indicator in prediction.Indicators)
                    {
                        var indicatorEntity = new PredictionIndicatorEntity
                        {
                            PredictionId = predictionEntity.Id,
                            IndicatorName = indicator.Key,
                            IndicatorValue = indicator.Value
                        };
                        _context.PredictionIndicators.Add(indicatorEntity);
                    }
                    await _context.SaveChangesAsync();
                }

                return predictionEntity.Id;
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
                var oldPredictions = await _context.StockPredictions
             .Where(p => p.CreatedDate < olderThan)
                   .ToListAsync();

                if (oldPredictions.Any())
                {
                    // Delete associated indicators first
                    var predictionIds = oldPredictions.Select(p => p.Id).ToList();
                    var indicators = await _context.PredictionIndicators
                               .Where(i => predictionIds.Contains(i.PredictionId))
                  .ToListAsync();

                    _context.PredictionIndicators.RemoveRange(indicators);
                    _context.StockPredictions.RemoveRange(oldPredictions);

                    await _context.SaveChangesAsync();

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
    }
}
