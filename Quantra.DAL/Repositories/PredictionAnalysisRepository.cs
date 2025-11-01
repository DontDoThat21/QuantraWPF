using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.Models;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.Repositories
{
    public class PredictionAnalysisRepository
    {
        private readonly QuantraDbContext _context;

        public PredictionAnalysisRepository(QuantraDbContext context)
        {
            _context = context;
        }

        // Parameterless constructor for backward compatibility
        public PredictionAnalysisRepository()
        {
            // For legacy code that doesn't use DI
            // Prefer a relational SQL Server connection named "QuantraRelational" provided
            // via environment variable or standard ConnectionStrings__QuantraRelational.
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();

            string sqlConn = Environment.GetEnvironmentVariable("QUANTRA_RELATIONAL_CONNECTION");
            if (string.IsNullOrWhiteSpace(sqlConn))
            {
                // Try ASP.NET style JSON environment variable name
                sqlConn = Environment.GetEnvironmentVariable("ConnectionStrings__QuantraRelational");
            }

            if (!string.IsNullOrWhiteSpace(sqlConn))
            {
                // Use SQL Server if connection string provided
                optionsBuilder.UseSqlServer(sqlConn);
            }
            else
            {
                // Fallback to local SQLite database for compatibility
                optionsBuilder.UseSqlite("Data Source=Quantra.db");
            }

            _context = new QuantraDbContext(optionsBuilder.Options);
        }

        public void SaveAnalysisResults(IEnumerable<PredictionAnalysisResult> results)
        {
            // Optionally implement DB save if needed
        }

        public List<PredictionAnalysisResult> GetLatestAnalyses(int count = 50)
        {
            return GetLatestAnalysesAsync(count).GetAwaiter().GetResult();
        }

        public async Task<List<PredictionAnalysisResult>> GetLatestAnalysesAsync(int count = 50)
        {
            var result = new List<PredictionAnalysisResult>();
            try
            {
                // Query using LINQ with EF Core
                var latestPredictions = await _context.StockPredictions
                    .AsNoTracking()
                    .GroupBy(p => p.Symbol)
                    .Select(g => g.OrderByDescending(p => p.CreatedDate).FirstOrDefault())
                    .Where(p => p != null)
                    .OrderByDescending(p => p.Confidence)
                    .Take(count)
                    .ToListAsync();

                foreach (var prediction in latestPredictions)
                {
                    var model = new PredictionAnalysisResult
                    {
                        Id = prediction.Id,
                        Symbol = prediction.Symbol,
                        PredictedAction = prediction.PredictedAction,
                        Confidence = prediction.Confidence,
                        CurrentPrice = prediction.CurrentPrice,
                        TargetPrice = prediction.TargetPrice,
                        PotentialReturn = prediction.PotentialReturn,
                        TradingRule = null, // Not stored in entity
                        AnalysisTime = prediction.CreatedDate,
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
            catch (Exception ex)
            {
                Console.WriteLine($"Error retrieving latest analyses from database: {ex.Message}");
            }
            return result;
        }

        public List<string> GetSymbols()
        {
            return GetSymbolsAsync().GetAwaiter().GetResult();
        }

        public async Task<List<string>> GetSymbolsAsync()
        {
            try
            {
                return await _context.StockPredictions
                    .AsNoTracking()
                    .Select(p => p.Symbol)
                    .Distinct()
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error retrieving symbols from database: {ex.Message}");
                return new List<string>();
            }
        }

        public PredictionAnalysisResult AnalyzeSymbol(string symbol)
        {
            // Optionally implement DB-backed or ML-backed analysis
            return null;
        }

        // Overload to support strategy profile
        public PredictionAnalysisResult AnalyzeSymbol(string symbol, StrategyProfile strategy)
        {
            // Use the provided strategy for analysis
            // Example: run backtest or generate signals using the strategy
            var historical = GetHistoricalPrices(symbol); // Use existing method
            var signal = strategy.GenerateSignal(historical, historical.Count - 1);
            // ...rest of analysis logic...
            return new PredictionAnalysisResult
            {
                Symbol = symbol,
                PredictedAction = signal ?? "HOLD",
                Confidence = 0.75, // Example
                CurrentPrice = historical.LastOrDefault()?.Close ?? 0,
                TargetPrice = (historical.LastOrDefault()?.Close ?? 0) * 1.05,
                PotentialReturn = 0.05,
                TradingRule = strategy.Name,
                Indicators = new Dictionary<string, double>()
            };
        }

        // Returns historical price data for a symbol, ordered by date ascending
        public List<HistoricalPrice> GetHistoricalPrices(string symbol)
        {
            return GetHistoricalPricesAsync(symbol).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Returns historical price data for a symbol, ordered by date ascending (async version)
        /// Uses Entity Framework Core to query the HistoricalPrices table.
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>List of historical prices ordered by date</returns>
        public async Task<List<HistoricalPrice>> GetHistoricalPricesAsync(string symbol)
        {
            try
            {
                // Query the HistoricalPrices table using raw SQL via EF Core
                // This assumes the HistoricalPrices table exists in the database
                // Using SqlQuery with FormattableString for proper parameterization
                var prices = await _context.Database
                    .SqlQuery<HistoricalPrice>($@"
                         SELECT Date, Open, High, Low, Close, Volume, AdjClose 
                         FROM HistoricalPrices 
                         WHERE Symbol = {symbol} 
                         ORDER BY Date ASC")
                    .ToListAsync();

                return prices;
            }
            catch (Exception ex)
            {
                // Log error if logging is available
                Console.WriteLine($"Error retrieving historical prices for {symbol}: {ex.Message}");
                return new List<HistoricalPrice>();
            }
        }
    }
}
