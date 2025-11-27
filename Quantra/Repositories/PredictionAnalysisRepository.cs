using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.Models;

namespace Quantra.Repositories
{
    public class PredictionAnalysisRepository
    {
        private readonly QuantraDbContext _context;

        public PredictionAnalysisRepository()
        {
            // Create DbContext with default connection
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();

            string sqlConn = Environment.GetEnvironmentVariable("QUANTRA_RELATIONAL_CONNECTION");
            if (string.IsNullOrWhiteSpace(sqlConn))
            {
                sqlConn = Environment.GetEnvironmentVariable("ConnectionStrings__QuantraRelational");
            }

            if (!string.IsNullOrWhiteSpace(sqlConn))
            {
                optionsBuilder.UseSqlServer(sqlConn);
            }
            else
            {
                // Fallback to connection helper
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            }

            _context = new QuantraDbContext(optionsBuilder.Options);
        }

        public PredictionAnalysisRepository(QuantraDbContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
        }

        public void SaveAnalysisResults(IEnumerable<PredictionAnalysisResult> results)
        {
            // Optionally implement DB save if needed
        }

        public List<PredictionAnalysisResult> GetLatestAnalyses(int count = 50)
        {
            // Use Task.Run to avoid deadlock - execute async method on background thread
            return Task.Run(async () => await GetLatestAnalysesAsync(count)).GetAwaiter().GetResult();
        }

        public async Task<List<PredictionAnalysisResult>> GetLatestAnalysesAsync(int count = 50)
        {
            var result = new List<PredictionAnalysisResult>();
            try
            {
                // Optimized approach: Get the latest date for each symbol in a single query,
                // then fetch the predictions. This is more efficient than the GroupBy approach.
                var latestDates = await _context.StockPredictions
                    .AsNoTracking()
                    .GroupBy(p => p.Symbol)
                    .Select(g => new { Symbol = g.Key, MaxDate = g.Max(p => p.CreatedDate) })
                    .ToListAsync();

                // Fetch predictions matching those symbol/date combinations
                var latestPredictions = new List<StockPredictionEntity>();
                foreach (var item in latestDates)
                {
                    var prediction = await _context.StockPredictions
                        .AsNoTracking()
                        .Include(p => p.Indicators)
                        .FirstOrDefaultAsync(p => p.Symbol == item.Symbol && p.CreatedDate == item.MaxDate);
                    
                    if (prediction != null)
                    {
                        latestPredictions.Add(prediction);
                    }
                }

                // Order by confidence and take the top results
                latestPredictions = latestPredictions
                    .OrderByDescending(p => p.Confidence)
                    .Take(count)
                    .ToList();

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

                    // Load indicators using the navigation property
                    if (prediction.Indicators != null && prediction.Indicators.Any())
                    {
                        foreach (var indicator in prediction.Indicators)
                        {
                            model.Indicators[indicator.IndicatorName] = indicator.IndicatorValue;
                        }
                    }

                    result.Add(model);
                }
            }
            catch (Exception ex)
            {
                // Log error - using simple console write since DatabaseMonolith is commented out
                Console.WriteLine($"Error retrieving latest analyses from database: {ex.Message}");
            }
            return result;
        }

        public List<string> GetSymbols()
        {
            // Use Task.Run to avoid deadlock - execute async method on background thread
            return Task.Run(async () => await GetSymbolsAsync()).GetAwaiter().GetResult();
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
            // Use Task.Run to avoid deadlock - execute async method on background thread
            return Task.Run(async () => await GetHistoricalPricesAsync(symbol)).GetAwaiter().GetResult();
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
                // Query the HistoricalPrices table using EF Core with parameterized SQL
                // Using FromSqlInterpolated for safe parameterization
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
                Console.WriteLine($"Error retrieving historical prices for {symbol}: {ex.Message}");
                return new List<HistoricalPrice>();
            }
        }
    }
}
