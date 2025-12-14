using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public class StockMetricsCalculationService
    {
        private readonly IDbContextFactory<QuantraDbContext> _dbContextFactory;
        private readonly StockDataCacheService _stockDataCacheService;
        private readonly TechnicalIndicatorService _technicalIndicatorService;
        private readonly AlphaVantageService _alphaVantageService;
        private readonly StockExplorerDataService _stockExplorerDataService;
        private readonly LoggingService _loggingService;

        private const int BATCH_SIZE = 50;
        private const int MAX_CONCURRENT_TASKS = 5;
        private readonly SemaphoreSlim _throttle;

        public event EventHandler<MetricsCalculationProgressEventArgs> ProgressChanged;

        public StockMetricsCalculationService(
            IDbContextFactory<QuantraDbContext> dbContextFactory,
            StockDataCacheService stockDataCacheService,
            TechnicalIndicatorService technicalIndicatorService,
            AlphaVantageService alphaVantageService,
            StockExplorerDataService stockExplorerDataService,
            LoggingService loggingService)
        {
            _dbContextFactory = dbContextFactory ?? throw new ArgumentNullException(nameof(dbContextFactory));
            _stockDataCacheService = stockDataCacheService ?? throw new ArgumentNullException(nameof(stockDataCacheService));
            _technicalIndicatorService = technicalIndicatorService ?? throw new ArgumentNullException(nameof(technicalIndicatorService));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _stockExplorerDataService = stockExplorerDataService ?? throw new ArgumentNullException(nameof(stockExplorerDataService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));

            _throttle = new SemaphoreSlim(MAX_CONCURRENT_TASKS, MAX_CONCURRENT_TASKS);
        }

        public async Task CalculateAllMetricsAsync(CancellationToken cancellationToken = default)
        {
            _loggingService.Log("Info", "Starting metrics calculation for all symbols...");

            try
            {
                var symbols = await GetAllSymbolsAsync();
                _loggingService.Log("Info", $"Found {symbols.Count} symbols to process");

                int totalSymbols = symbols.Count;
                int processedCount = 0;
                int successCount = 0;
                int errorCount = 0;

                for (int i = 0; i < symbols.Count; i += BATCH_SIZE)
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        _loggingService.Log("Info", "Metrics calculation cancelled by user");
                        break;
                    }

                    var batch = symbols.Skip(i).Take(BATCH_SIZE).ToList();
                    var tasks = batch.Select(symbol => ProcessSymbolAsync(symbol, cancellationToken)).ToList();

                    var results = await Task.WhenAll(tasks);
                    
                    processedCount += batch.Count;
                    successCount += results.Count(r => r);
                    errorCount += results.Count(r => !r);

                    OnProgressChanged(new MetricsCalculationProgressEventArgs
                    {
                        TotalSymbols = totalSymbols,
                        ProcessedSymbols = processedCount,
                        SuccessCount = successCount,
                        ErrorCount = errorCount
                    });

                    _loggingService.Log("Info", $"Processed batch {i / BATCH_SIZE + 1}: {processedCount}/{totalSymbols} symbols ({successCount} success, {errorCount} errors)");
                }

                _loggingService.Log("Info", $"Metrics calculation completed: {successCount} success, {errorCount} errors out of {totalSymbols} total");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error in CalculateAllMetricsAsync", ex.ToString());
                throw;
            }
        }

        public async Task CalculateMetricsForSymbolAsync(string symbol, CancellationToken cancellationToken = default)
        {
            await ProcessSymbolAsync(symbol, cancellationToken);
        }

        private async Task<bool> ProcessSymbolAsync(string symbol, CancellationToken cancellationToken)
        {
            await _throttle.WaitAsync(cancellationToken);

            try
            {
                var prices = await GetHistoricalPricesAsync(symbol);
                
                if (prices == null || prices.Count < 14)
                {
                    return false;
                }

                var rsi = await CalculateRSIAsync(symbol, prices);
                var vwap = CalculateVWAP(prices);
                var marketCap = await GetMarketCapAsync(symbol);
                var volume = prices.LastOrDefault()?.Volume ?? 0;
                var price = prices.LastOrDefault()?.Close ?? 0;
                var change = prices.Count >= 2 ? prices.Last().Close - prices[prices.Count - 2].Close : 0;
                var changePercent = prices.Count >= 2 && prices[prices.Count - 2].Close != 0 
                    ? ((prices.Last().Close - prices[prices.Count - 2].Close) / prices[prices.Count - 2].Close) * 100 
                    : 0;

                var name = await GetSymbolNameAsync(symbol);
                var sector = await GetSymbolSectorAsync(symbol);
                var peRatio = await GetPERatioAsync(symbol);

                await _stockExplorerDataService.SaveStockDataAsync(
                    symbol: symbol,
                    name: name,
                    price: price,
                    change: change,
                    changePercent: changePercent,
                    dayHigh: prices.LastOrDefault()?.High ?? 0,
                    dayLow: prices.LastOrDefault()?.Low ?? 0,
                    marketCap: marketCap,
                    volume: volume,
                    sector: sector,
                    rsi: rsi,
                    peRatio: peRatio,
                    vwap: vwap
                );

                return true;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error processing symbol {symbol}", ex.ToString());
                return false;
            }
            finally
            {
                _throttle.Release();
            }
        }

        private async Task<List<string>> GetAllSymbolsAsync()
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();
                return await context.StockDataCache
                    .Select(c => c.Symbol)
                    .Distinct()
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error getting all symbols", ex.ToString());
                return new List<string>();
            }
        }

        private async Task<List<HistoricalPrice>> GetHistoricalPricesAsync(string symbol)
        {
            try
            {
                return await _stockDataCacheService.GetStockData(symbol, range: "1mo", interval: "1d", forceRefresh: false);
            }
            catch (Exception ex)
            {
                _loggingService.Log("Warning", $"Could not get historical data for {symbol}", ex.Message);
                return new List<HistoricalPrice>();
            }
        }

        private async Task<double> CalculateRSIAsync(string symbol, List<HistoricalPrice> prices)
        {
            try
            {
                if (prices == null || prices.Count < 14)
                {
                    return 0;
                }

                var closePrices = prices.Select(p => p.Close).ToList();
                var rsiValues = _technicalIndicatorService.CalculateRSI(closePrices, 14);
                
                return rsiValues.LastOrDefault();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Warning", $"Could not calculate RSI for {symbol}", ex.Message);
                return 0;
            }
        }

        private double CalculateVWAP(List<HistoricalPrice> prices)
        {
            try
            {
                if (prices == null || prices.Count == 0)
                {
                    return 0;
                }

                var highPrices = prices.Select(p => p.High).ToList();
                var lowPrices = prices.Select(p => p.Low).ToList();
                var closePrices = prices.Select(p => p.Close).ToList();
                var volumes = prices.Select(p => (double)p.Volume).ToList();

                var vwapValues = _technicalIndicatorService.CalculateVWAP(highPrices, lowPrices, closePrices, volumes);
                
                return vwapValues.LastOrDefault();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Warning", $"Could not calculate VWAP", ex.Message);
                return 0;
            }
        }

        private async Task<double> GetMarketCapAsync(string symbol)
        {
            try
            {
                var overview = await _alphaVantageService.GetCompanyOverviewAsync(symbol);
                if (overview != null && overview.MarketCapitalization.HasValue)
                {
                    return (double)overview.MarketCapitalization.Value;
                }
                return 0;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Warning", $"Could not get market cap for {symbol}", ex.Message);
                return 0;
            }
        }

        private async Task<string> GetSymbolNameAsync(string symbol)
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();
                var stockSymbol = await context.StockSymbols
                    .AsNoTracking()
                    .FirstOrDefaultAsync(s => s.Symbol == symbol);
                
                return stockSymbol?.Name ?? string.Empty;
            }
            catch (Exception)
            {
                return string.Empty;
            }
        }

        private async Task<string> GetSymbolSectorAsync(string symbol)
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();
                var stockSymbol = await context.StockSymbols
                    .AsNoTracking()
                    .FirstOrDefaultAsync(s => s.Symbol == symbol);
                
                return stockSymbol?.Sector ?? string.Empty;
            }
            catch (Exception)
            {
                return string.Empty;
            }
        }

        private async Task<double> GetPERatioAsync(string symbol)
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();
                var fundamental = await context.FundamentalDataCache
                    .AsNoTracking()
                    .FirstOrDefaultAsync(f => f.Symbol == symbol && f.DataType == "PERatio");
                
                return fundamental?.Value ?? 0;
            }
            catch (Exception)
            {
                return 0;
            }
        }

        protected virtual void OnProgressChanged(MetricsCalculationProgressEventArgs e)
        {
            ProgressChanged?.Invoke(this, e);
        }
    }

    public class MetricsCalculationProgressEventArgs : EventArgs
    {
        public int TotalSymbols { get; set; }
        public int ProcessedSymbols { get; set; }
        public int SuccessCount { get; set; }
        public int ErrorCount { get; set; }
        public double ProgressPercentage => TotalSymbols > 0 ? (double)ProcessedSymbols / TotalSymbols * 100 : 0;
    }
}
