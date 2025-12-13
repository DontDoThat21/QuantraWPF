using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing stock data displayed in the StockExplorer view.
    /// This service handles saving and retrieving stock data from the StockExplorerData table.
    /// </summary>
    public class StockExplorerDataService
    {
        private readonly IDbContextFactory<QuantraDbContext> _dbContextFactory;
        private readonly LoggingService _loggingService;

        public StockExplorerDataService(
            IDbContextFactory<QuantraDbContext> dbContextFactory,
            LoggingService loggingService)
        {
            _dbContextFactory = dbContextFactory ?? throw new ArgumentNullException(nameof(dbContextFactory));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        /// <summary>
        /// Saves or updates stock data in the StockExplorerData table
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="name">Company name</param>
        /// <param name="price">Current price</param>
        /// <param name="change">Price change</param>
        /// <param name="changePercent">Price change percentage</param>
        /// <param name="dayHigh">Day's high price</param>
        /// <param name="dayLow">Day's low price</param>
        /// <param name="marketCap">Market capitalization</param>
        /// <param name="volume">Trading volume</param>
        /// <param name="sector">Stock sector</param>
        /// <param name="rsi">RSI indicator value</param>
        /// <param name="peRatio">P/E ratio</param>
        /// <param name="vwap">VWAP value</param>
        public async Task SaveStockDataAsync(
            string symbol,
            string name = null,
            double price = 0,
            double change = 0,
            double changePercent = 0,
            double dayHigh = 0,
            double dayLow = 0,
            double marketCap = 0,
            double volume = 0,
            string sector = null,
            double rsi = 0,
            double peRatio = 0,
            double vwap = 0)
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();

                var now = DateTime.Now;

                // Check if stock data already exists
                var existingData = await context.StockExplorerData
                    .FirstOrDefaultAsync(s => s.Symbol == symbol);

                if (existingData != null)
                {
                    // Update existing record
                    existingData.Name = name ?? existingData.Name;
                    existingData.Price = price;
                    existingData.Change = change;
                    existingData.ChangePercent = changePercent;
                    existingData.DayHigh = dayHigh;
                    existingData.DayLow = dayLow;
                    existingData.MarketCap = marketCap;
                    existingData.Volume = volume;
                    existingData.Sector = sector ?? existingData.Sector;
                    existingData.RSI = rsi;
                    existingData.PERatio = peRatio;
                    existingData.VWAP = vwap;
                    existingData.LastUpdated = now;
                    existingData.LastAccessed = now;
                    existingData.Timestamp = now;
                    existingData.CacheTime = now;
                }
                else
                {
                    // Create new record
                    var newData = new StockExplorerDataEntity
                    {
                        Symbol = symbol,
                        Name = name ?? string.Empty,
                        Price = price,
                        Change = change,
                        ChangePercent = changePercent,
                        DayHigh = dayHigh,
                        DayLow = dayLow,
                        MarketCap = marketCap,
                        Volume = volume,
                        Sector = sector ?? string.Empty,
                        RSI = rsi,
                        PERatio = peRatio,
                        VWAP = vwap,
                        Date = now,
                        LastUpdated = now,
                        LastAccessed = now,
                        Timestamp = now,
                        CacheTime = now
                    };

                    context.StockExplorerData.Add(newData);
                }

                await context.SaveChangesAsync();
                _loggingService?.Log("Info", $"Saved stock data for {symbol} to StockExplorerData table");
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to save stock data for {symbol}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Saves stock data from a QuoteData object
        /// </summary>
        public async Task SaveStockDataFromQuoteAsync(QuoteData quoteData)
        {
            if (quoteData == null)
            {
                throw new ArgumentNullException(nameof(quoteData));
            }

            await SaveStockDataAsync(
                symbol: quoteData.Symbol,
                name: quoteData.Name,
                price: quoteData.Price,
                change: quoteData.Change,
                changePercent: quoteData.ChangePercent,
                dayHigh: quoteData.DayHigh,
                dayLow: quoteData.DayLow,
                marketCap: quoteData.MarketCap,
                volume: quoteData.Volume,
                sector: quoteData.Sector,
                rsi: quoteData.RSI,
                peRatio: quoteData.PERatio,
                vwap: quoteData.VWAP
            );
        }

        /// <summary>
        /// Saves multiple stock data records in a batch operation
        /// </summary>
        public async Task SaveStockDataBatchAsync(IEnumerable<QuoteData> quoteDataList)
        {
            if (quoteDataList == null || !quoteDataList.Any())
            {
                return;
            }

            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();

                var now = DateTime.Now;
                var symbols = quoteDataList.Select(q => q.Symbol).Distinct().ToList();

                // Get existing records for all symbols
                var existingData = await context.StockExplorerData
                    .Where(s => symbols.Contains(s.Symbol))
                    .ToDictionaryAsync(s => s.Symbol);

                foreach (var quoteData in quoteDataList)
                {
                    if (existingData.TryGetValue(quoteData.Symbol, out var existing))
                    {
                        // Update existing record
                        existing.Name = quoteData.Name ?? existing.Name;
                        existing.Price = quoteData.Price;
                        existing.Change = quoteData.Change;
                        existing.ChangePercent = quoteData.ChangePercent;
                        existing.DayHigh = quoteData.DayHigh;
                        existing.DayLow = quoteData.DayLow;
                        existing.MarketCap = quoteData.MarketCap;
                        existing.Volume = quoteData.Volume;
                        existing.Sector = quoteData.Sector ?? existing.Sector;
                        existing.RSI = quoteData.RSI;
                        existing.PERatio = quoteData.PERatio;
                        existing.VWAP = quoteData.VWAP;
                        existing.LastUpdated = now;
                        existing.LastAccessed = now;
                        existing.Timestamp = now;
                        existing.CacheTime = now;
                    }
                    else
                    {
                        // Create new record
                        var newData = new StockExplorerDataEntity
                        {
                            Symbol = quoteData.Symbol,
                            Name = quoteData.Name ?? string.Empty,
                            Price = quoteData.Price,
                            Change = quoteData.Change,
                            ChangePercent = quoteData.ChangePercent,
                            DayHigh = quoteData.DayHigh,
                            DayLow = quoteData.DayLow,
                            MarketCap = quoteData.MarketCap,
                            Volume = quoteData.Volume,
                            Sector = quoteData.Sector ?? string.Empty,
                            RSI = quoteData.RSI,
                            PERatio = quoteData.PERatio,
                            VWAP = quoteData.VWAP,
                            Date = now,
                            LastUpdated = now,
                            LastAccessed = now,
                            Timestamp = now,
                            CacheTime = now
                        };

                        context.StockExplorerData.Add(newData);
                    }
                }

                await context.SaveChangesAsync();
                _loggingService?.Log("Info", $"Saved batch of {quoteDataList.Count()} stock data records to StockExplorerData table");
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Failed to save stock data batch", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Gets stock data for a specific symbol
        /// </summary>
        public async Task<StockExplorerDataEntity> GetStockDataAsync(string symbol)
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();

                var data = await context.StockExplorerData
                    .AsNoTracking()
                    .FirstOrDefaultAsync(s => s.Symbol == symbol);

                // Update last accessed time if found
                if (data != null)
                {
                    using var updateContext = await _dbContextFactory.CreateDbContextAsync();
                    var entity = await updateContext.StockExplorerData.FindAsync(data.Id);
                    if (entity != null)
                    {
                        entity.LastAccessed = DateTime.Now;
                        await updateContext.SaveChangesAsync();
                    }
                }

                return data;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to get stock data for {symbol}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Gets all stock data from the table
        /// </summary>
        public async Task<List<StockExplorerDataEntity>> GetAllStockDataAsync()
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();

                return await context.StockExplorerData
                    .AsNoTracking()
                    .OrderBy(s => s.Symbol)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Failed to get all stock data", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Checks if stock data exists for a given symbol
        /// </summary>
        public async Task<bool> HasStockDataAsync(string symbol)
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();

                return await context.StockExplorerData
                    .AnyAsync(s => s.Symbol == symbol);
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to check if stock data exists for {symbol}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Gets all symbols that have data in the StockExplorerData table
        /// </summary>
        public async Task<List<string>> GetAllSymbolsWithDataAsync()
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();

                return await context.StockExplorerData
                    .AsNoTracking()
                    .Select(s => s.Symbol)
                    .Distinct()
                    .OrderBy(s => s)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Failed to get symbols with data", ex.ToString());
                return new List<string>();
            }
        }

        /// <summary>
        /// Deletes stock data for a specific symbol
        /// </summary>
        public async Task DeleteStockDataAsync(string symbol)
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();

                var data = await context.StockExplorerData
                    .FirstOrDefaultAsync(s => s.Symbol == symbol);

                if (data != null)
                {
                    context.StockExplorerData.Remove(data);
                    await context.SaveChangesAsync();
                    _loggingService?.Log("Info", $"Deleted stock data for {symbol}");
                }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to delete stock data for {symbol}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Clears all stock data from the table
        /// </summary>
        public async Task ClearAllDataAsync()
        {
            try
            {
                using var context = await _dbContextFactory.CreateDbContextAsync();

                await context.Database.ExecuteSqlRawAsync("DELETE FROM StockExplorerData");
                _loggingService?.Log("Info", "Cleared all stock data from StockExplorerData table");
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Failed to clear all stock data", ex.ToString());
                throw;
            }
        }
    }
}
