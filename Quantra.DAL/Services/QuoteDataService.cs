using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Quantra.DAL.Services
{
    public class QuoteDataService
    {
        private AlphaVantageService _alphaVantageService;
        private LoggingService _loggingService;

        public QuoteDataService(AlphaVantageService alphaVantageService, LoggingService loggingService) 
        {
            _alphaVantageService = alphaVantageService;
            _loggingService = loggingService;
        }


        public async Task<QuoteData> GetLatestQuoteData(string symbol)
        {
            return await _alphaVantageService.GetQuoteDataAsync(symbol);
        }

        public async Task<List<QuoteData>> GetLatestQuoteData(IEnumerable<string> symbols)
        {
            var quoteDataList = new List<QuoteData>();
 
            try
            {
                // Create DbContext using SQL Server configuration
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Ensure database is created
                    dbContext.Database.EnsureCreated();

                    // Query StockDataCache for the latest cached data for each symbol
                    foreach (var symbol in symbols)
                    {
                        var latest = await dbContext.StockDataCache
                            .Where(c => c.Symbol == symbol)
                            .OrderByDescending(c => c.CachedAt)
                            .FirstOrDefaultAsync();

                        if (latest != null)
                        {
                            var storedData = latest.Data;
                            string jsonData;

                            // Check if data is compressed and decompress if needed
                            if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
                            {
                                jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
                            }
                            else
                            {
                                jsonData = storedData;
                            }

                            var prices = Newtonsoft.Json.JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);
                            if (prices != null && prices.Any())
                            {
                                var lastPrice = prices.Last();
                                quoteDataList.Add(new QuoteData
                                {
                                    Symbol = symbol,
                                    Price = lastPrice.Close,
                                    Date = lastPrice.Date,
                                    LastUpdated = latest.CachedAt,
                                    LastAccessed = DateTime.Now,
                                    Timestamp = lastPrice.Date,
                                    DayHigh = lastPrice.High,
                                    DayLow = lastPrice.Low,
                                    Volume = lastPrice.Volume,
                                    Change = 0, // Calculate from previous day if needed
                                    ChangePercent = 0,
                                    MarketCap = 0,
                                    RSI = 0,
                                    PERatio = 0,
                                    VWAP = 0
                                });
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogError($"Error retrieving latest quote data for multiple symbols", ex);
            }

            return quoteDataList;
        }

        public async Task<(QuoteData, DateTime?)> GetLatestQuoteDataWithTimestamp(string symbol)
        {
            QuoteData quoteData = null;
            DateTime? timestamp = null;

            try
            {
                // Create DbContext using SQL Server configuration
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Ensure database is created
                    dbContext.Database.EnsureCreated();

                    // Query StockDataCache for the latest cached data
                    var latest = await dbContext.StockDataCache
                        .Where(c => c.Symbol == symbol)
                        .OrderByDescending(c => c.CachedAt)
                        .FirstOrDefaultAsync();

                    if (latest != null)
                    {
                        timestamp = latest.CachedAt;
                        var storedData = latest.Data;
                        string jsonData;

                        // Check if data is compressed and decompress if needed
                        if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
                        {
                            jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
                        }
                        else
                        {
                            jsonData = storedData;
                        }

                        var prices = Newtonsoft.Json.JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);
                        if (prices != null && prices.Any())
                        {
                            var lastPrice = prices.Last();
                            quoteData = new QuoteData
                            {
                                Symbol = symbol,
                                Price = lastPrice.Close,
                                Date = lastPrice.Date,
                                LastUpdated = latest.CachedAt,
                                LastAccessed = DateTime.Now,
                                Timestamp = lastPrice.Date,
                                DayHigh = lastPrice.High,
                                DayLow = lastPrice.Low,
                                Volume = lastPrice.Volume,
                                Change = 0, // Calculate from previous day if needed
                                ChangePercent = 0,
                                MarketCap = 0,
                                RSI = 0,
                                PERatio = 0,
                                VWAP = 0
                            };
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogError($"Error retrieving latest quote data with timestamp for {symbol}", ex);
            }

            return (quoteData, timestamp);
        }
    }
}
