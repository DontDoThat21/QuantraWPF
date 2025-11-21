using Xunit;
using Quantra.DAL.Services.Interfaces;
using Quantra;
using System;
using Quantra.DAL.Services;

namespace Quantra.Tests.Services
{
    public class StockDataCacheServiceTests
    {
        private StockDataCacheService _cacheService;

        public StockDataCacheServiceTests()
        {
            _cacheService = new ContinStockDataCacheService();
        }

        [Fact]
        public void CacheQuoteData_ShouldStoreCompleteQuoteDataWithAllFields()
        {
            // Arrange
            var testQuoteData = new QuoteData
            {
                Symbol = "TEST",
                Price = 100.50,
                Change = 2.30,
                ChangePercent = 2.34,
                DayHigh = 102.00,
                DayLow = 98.50,
                MarketCap = 1000000000,
                Volume = 1500000,
                RSI = 65.5,
                PERatio = 15.2,
                Date = DateTime.Parse("2024-01-15 09:30:00"),
                LastUpdated = DateTime.Now,
                Timestamp = DateTime.Parse("2024-01-15 09:30:00")
            };

            // Act
            _cacheService.CacheQuoteData(testQuoteData);
            var retrieved = _cacheService.GetCachedStock("TEST");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(testQuoteData.Symbol, retrieved.Symbol);
            Assert.Equal(testQuoteData.Price, retrieved.Price, 2);
            Assert.Equal(testQuoteData.Change, retrieved.Change, 2);
            Assert.Equal(testQuoteData.ChangePercent, retrieved.ChangePercent, 2);
            Assert.Equal(testQuoteData.DayHigh, retrieved.DayHigh, 2);
            Assert.Equal(testQuoteData.DayLow, retrieved.DayLow, 2);
            Assert.Equal(testQuoteData.MarketCap, retrieved.MarketCap, 2);
            Assert.Equal(testQuoteData.Volume, retrieved.Volume, 2);
            Assert.Equal(testQuoteData.RSI, retrieved.RSI, 2);
            Assert.Equal(testQuoteData.PERatio, retrieved.PERatio, 2);
        }

        [Fact]
        public void GetAllCachedStocks_ShouldReturnAllStocksWithCompleteData()
        {
            // Arrange
            var testData1 = new QuoteData
            {
                Symbol = "AAPL",
                Price = 150.00,
                RSI = 70.0,
                PERatio = 25.5,
                Volume = 2000000,
                Date = DateTime.Now,
                LastUpdated = DateTime.Now,
                Timestamp = DateTime.Now
            };

            var testData2 = new QuoteData
            {
                Symbol = "MSFT",
                Price = 300.00,
                RSI = 60.0,
                PERatio = 30.2,
                Volume = 1800000,
                Date = DateTime.Now,
                LastUpdated = DateTime.Now,
                Timestamp = DateTime.Now
            };

            // Act
            _cacheService.CacheQuoteData(testData1);
            _cacheService.CacheQuoteData(testData2);
            var allStocks = _cacheService.GetAllCachedStocks();

            // Assert
            Assert.True(allStocks.Count >= 2);
            
            var aaplStock = allStocks.Find(s => s.Symbol == "AAPL");
            var msftStock = allStocks.Find(s => s.Symbol == "MSFT");
            
            Assert.NotNull(aaplStock);
            Assert.NotNull(msftStock);
            
            Assert.Equal(150.00, aaplStock.Price, 2);
            Assert.Equal(70.0, aaplStock.RSI, 2);
            Assert.Equal(300.00, msftStock.Price, 2);
            Assert.Equal(60.0, msftStock.RSI, 2);
        }

        [Fact]
        public void GetCachedStock_WithNonExistentSymbol_ShouldReturnNull()
        {
            // Act
            var result = _cacheService.GetCachedStock("NONEXISTENT");

            // Assert
            Assert.Null(result);
        }
    }
}