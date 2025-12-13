using System;
using NUnit.Framework;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Moq;

namespace Quantra.Tests
{
    [TestFixture]
    public class OrderServiceTests
    {
        private OrderService _orderService;
        private UserSettingsService _userSettingsService;
        private LoggingService _loggingService;
        private HistoricalDataService _historicalDataService;
        private AlphaVantageService _alphaVantageService;
        private TechnicalIndicatorService _technicalIndicatorService;

        [SetUp]
        public void Setup()
        {
            // Create an in-memory SQLite database for testing
            var connection = new Microsoft.Data.Sqlite.SqliteConnection("DataSource=:memory:");
            connection.Open();

            var options = new DbContextOptionsBuilder<QuantraDbContext>()
                .UseSqlite(connection)
                .Options;
            var dbContext = new QuantraDbContext(options);

            // Ensure database is created
            dbContext.Database.EnsureCreated();

            // Create instances of the required services
            _loggingService = new LoggingService();
            _userSettingsService = new UserSettingsService(dbContext, _loggingService);
            var stockSymbolCacheService = new StockSymbolCacheService(_loggingService);
            _historicalDataService = new HistoricalDataService(_userSettingsService, _loggingService, stockSymbolCacheService);
            _alphaVantageService = new AlphaVantageService(_userSettingsService, _loggingService, stockSymbolCacheService);
            _technicalIndicatorService = new TechnicalIndicatorService(_alphaVantageService, _userSettingsService, _loggingService, stockSymbolCacheService);

            // Create the OrderService with all required dependencies
            _orderService = new OrderService(
                _userSettingsService,
                _historicalDataService,
                _alphaVantageService,
                _technicalIndicatorService
            );
        }

        [Test]
        public void SetupDollarCostAveraging_ValidParameters_ReturnsTrue()
        {
            // Arrange
            string symbol = "AAPL";
            int totalShares = 100;
            int numberOfOrders = 4;
            int intervalDays = 7;

            // Act
            bool result = _orderService.SetupDollarCostAveraging(symbol, totalShares, numberOfOrders, intervalDays);

            // Assert
            Assert.That(result, Is.True, "SetupDollarCostAveraging should return true for valid parameters");
        }

        [Test]
        public void SetupDollarCostAveraging_InvalidSymbol_ReturnsFalse()
        {
            // Arrange
            string symbol = "";
            int totalShares = 100;
            int numberOfOrders = 4;
            int intervalDays = 7;

            // Act
            bool result = _orderService.SetupDollarCostAveraging(symbol, totalShares, numberOfOrders, intervalDays);

            // Assert
            Assert.That(result, Is.False, "SetupDollarCostAveraging should return false for invalid symbol");
        }

        [Test]
        public void SetupDollarCostAveraging_InvalidShares_ReturnsFalse()
        {
            // Arrange
            string symbol = "AAPL";
            int totalShares = 0;
            int numberOfOrders = 4;
            int intervalDays = 7;

            // Act
            bool result = _orderService.SetupDollarCostAveraging(symbol, totalShares, numberOfOrders, intervalDays);

            // Assert
            Assert.That(result, Is.False, "SetupDollarCostAveraging should return false for zero shares");
        }
    }
}