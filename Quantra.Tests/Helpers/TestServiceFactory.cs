using System;
using System.Collections.Generic;
using Quantra.DAL.Services;
using Quantra.DAL.Data;
using Microsoft.EntityFrameworkCore;

namespace Quantra.Tests
{
    /// <summary>
    /// Factory class for creating stub/mock services for unit testing.
    /// These services provide minimal implementations that allow ViewModels
    /// to be instantiated without requiring a full DI container.
    /// </summary>
    public static class TestServiceFactory
    {
        /// <summary>
        /// Creates a stub QuantraDbContext for testing using in-memory database.
        /// </summary>
        public static QuantraDbContext CreateStubDbContext()
        {
            var options = new DbContextOptionsBuilder<QuantraDbContext>()
                .UseInMemoryDatabase(databaseName: $"TestDb_{Guid.NewGuid()}")
                .Options;
            return new QuantraDbContext(options);
        }

        /// <summary>
        /// Creates a stub StockDataCacheService for testing.
        /// </summary>
        public static StockDataCacheService CreateStubStockDataCacheService()
        {
            var loggingService = CreateStubLoggingService();
            var userSettingsService = CreateStubUserSettingsService();
            return new StockDataCacheService(userSettingsService, loggingService);
        }

        /// <summary>
        /// Creates a stub AlphaVantageService for testing.
        /// </summary>
        public static AlphaVantageService CreateStubAlphaVantageService()
        {
            var loggingService = CreateStubLoggingService();
            var userSettingsService = CreateStubUserSettingsService();
            return new AlphaVantageService(userSettingsService, loggingService);
        }

        /// <summary>
        /// Creates a stub RealTimeInferenceService for testing.
        /// </summary>
        public static RealTimeInferenceService CreateStubRealTimeInferenceService()
        {
            return new RealTimeInferenceService(10);
        }

        /// <summary>
        /// Creates a stub PredictionCacheService for testing.
        /// </summary>
        public static PredictionCacheService CreateStubPredictionCacheService()
        {
            var loggingService = CreateStubLoggingService();
            return new PredictionCacheService(loggingService);
        }

        /// <summary>
        /// Creates a stub LoggingService for testing.
        /// </summary>
        public static LoggingService CreateStubLoggingService()
        {
            return new LoggingService();
        }

        /// <summary>
        /// Creates a stub UserSettingsService for testing.
        /// </summary>
        public static UserSettingsService CreateStubUserSettingsService()
        {
            var dbContext = CreateStubDbContext();
            var loggingService = CreateStubLoggingService();
            return new UserSettingsService(dbContext, loggingService);
        }
    }
}
