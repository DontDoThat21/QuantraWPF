using System;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Models;
using Quantra.DAL.Services;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Tests for PredictionDataService cache integration (MarketChat story 3)
    /// </summary>
    [TestClass]
    public class PredictionDataServiceCacheTests
    {
        #region PredictionContextResult Tests

        [TestMethod]
        public void PredictionContextResult_Empty_ReturnsCorrectDefaults()
        {
            // Act
            var result = PredictionContextResult.Empty;

            // Assert
            Assert.IsNull(result.Context);
            Assert.IsFalse(result.IsCached);
            Assert.IsNull(result.CacheAge);
            Assert.IsNull(result.PredictionTimestamp);
            Assert.IsNull(result.ModelVersion);
        }

        [TestMethod]
        public void PredictionContextResult_CacheStatusDisplay_ReturnsCorrectMessage_ForFreshPrediction()
        {
            // Arrange
            var result = new PredictionContextResult
            {
                IsCached = false,
                CacheAge = null
            };

            // Act
            var displayMessage = result.CacheStatusDisplay;

            // Assert
            Assert.AreEqual("Freshly generated prediction", displayMessage);
        }

        [TestMethod]
        public void PredictionContextResult_CacheStatusDisplay_ReturnsCorrectMessage_ForRecentCache()
        {
            // Arrange
            var result = new PredictionContextResult
            {
                IsCached = true,
                CacheAge = TimeSpan.FromSeconds(30) // Less than 1 minute
            };

            // Act
            var displayMessage = result.CacheStatusDisplay;

            // Assert
            Assert.AreEqual("Based on prediction from just now", displayMessage);
        }

        [TestMethod]
        public void PredictionContextResult_CacheStatusDisplay_ReturnsCorrectMessage_ForMinutesOldCache()
        {
            // Arrange
            var result = new PredictionContextResult
            {
                IsCached = true,
                CacheAge = TimeSpan.FromMinutes(23)
            };

            // Act
            var displayMessage = result.CacheStatusDisplay;

            // Assert
            Assert.AreEqual("Based on prediction from 23 minutes ago", displayMessage);
        }

        [TestMethod]
        public void PredictionContextResult_CacheStatusDisplay_ReturnsCorrectMessage_ForOneMinuteOldCache()
        {
            // Arrange
            var result = new PredictionContextResult
            {
                IsCached = true,
                CacheAge = TimeSpan.FromMinutes(1)
            };

            // Act
            var displayMessage = result.CacheStatusDisplay;

            // Assert
            Assert.AreEqual("Based on prediction from 1 minute ago", displayMessage);
        }

        [TestMethod]
        public void PredictionContextResult_CacheStatusDisplay_ReturnsCorrectMessage_ForHoursOldCache()
        {
            // Arrange
            var result = new PredictionContextResult
            {
                IsCached = true,
                CacheAge = TimeSpan.FromHours(2)
            };

            // Act
            var displayMessage = result.CacheStatusDisplay;

            // Assert
            Assert.AreEqual("Based on prediction from 2 hours ago", displayMessage);
        }

        [TestMethod]
        public void PredictionContextResult_CacheStatusDisplay_ReturnsCorrectMessage_ForOneHourOldCache()
        {
            // Arrange
            var result = new PredictionContextResult
            {
                IsCached = true,
                CacheAge = TimeSpan.FromHours(1)
            };

            // Act
            var displayMessage = result.CacheStatusDisplay;

            // Assert
            Assert.AreEqual("Based on prediction from 1 hour ago", displayMessage);
        }

        [TestMethod]
        public void PredictionContextResult_CacheStatusDisplay_ReturnsCorrectMessage_ForDaysOldCache()
        {
            // Arrange
            var result = new PredictionContextResult
            {
                IsCached = true,
                CacheAge = TimeSpan.FromDays(3)
            };

            // Act
            var displayMessage = result.CacheStatusDisplay;

            // Assert
            Assert.AreEqual("Based on prediction from 3 days ago", displayMessage);
        }

        [TestMethod]
        public void PredictionContextResult_CacheStatusDisplay_ReturnsCorrectMessage_ForOneDayOldCache()
        {
            // Arrange
            var result = new PredictionContextResult
            {
                IsCached = true,
                CacheAge = TimeSpan.FromDays(1)
            };

            // Act
            var displayMessage = result.CacheStatusDisplay;

            // Assert
            Assert.AreEqual("Based on prediction from 1 day ago", displayMessage);
        }

        #endregion

        #region PredictionDataService Tests

        [TestMethod]
        public void PredictionDataService_GetPopularSymbols_ReturnsNonEmptyList()
        {
            // Act
            var symbols = PredictionDataService.GetPopularSymbols();

            // Assert
            Assert.IsNotNull(symbols);
            var symbolList = new System.Collections.Generic.List<string>(symbols);
            Assert.IsTrue(symbolList.Count > 0);
        }

        [TestMethod]
        public void PredictionDataService_GetPopularSymbols_ContainsCommonSymbols()
        {
            // Act
            var symbols = PredictionDataService.GetPopularSymbols();
            var symbolList = new System.Collections.Generic.List<string>(symbols);

            // Assert - Should contain common popular symbols
            Assert.IsTrue(symbolList.Contains("AAPL"));
            Assert.IsTrue(symbolList.Contains("MSFT"));
            Assert.IsTrue(symbolList.Contains("GOOGL"));
            Assert.IsTrue(symbolList.Contains("AMZN"));
            Assert.IsTrue(symbolList.Contains("TSLA"));
        }

        #endregion
    }
}
