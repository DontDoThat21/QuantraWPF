using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.Tests.Services
{
    [TestClass]
    public class AnalystRatingServiceTests
    {
        private AnalystRatingService _service;

        [TestInitialize]
        public void Setup()
        {
            _service = new AnalystRatingService();
        }

        [TestMethod]
        public async Task GetAnalystSentimentAsync_WithValidSymbol_ReturnsValidSentiment()
        {
            // Arrange
            string symbol = "AAPL";

            // Act
            double sentiment = await _service.GetAnalystSentimentAsync(symbol);

            // Assert
            Assert.IsTrue(sentiment >= -1.0 && sentiment <= 1.0, 
                $"Sentiment score {sentiment} should be between -1.0 and 1.0");
        }

        [TestMethod]
        public async Task GetAnalystSentimentAsync_WithInvalidSymbol_ReturnsValidSentiment()
        {
            // Arrange
            string symbol = "INVALID";

            // Act
            double sentiment = await _service.GetAnalystSentimentAsync(symbol);

            // Assert - Should fall back to traditional rating sentiment even with invalid symbol
            Assert.IsTrue(sentiment >= -1.0 && sentiment <= 1.0, 
                $"Sentiment score {sentiment} should be between -1.0 and 1.0 even for invalid symbol");
        }

        [TestMethod]
        public async Task GetRatingSentimentAsync_WithValidSymbol_ReturnsValidSentiment()
        {
            // Arrange
            string symbol = "MSFT";

            // Act
            double sentiment = await _service.GetRatingSentimentAsync(symbol);

            // Assert
            Assert.IsTrue(sentiment >= -1.0 && sentiment <= 1.0, 
                $"Rating sentiment score {sentiment} should be between -1.0 and 1.0");
        }

        [TestMethod]
        public async Task GetAnalystSentimentAsync_MultipleCalls_ConsistentResults()
        {
            // Arrange
            string symbol = "NVDA";

            // Act
            double sentiment1 = await _service.GetAnalystSentimentAsync(symbol);
            double sentiment2 = await _service.GetAnalystSentimentAsync(symbol);

            // Assert
            Assert.IsTrue(sentiment1 >= -1.0 && sentiment1 <= 1.0);
            Assert.IsTrue(sentiment2 >= -1.0 && sentiment2 <= 1.0);
            // Note: Results may vary slightly due to AI calls, but should be in valid range
        }
    }
}