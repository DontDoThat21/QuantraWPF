using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Tests for MarketChatService sentiment correlation integration (MarketChat story 6)
    /// </summary>
    [TestClass]
    public class MarketChatSentimentCorrelationTests
    {
        #region SentimentPriceCorrelationAnalysis Integration Tests

        [TestMethod]
        public async Task SentimentPriceCorrelationAnalysis_AnalyzeSentimentPriceCorrelation_ReturnsValidResult()
        {
            // Arrange
            var analyzer = new SentimentPriceCorrelationAnalysis();
            string symbol = "AAPL";
            int days = 30;

            // Act
            var result = await analyzer.AnalyzeSentimentPriceCorrelation(symbol, days);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(symbol, result.Symbol);
            Assert.IsNotNull(result.SourceCorrelations);
            Assert.IsNotNull(result.SentimentShiftEvents);
            Assert.IsNotNull(result.AlignedData);
        }

        [TestMethod]
        public async Task SentimentPriceCorrelationAnalysis_AnalyzeSectorSentimentCorrelation_ReturnsValidResult()
        {
            // Arrange
            var analyzer = new SentimentPriceCorrelationAnalysis();
            string sector = "Technology";

            // Act
            var result = await analyzer.AnalyzeSectorSentimentCorrelation(sector);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsNotNull(result.SentimentShiftEvents);
        }

        [TestMethod]
        public async Task GetHistoricalSentimentContext_WithValidSymbol_ReturnsContextOrEmpty()
        {
            // Arrange
            var analyzer = new SentimentPriceCorrelationAnalysis();
            string symbol = "NVDA";
            int days = 30;

            // Act
            var context = await analyzer.GetHistoricalSentimentContext(symbol, days);

            // Assert - Context may be empty if no correlation data, but should not be null
            Assert.IsNotNull(context);
        }

        [TestMethod]
        public async Task GetHistoricalSentimentContext_WithNullSymbol_ReturnsEmpty()
        {
            // Arrange
            var analyzer = new SentimentPriceCorrelationAnalysis();

            // Act
            var context = await analyzer.GetHistoricalSentimentContext(null, 30);

            // Assert
            Assert.AreEqual(string.Empty, context);
        }

        [TestMethod]
        public async Task GetHistoricalSentimentContext_WithEmptySymbol_ReturnsEmpty()
        {
            // Arrange
            var analyzer = new SentimentPriceCorrelationAnalysis();

            // Act
            var context = await analyzer.GetHistoricalSentimentContext(string.Empty, 30);

            // Assert
            Assert.AreEqual(string.Empty, context);
        }

        #endregion

        #region SentimentShiftEvent Tests

        [TestMethod]
        public void SentimentShiftEvent_DefaultValues_AreCorrect()
        {
            // Arrange & Act
            var shiftEvent = new SentimentShiftEvent();

            // Assert
            Assert.AreEqual(default(DateTime), shiftEvent.Date);
            Assert.IsNull(shiftEvent.Source);
            Assert.AreEqual(0.0, shiftEvent.SentimentShift);
            Assert.AreEqual(0.0, shiftEvent.SubsequentPriceChange);
            Assert.IsFalse(shiftEvent.PriceFollowedSentiment);
        }

        [TestMethod]
        public void SentimentShiftEvent_CanSetProperties()
        {
            // Arrange
            var shiftEvent = new SentimentShiftEvent
            {
                Date = DateTime.Now,
                Source = "Twitter",
                SentimentShift = 0.5,
                SubsequentPriceChange = 2.5,
                PriceFollowedSentiment = true
            };

            // Assert
            Assert.IsNotNull(shiftEvent.Date);
            Assert.AreEqual("Twitter", shiftEvent.Source);
            Assert.AreEqual(0.5, shiftEvent.SentimentShift);
            Assert.AreEqual(2.5, shiftEvent.SubsequentPriceChange);
            Assert.IsTrue(shiftEvent.PriceFollowedSentiment);
        }

        #endregion

        #region SentimentPriceCorrelationResult Tests

        [TestMethod]
        public void SentimentPriceCorrelationResult_DefaultValues_AreCorrect()
        {
            // Arrange & Act
            var result = new SentimentPriceCorrelationResult();

            // Assert
            Assert.IsNull(result.Symbol);
            Assert.AreEqual(0.0, result.OverallCorrelation);
            Assert.IsNotNull(result.SourceCorrelations);
            Assert.AreEqual(0, result.SourceCorrelations.Count);
            Assert.AreEqual(0.0, result.LeadLagRelationship);
            Assert.AreEqual(0.0, result.PredictiveAccuracy);
            Assert.IsNotNull(result.SentimentShiftEvents);
            Assert.AreEqual(0, result.SentimentShiftEvents.Count);
            Assert.IsNotNull(result.AlignedData);
        }

        [TestMethod]
        public void SentimentPriceCorrelationResult_CanSetSymbol()
        {
            // Arrange
            var result = new SentimentPriceCorrelationResult
            {
                Symbol = "TSLA"
            };

            // Assert
            Assert.AreEqual("TSLA", result.Symbol);
        }

        [TestMethod]
        public void SentimentPriceCorrelationResult_CanSetCorrelationValues()
        {
            // Arrange
            var result = new SentimentPriceCorrelationResult
            {
                OverallCorrelation = 0.65,
                LeadLagRelationship = 1.5,
                PredictiveAccuracy = 0.68
            };

            // Assert
            Assert.AreEqual(0.65, result.OverallCorrelation);
            Assert.AreEqual(1.5, result.LeadLagRelationship);
            Assert.AreEqual(0.68, result.PredictiveAccuracy);
        }

        #endregion

        #region SentimentPriceAlignedData Tests

        [TestMethod]
        public void SentimentPriceAlignedData_DefaultValues_AreCorrect()
        {
            // Arrange & Act
            var data = new SentimentPriceAlignedData();

            // Assert
            Assert.IsNotNull(data.Dates);
            Assert.AreEqual(0, data.Dates.Count);
            Assert.IsNotNull(data.Prices);
            Assert.AreEqual(0, data.Prices.Count);
            Assert.IsNotNull(data.PriceChanges);
            Assert.AreEqual(0, data.PriceChanges.Count);
            Assert.IsNotNull(data.SentimentBySource);
            Assert.AreEqual(0, data.SentimentBySource.Count);
            Assert.IsNotNull(data.CombinedSentiment);
            Assert.AreEqual(0, data.CombinedSentiment.Count);
        }

        #endregion

        #region CorrelationResult Tests

        [TestMethod]
        public void CorrelationResult_DefaultValues_AreCorrect()
        {
            // Arrange & Act
            var result = new CorrelationResult();

            // Assert
            Assert.AreEqual(0.0, result.OverallCorrelation);
            Assert.AreEqual(0.0, result.LeadLagRelationship);
            Assert.IsNotNull(result.SentimentShiftEvents);
            Assert.AreEqual(0, result.SentimentShiftEvents.Count);
        }

        [TestMethod]
        public void CorrelationResult_CanSetProperties()
        {
            // Arrange
            var result = new CorrelationResult
            {
                OverallCorrelation = 0.75,
                LeadLagRelationship = 2.0
            };

            // Assert
            Assert.AreEqual(0.75, result.OverallCorrelation);
            Assert.AreEqual(2.0, result.LeadLagRelationship);
        }

        #endregion
    }
}
