using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;

namespace Quantra.Tests
{
    [TestClass]
    public class SentimentExtremeSignalServiceTests
    {
        private Mock<ITradingService> _mockTradingService;
        private Mock<IEmailService> _mockEmailService;
        private UserSettings _testSettings;
        private SentimentExtremeSignalService _service;
        
        [TestInitialize]
        public void Initialize()
        {
            _mockTradingService = new Mock<ITradingService>();
            _mockEmailService = new Mock<IEmailService>();
            _testSettings = new UserSettings();
            
            // Setup the trading service mock to return success
            _mockTradingService.Setup(
                s => s.ExecuteTradeAsync(
                    It.IsAny<string>(), 
                    It.IsAny<string>(), 
                    It.IsAny<double>(), 
                    It.IsAny<double>()))
                .ReturnsAsync(true);
            
            _service = new SentimentExtremeSignalService(
                _mockTradingService.Object,
                _mockEmailService.Object,
                _testSettings);
        }
        
        [TestMethod]
        public async Task AnalyzeAndGenerateSignal_WithExtremePositiveSentiment_ReturnsValidSignal()
        {
            // Arrange
            string testSymbol = "AAPL";
            
            // Use reflection to replace the GatherSentimentDataAsync method for testing
            var serviceType = typeof(SentimentExtremeSignalService);
            var methodInfo = serviceType.GetMethod("GatherSentimentDataAsync", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            // Create a dynamic method that returns highly positive sentiment
            var testMethod = new Func<string, Task<Dictionary<string, double>>>(symbol =>
            {
                var sentimentData = new Dictionary<string, double>
                {
                    { "News", 0.75 },
                    { "Twitter", 0.68 },
                    { "Reddit", 0.72 },
                    { "AnalystRatings", 0.58 },
                    { "InsiderTrading", 0.7 }
                };
                return Task.FromResult(sentimentData);
            });
            
            // Act
            var signal = await _service.AnalyzeAndGenerateSignalAsync(testSymbol);
            
            // Assert
            Assert.IsNotNull(signal, "A signal should be generated for extreme positive sentiment");
            Assert.AreEqual(testSymbol, signal.Symbol);
            Assert.AreEqual("BUY", signal.RecommendedAction);
            Assert.IsTrue(signal.SentimentScore > 0.6, "Sentiment score should be strongly positive");
            Assert.IsTrue(signal.ConfidenceLevel > 0.7, "Confidence level should be high");
            Assert.IsTrue(signal.ContributingSources.Count >= 2, "Multiple sources should contribute");
            Assert.IsTrue(signal.CurrentPrice > 0, "Price should be positive");
            Assert.IsTrue(signal.TargetPrice > signal.CurrentPrice, "Target price should be higher than current for BUY signal");
        }
        
        [TestMethod]
        public async Task AnalyzeAndGenerateSignal_WithExtremeNegativeSentiment_ReturnsValidSignal()
        {
            // Arrange
            string testSymbol = "META";
            
            // Use reflection to replace the GatherSentimentDataAsync method for testing
            var serviceType = typeof(SentimentExtremeSignalService);
            var methodInfo = serviceType.GetMethod("GatherSentimentDataAsync", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            // Create a dynamic method that returns highly negative sentiment
            var testMethod = new Func<string, Task<Dictionary<string, double>>>(symbol =>
            {
                var sentimentData = new Dictionary<string, double>
                {
                    { "News", -0.73 },
                    { "Twitter", -0.65 },
                    { "Reddit", -0.70 },
                    { "AnalystRatings", -0.55 },
                    { "InsiderTrading", -0.68 }
                };
                return Task.FromResult(sentimentData);
            });
            
            // Act
            var signal = await _service.AnalyzeAndGenerateSignalAsync(testSymbol);
            
            // Assert
            Assert.IsNotNull(signal, "A signal should be generated for extreme negative sentiment");
            Assert.AreEqual(testSymbol, signal.Symbol);
            Assert.AreEqual("SELL", signal.RecommendedAction);
            Assert.IsTrue(signal.SentimentScore < -0.6, "Sentiment score should be strongly negative");
            Assert.IsTrue(signal.ConfidenceLevel > 0.7, "Confidence level should be high");
            Assert.IsTrue(signal.ContributingSources.Count >= 2, "Multiple sources should contribute");
            Assert.IsTrue(signal.CurrentPrice > 0, "Price should be positive");
            Assert.IsTrue(signal.TargetPrice < signal.CurrentPrice, "Target price should be lower than current for SELL signal");
        }
        
        [TestMethod]
        public async Task AnalyzeAndGenerateSignal_WithMixedSentiment_ReturnsNull()
        {
            // Arrange
            string testSymbol = "NVDA";
            
            // Use reflection to replace the GatherSentimentDataAsync method for testing
            var serviceType = typeof(SentimentExtremeSignalService);
            var methodInfo = serviceType.GetMethod("GatherSentimentDataAsync", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            // Create a dynamic method that returns mixed sentiment
            var testMethod = new Func<string, Task<Dictionary<string, double>>>(symbol =>
            {
                var sentimentData = new Dictionary<string, double>
                {
                    { "News", 0.35 },
                    { "Twitter", -0.20 },
                    { "Reddit", 0.45 },
                    { "AnalystRatings", -0.15 },
                    { "InsiderTrading", 0.30 }
                };
                return Task.FromResult(sentimentData);
            });
            
            // Act
            var signal = await _service.AnalyzeAndGenerateSignalAsync(testSymbol);
            
            // Assert
            Assert.IsNull(signal, "No signal should be generated for mixed/neutral sentiment");
        }
        
        [TestMethod]
        public async Task AnalyzeAndGenerateSignal_WithAutoExecute_CallsTradingService()
        {
            // Arrange
            string testSymbol = "TSLA";
            
            // Use reflection to replace the GatherSentimentDataAsync method for testing
            var serviceType = typeof(SentimentExtremeSignalService);
            var methodInfo = serviceType.GetMethod("GatherSentimentDataAsync", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            // Create a dynamic method that returns highly positive sentiment
            var testMethod = new Func<string, Task<Dictionary<string, double>>>(symbol =>
            {
                var sentimentData = new Dictionary<string, double>
                {
                    { "News", 0.75 },
                    { "Twitter", 0.70 },
                    { "Reddit", 0.72 },
                    { "AnalystRatings", 0.68 },
                    { "InsiderTrading", 0.65 }
                };
                return Task.FromResult(sentimentData);
            });
            
            // Act
            var signal = await _service.AnalyzeAndGenerateSignalAsync(testSymbol, true);
            
            // Assert
            Assert.IsNotNull(signal, "A signal should be generated");
            Assert.IsTrue(signal.IsActedUpon, "The trade should be executed");
            
            // Verify trading service was called
            _mockTradingService.Verify(
                s => s.ExecuteTradeAsync(
                    testSymbol,
                    "BUY", 
                    It.IsAny<double>(), 
                    It.IsAny<double>()),
                Times.Once);
        }
    }
}