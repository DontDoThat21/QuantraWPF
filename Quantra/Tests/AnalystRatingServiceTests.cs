using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using Quantra.DAL.Services;

namespace Quantra.Tests
{
    [TestClass]
    public class AnalystRatingServiceTests
    {
        private Mock<IAnalystRatingService> _mockAnalystRatingService;
        private AnalystConsensusReportService _consensusReportService;
        
        [TestInitialize]
        public void Setup()
        {
            _mockAnalystRatingService = new Mock<IAnalystRatingService>();
            _consensusReportService = new AnalystConsensusReportService(_mockAnalystRatingService.Object);
        }
        
        [TestMethod]
        public async Task AnalyzeConsensusHistory_WithHistoricalData_ReturnsCorrectTrend()
        {
            // Arrange
            string testSymbol = "AAPL";
            var oldConsensus = new AnalystRatingAggregate
            {
                Symbol = testSymbol,
                ConsensusRating = "Hold",
                ConsensusScore = 0.1,
                BuyCount = 5,
                HoldCount = 10,
                SellCount = 3,
                LastUpdated = DateTime.Now.AddDays(-30)
            };
            
            var newConsensus = new AnalystRatingAggregate
            {
                Symbol = testSymbol,
                ConsensusRating = "Buy",
                ConsensusScore = 0.4,
                BuyCount = 8,
                HoldCount = 8,
                SellCount = 2,
                LastUpdated = DateTime.Now
            };
            
            var historyData = new List<AnalystRatingAggregate> { oldConsensus, newConsensus };
            
            _mockAnalystRatingService.Setup(s => s.GetAggregatedRatingsAsync(testSymbol))
                .ReturnsAsync(newConsensus);
                
            _mockAnalystRatingService.Setup(s => s.GetConsensusHistoryAsync(testSymbol, It.IsAny<int>()))
                .ReturnsAsync(historyData);
                
            _mockAnalystRatingService.Setup(s => s.AnalyzeConsensusHistoryAsync(testSymbol, It.IsAny<int>()))
                .ReturnsAsync(newConsensus);
                
            // Act
            var result = await _consensusReportService.GenerateConsensusReportAsync(testSymbol);
            
            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(testSymbol, result.Symbol);
            Assert.IsNotNull(result.ConsensusChangeStats);
            Assert.AreEqual(3, result.ConsensusChangeStats.BuyCountChange);
            Assert.AreEqual(-2, result.ConsensusChangeStats.HoldCountChange);
            Assert.AreEqual(-1, result.ConsensusChangeStats.SellCountChange);
            Assert.AreEqual(0.3, result.ConsensusChangeStats.ScoreChange, 0.01);
            Assert.AreEqual("Hold", result.ConsensusChangeStats.StartConsensusRating);
            Assert.AreEqual("Buy", result.ConsensusChangeStats.EndConsensusRating);
        }
        
        [TestMethod]
        public async Task GenerateConsensusReport_WithNoData_ReturnsBasicReport()
        {
            // Arrange
            string testSymbol = "MISSING";
            var emptyConsensus = new AnalystRatingAggregate
            {
                Symbol = testSymbol,
                ConsensusRating = "No Ratings",
                ConsensusScore = 0,
                BuyCount = 0,
                HoldCount = 0,
                SellCount = 0,
                LastUpdated = DateTime.Now
            };
            
            _mockAnalystRatingService.Setup(s => s.GetAggregatedRatingsAsync(testSymbol))
                .ReturnsAsync(emptyConsensus);
                
            _mockAnalystRatingService.Setup(s => s.GetConsensusHistoryAsync(testSymbol, It.IsAny<int>()))
                .ReturnsAsync(new List<AnalystRatingAggregate>());
                
            _mockAnalystRatingService.Setup(s => s.GetRecentRatingsAsync(testSymbol, It.IsAny<int>()))
                .ReturnsAsync(new List<AnalystRating>());
                
            // Act
            var result = await _consensusReportService.GenerateConsensusReportAsync(testSymbol);
            
            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(testSymbol, result.Symbol);
            Assert.IsNull(result.ConsensusChangeStats);
            Assert.IsNull(result.RatingDistributionTrend);
            Assert.AreEqual(0, result.SignificantChanges.Count);
            Assert.AreEqual("No Ratings", result.CurrentConsensus.ConsensusRating);
        }
    }
}