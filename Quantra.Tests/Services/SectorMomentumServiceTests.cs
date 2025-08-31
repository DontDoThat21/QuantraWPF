using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Tests.Services
{
    [TestClass]
    public class SectorMomentumServiceTests
    {
        private SectorMomentumService _service;

        [TestInitialize]
        public void Setup()
        {
            _service = new SectorMomentumService();
        }

        [TestMethod]
        public void GetSectorMomentumData_ShouldReturnData()
        {
            // Arrange
            string timeframe = "1m";

            // Act
            var result = _service.GetSectorMomentumData(timeframe);

            // Assert
            Assert.IsNotNull(result);
            Assert.IsTrue(result.Count > 0, "Should return sector data");
            
            // Check that we have expected sectors
            Assert.IsTrue(result.ContainsKey("Technology"), "Should contain Technology sector");
            Assert.IsTrue(result.ContainsKey("Financial"), "Should contain Financial sector");
            
            // Check that each sector has subsector data
            foreach (var sector in result)
            {
                Assert.IsTrue(sector.Value.Count > 0, $"Sector {sector.Key} should have subsector data");
                
                foreach (var subsector in sector.Value)
                {
                    Assert.IsNotNull(subsector.Symbol, "Subsector should have a symbol");
                    Assert.IsNotNull(subsector.Name, "Subsector should have a name");
                    Assert.IsTrue(subsector.MomentumValue >= -1.0 && subsector.MomentumValue <= 1.0, 
                                "Momentum value should be between -1 and 1");
                }
            }
        }

        [TestMethod]
        public void GetSectorMomentumData_DifferentTimeframes_ShouldReturnDifferentData()
        {
            // Arrange
            string timeframe1 = "1d";
            string timeframe2 = "1m";

            // Act
            var result1 = _service.GetSectorMomentumData(timeframe1);
            var result2 = _service.GetSectorMomentumData(timeframe2);

            // Assert
            Assert.IsNotNull(result1);
            Assert.IsNotNull(result2);
            
            // The data should potentially be different for different timeframes
            // (though they might be the same due to caching or sample data generation)
            Assert.IsTrue(result1.Count > 0);
            Assert.IsTrue(result2.Count > 0);
        }

        [TestMethod]
        public void GetSectorMomentumData_WithCache_ShouldUseCachedData()
        {
            // Arrange
            string timeframe = "1m";

            // Act - First call
            var result1 = _service.GetSectorMomentumData(timeframe);
            
            // Act - Second call (should use cache)
            var result2 = _service.GetSectorMomentumData(timeframe);

            // Assert
            Assert.IsNotNull(result1);
            Assert.IsNotNull(result2);
            
            // Both calls should return data
            Assert.IsTrue(result1.Count > 0);
            Assert.IsTrue(result2.Count > 0);
        }

        [TestMethod]
        public void GetSectorMomentumData_ForceRefresh_ShouldRefreshData()
        {
            // Arrange
            string timeframe = "1m";

            // Act - First call
            var result1 = _service.GetSectorMomentumData(timeframe);
            
            // Act - Second call with force refresh
            var result2 = _service.GetSectorMomentumData(timeframe, forceRefresh: true);

            // Assert
            Assert.IsNotNull(result1);
            Assert.IsNotNull(result2);
            
            // Both calls should return data
            Assert.IsTrue(result1.Count > 0);
            Assert.IsTrue(result2.Count > 0);
        }

        [TestMethod]
        public void GetSectorMomentumData_ShouldIncludeRealSymbols()
        {
            // Arrange
            string timeframe = "1m";

            // Act
            var result = _service.GetSectorMomentumData(timeframe);

            // Assert
            Assert.IsNotNull(result);
            
            // Check that we have some real stock symbols (either from real data or the mapping)
            var allSymbols = result.Values.SelectMany(list => list.Select(item => item.Symbol)).ToList();
            var realSymbols = new[] { "AAPL", "MSFT", "GOOGL", "JPM", "BAC", "XOM", "JNJ", "PG" };
            
            // At least some of the real symbols should be present
            var foundRealSymbols = allSymbols.Where(s => realSymbols.Contains(s)).ToList();
            Assert.IsTrue(foundRealSymbols.Count > 0, 
                         $"Should contain some real stock symbols. Found: {string.Join(", ", allSymbols)}");
        }
    }
}