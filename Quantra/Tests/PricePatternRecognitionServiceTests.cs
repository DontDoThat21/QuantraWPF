using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Moq;
using NUnit.Framework;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;

namespace Quantra.Tests
{
    [TestFixture]
    public class PricePatternRecognitionServiceTests
    {
        private Mock<StockDataCacheService> _mockStockDataService;
        private PricePatternRecognitionService _patternService;

        [SetUp]
        public void Setup()
        {
            _mockStockDataService = new Mock<StockDataCacheService>();
            _patternService = new PricePatternRecognitionService(_mockStockDataService.Object);
        }

        [Test]
        public async Task DetectDoubleTop_ReturnsPattern_WhenValidDataProvided()
        {
            // Arrange
            var historicalData = CreateDoubleTopPattern();
            string symbol = "AAPL";

            // Act
            var patterns = await _patternService.DetectDoubleTopsAsync(symbol, historicalData);

            // Assert
            Assert.That(patterns, Is.Not.Empty, "Should detect at least one double top pattern");
            Assert.That(patterns[0].Type, Is.EqualTo(PricePatternRecognitionService.PatternType.DoubleTop), "Should identify pattern as double top");
            Assert.That(patterns[0].Bias, Is.EqualTo(PricePatternRecognitionService.PatternBias.Bearish), "Double top should have bearish bias");
        }

        [Test]
        public async Task DetectDoubleBottom_ReturnsPattern_WhenValidDataProvided()
        {
            // Arrange
            var historicalData = CreateDoubleBottomPattern();
            string symbol = "AAPL";

            // Act
            var patterns = await _patternService.DetectDoubleBottomsAsync(symbol, historicalData);

            // Assert
            Assert.That(patterns, Is.Not.Empty, "Should detect at least one double bottom pattern");
            Assert.That(patterns[0].Type, Is.EqualTo(PricePatternRecognitionService.PatternType.DoubleBottom), "Should identify pattern as double bottom");
            Assert.That(patterns[0].Bias, Is.EqualTo(PricePatternRecognitionService.PatternBias.Bullish), "Double bottom should have bullish bias");
        }

        // Helper method to create a simulated double top pattern
        private List<HistoricalPrice> CreateDoubleTopPattern()
        {
            var prices = new List<HistoricalPrice>();
            DateTime baseDate = DateTime.Today.AddDays(-100);
            
            // Create a rising trend
            for (int i = 0; i < 30; i++)
            {
                double price = 100 + (i * 1.5); // Rising price
                prices.Add(new HistoricalPrice
                {
                    Date = baseDate.AddDays(i),
                    Open = price - 0.5,
                    Close = price + 0.5,
                    High = price + 1.0,
                    Low = price - 1.0,
                    Volume = 1000000
                });
            }
            
            // Create first top
            prices.Add(new HistoricalPrice
            {
                Date = baseDate.AddDays(30),
                Open = 145,
                Close = 148,
                High = 150,
                Low = 144,
                Volume = 2000000
            });
            
            // Create pullback
            for (int i = 1; i <= 10; i++)
            {
                double price = 148 - (i * 0.8); // Declining price
                prices.Add(new HistoricalPrice
                {
                    Date = baseDate.AddDays(30 + i),
                    Open = price + 0.5,
                    Close = price - 0.5,
                    High = price + 1.0,
                    Low = price - 1.0,
                    Volume = 1000000 + i * 50000
                });
            }
            
            // Create second top
            for (int i = 1; i <= 10; i++)
            {
                double price = 140 + (i * 0.9); // Rising price
                prices.Add(new HistoricalPrice
                {
                    Date = baseDate.AddDays(40 + i),
                    Open = price - 0.5,
                    Close = price + 0.5,
                    High = price + 1.0,
                    Low = price - 1.0,
                    Volume = 1200000
                });
            }
            
            prices.Add(new HistoricalPrice
            {
                Date = baseDate.AddDays(51),
                Open = 149,
                Close = 148,
                High = 149.5,
                Low = 146,
                Volume = 1800000
            });
            
            // Create decline after second top
            for (int i = 1; i <= 10; i++)
            {
                double price = 148 - (i * 1.2); // Declining price
                prices.Add(new HistoricalPrice
                {
                    Date = baseDate.AddDays(51 + i),
                    Open = price + 0.5,
                    Close = price - 0.5,
                    High = price + 1.0,
                    Low = price - 1.0,
                    Volume = 1500000
                });
            }
            
            return prices;
        }
        
        // Helper method to create a simulated double bottom pattern
        private List<HistoricalPrice> CreateDoubleBottomPattern()
        {
            var prices = new List<HistoricalPrice>();
            DateTime baseDate = DateTime.Today.AddDays(-100);
            
            // Create a declining trend
            for (int i = 0; i < 30; i++)
            {
                double price = 150 - (i * 1.5); // Falling price
                prices.Add(new HistoricalPrice
                {
                    Date = baseDate.AddDays(i),
                    Open = price + 0.5,
                    Close = price - 0.5,
                    High = price + 1.0,
                    Low = price - 1.0,
                    Volume = 1000000
                });
            }
            
            // Create first bottom
            prices.Add(new HistoricalPrice
            {
                Date = baseDate.AddDays(30),
                Open = 105,
                Close = 102,
                High = 106,
                Low = 100,
                Volume = 2000000
            });
            
            // Create bounce
            for (int i = 1; i <= 10; i++)
            {
                double price = 102 + (i * 0.8); // Rising price
                prices.Add(new HistoricalPrice
                {
                    Date = baseDate.AddDays(30 + i),
                    Open = price - 0.5,
                    Close = price + 0.5,
                    High = price + 1.0,
                    Low = price - 1.0,
                    Volume = 1000000 + i * 50000
                });
            }
            
            // Create second bottom
            for (int i = 1; i <= 10; i++)
            {
                double price = 110 - (i * 0.9); // Falling price
                prices.Add(new HistoricalPrice
                {
                    Date = baseDate.AddDays(40 + i),
                    Open = price + 0.5,
                    Close = price - 0.5,
                    High = price + 1.0,
                    Low = price - 1.0,
                    Volume = 1200000
                });
            }
            
            prices.Add(new HistoricalPrice
            {
                Date = baseDate.AddDays(51),
                Open = 101,
                Close = 102,
                High = 104,
                Low = 100.5,
                Volume = 1800000
            });
            
            // Create rise after second bottom
            for (int i = 1; i <= 10; i++)
            {
                double price = 102 + (i * 1.2); // Rising price
                prices.Add(new HistoricalPrice
                {
                    Date = baseDate.AddDays(51 + i),
                    Open = price - 0.5,
                    Close = price + 0.5,
                    High = price + 1.0,
                    Low = price - 1.0,
                    Volume = 1500000
                });
            }
            
            return prices;
        }
    }
}