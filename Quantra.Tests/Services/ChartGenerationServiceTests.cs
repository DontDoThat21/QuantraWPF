using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Models;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Tests for ChartGenerationService (MarketChat story 8)
    /// </summary>
    [TestClass]
    public class ChartGenerationServiceTests
    {
        private ChartGenerationService _chartService;

        [TestInitialize]
        public void Setup()
        {
            // Initialize service without dependencies for basic tests
            _chartService = new ChartGenerationService();
        }

        #region IsChartRequest Tests

        [TestMethod]
        public void IsChartRequest_ReturnsTrue_ForChartKeyword()
        {
            // Arrange
            var messages = new[]
            {
                "Show me a chart for AAPL",
                "Can you display a price chart?",
                "Plot the historical data for MSFT",
                "Visualize the prediction for GOOGL",
                "Graph the price trend for TSLA"
            };

            // Act & Assert
            foreach (var message in messages)
            {
                Assert.IsTrue(_chartService.IsChartRequest(message), $"Should recognize '{message}' as chart request");
            }
        }

        [TestMethod]
        public void IsChartRequest_ReturnsTrue_ForAlternativePatterns()
        {
            // Arrange
            var messages = new[]
            {
                "I need a price chart",
                "Show me the prediction chart",
                "Can I see a forecast graph?",
                "Display the projection plot"
            };

            // Act & Assert
            foreach (var message in messages)
            {
                Assert.IsTrue(_chartService.IsChartRequest(message), $"Should recognize '{message}' as chart request");
            }
        }

        [TestMethod]
        public void IsChartRequest_ReturnsFalse_ForNonChartMessages()
        {
            // Arrange
            var messages = new[]
            {
                "What is the price of AAPL?",
                "Tell me about Microsoft",
                "How is the market doing?",
                "Buy signal for TSLA"
            };

            // Act & Assert
            foreach (var message in messages)
            {
                Assert.IsFalse(_chartService.IsChartRequest(message), $"Should not recognize '{message}' as chart request");
            }
        }

        [TestMethod]
        public void IsChartRequest_ReturnsFalse_ForNullOrEmpty()
        {
            // Act & Assert
            Assert.IsFalse(_chartService.IsChartRequest(null));
            Assert.IsFalse(_chartService.IsChartRequest(""));
            Assert.IsFalse(_chartService.IsChartRequest("   "));
        }

        #endregion

        #region ExtractChartParameters Tests

        [TestMethod]
        public void ExtractChartParameters_ExtractsSymbol_FromUpperCase()
        {
            // Arrange
            var message = "Show me a chart for AAPL";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.IsTrue(parameters.Symbols.Contains("AAPL"));
        }

        [TestMethod]
        public void ExtractChartParameters_ExtractsSymbol_FromLowerCase()
        {
            // Arrange
            var message = "show me a chart for aapl";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.IsTrue(parameters.Symbols.Contains("AAPL"));
        }

        [TestMethod]
        public void ExtractChartParameters_ExtractsSymbol_FromMixedCase()
        {
            // Arrange
            var message = "Show me a chart for Aapl";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.IsTrue(parameters.Symbols.Contains("AAPL"));
        }

        [TestMethod]
        public void ExtractChartParameters_ExtractsMultipleSymbols()
        {
            // Arrange
            var message = "Compare charts for AAPL and MSFT";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.IsTrue(parameters.Symbols.Contains("AAPL"));
            Assert.IsTrue(parameters.Symbols.Contains("MSFT"));
        }

        [TestMethod]
        public void ExtractChartParameters_ExtractsForecastDays()
        {
            // Arrange
            var message = "Show me a 30 day forecast for AAPL";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.AreEqual(30, parameters.ForecastDays);
        }

        [TestMethod]
        public void ExtractChartParameters_ExtractsWeeklyForecast()
        {
            // Arrange
            var message = "Show me a 2 week forecast for AAPL";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.AreEqual(14, parameters.ForecastDays);
        }

        [TestMethod]
        public void ExtractChartParameters_ExtractsMonthlyForecast()
        {
            // Arrange
            var message = "Show me a 2 month forecast for AAPL";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.AreEqual(60, parameters.ForecastDays);
        }

        [TestMethod]
        public void ExtractChartParameters_CapsForecastAt90Days()
        {
            // Arrange
            var message = "Show me a 6 month forecast for AAPL";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.AreEqual(90, parameters.ForecastDays);
        }

        [TestMethod]
        public void ExtractChartParameters_ExcludesCommonWords()
        {
            // Arrange
            var message = "I want a chart for AAPL";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert - Should not contain common words like "I", "A"
            Assert.IsFalse(parameters.Symbols.Contains("I"));
            Assert.IsFalse(parameters.Symbols.Contains("A"));
            Assert.IsTrue(parameters.Symbols.Contains("AAPL"));
        }

        [TestMethod]
        public void ExtractChartParameters_DefaultIncludesBollingerBands()
        {
            // Arrange
            var message = "Show me a chart for AAPL";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.IsTrue(parameters.IncludeBollingerBands);
        }

        [TestMethod]
        public void ExtractChartParameters_ExcludesBollingerBands_WhenRequested()
        {
            // Arrange
            var message = "Show me a chart for AAPL without bollinger bands";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.IsFalse(parameters.IncludeBollingerBands);
        }

        [TestMethod]
        public void ExtractChartParameters_DefaultIncludesSupportResistance()
        {
            // Arrange
            var message = "Show me a chart for AAPL";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.IsTrue(parameters.IncludeSupportResistance);
        }

        [TestMethod]
        public void ExtractChartParameters_ReturnsEmptySymbols_ForNullMessage()
        {
            // Act
            var parameters = _chartService.ExtractChartParameters(null);

            // Assert
            Assert.IsNotNull(parameters);
            Assert.AreEqual(0, parameters.Symbols.Count);
        }

        [TestMethod]
        public void ExtractChartParameters_DefaultForecastDaysIs30()
        {
            // Arrange
            var message = "Show me a chart for AAPL";

            // Act
            var parameters = _chartService.ExtractChartParameters(message);

            // Assert
            Assert.AreEqual(30, parameters.ForecastDays);
        }

        #endregion

        #region GenerateChartFromData Tests

        [TestMethod]
        public void GenerateChartFromData_ReturnsValidChart_WithHistoricalData()
        {
            // Arrange
            var historicalData = new List<HistoricalPrice>
            {
                new HistoricalPrice { Date = DateTime.Now.AddDays(-5), Close = 100.0 },
                new HistoricalPrice { Date = DateTime.Now.AddDays(-4), Close = 101.0 },
                new HistoricalPrice { Date = DateTime.Now.AddDays(-3), Close = 102.0 },
                new HistoricalPrice { Date = DateTime.Now.AddDays(-2), Close = 103.0 },
                new HistoricalPrice { Date = DateTime.Now.AddDays(-1), Close = 104.0 }
            };
            var prediction = new PredictionResult
            {
                Action = "BUY",
                Confidence = 0.85,
                TargetPrice = 110.0
            };

            // Act
            var chartData = _chartService.GenerateChartFromData(historicalData, prediction, "AAPL");

            // Assert
            Assert.IsNotNull(chartData);
            Assert.IsTrue(chartData.IsValid);
            Assert.AreEqual("AAPL", chartData.Symbol);
            Assert.AreEqual(5, chartData.HistoricalPrices.Count);
            Assert.AreEqual("BUY", chartData.PredictedAction);
        }

        [TestMethod]
        public void GenerateChartFromData_SetsErrorMessage_WhenNoHistoricalData()
        {
            // Arrange
            var historicalData = new List<HistoricalPrice>();
            var prediction = new PredictionResult
            {
                Action = "BUY",
                Confidence = 0.85
            };

            // Act
            var chartData = _chartService.GenerateChartFromData(historicalData, prediction, "AAPL");

            // Assert
            Assert.IsFalse(chartData.IsValid);
            Assert.IsNotNull(chartData.ErrorMessage);
        }

        [TestMethod]
        public void GenerateChartFromData_SetsCurrentPrice_FromLastHistoricalPrice()
        {
            // Arrange
            var historicalData = new List<HistoricalPrice>
            {
                new HistoricalPrice { Date = DateTime.Now.AddDays(-2), Close = 100.0 },
                new HistoricalPrice { Date = DateTime.Now.AddDays(-1), Close = 105.0 }
            };

            // Act
            var chartData = _chartService.GenerateChartFromData(historicalData, null, "AAPL");

            // Assert
            Assert.AreEqual(105.0, chartData.CurrentPrice);
        }

        [TestMethod]
        public void GenerateChartFromData_GeneratesCombinedSeries()
        {
            // Arrange
            var historicalData = new List<HistoricalPrice>
            {
                new HistoricalPrice { Date = DateTime.Now.AddDays(-2), Close = 100.0 },
                new HistoricalPrice { Date = DateTime.Now.AddDays(-1), Close = 105.0 }
            };
            var prediction = new PredictionResult
            {
                Action = "BUY",
                TargetPrice = 110.0,
                TimeSeries = new TimeSeriesPrediction
                {
                    PricePredictions = new List<double> { 106.0, 107.0, 108.0 },
                    TimePoints = new List<DateTime> { DateTime.Now.AddDays(1), DateTime.Now.AddDays(2), DateTime.Now.AddDays(3) }
                }
            };

            // Act
            var chartData = _chartService.GenerateChartFromData(historicalData, prediction, "AAPL");

            // Assert
            Assert.AreEqual(2, chartData.PredictionStartIndex);
            Assert.AreEqual(5, chartData.CombinedPrices.Count); // 2 historical + 3 prediction
        }

        #endregion

        #region ProjectionChartData Tests

        [TestMethod]
        public void ProjectionChartData_IsValid_ReturnsFalse_WhenHistoricalPricesNull()
        {
            // Arrange
            var chartData = new ProjectionChartData
            {
                HistoricalPrices = null
            };

            // Act & Assert
            Assert.IsFalse(chartData.IsValid);
        }

        [TestMethod]
        public void ProjectionChartData_IsValid_ReturnsFalse_WhenHistoricalPricesEmpty()
        {
            // Arrange
            var chartData = new ProjectionChartData
            {
                HistoricalPrices = new List<double>()
            };

            // Act & Assert
            Assert.IsFalse(chartData.IsValid);
        }

        [TestMethod]
        public void ProjectionChartData_IsValid_ReturnsTrue_WhenHistoricalPricesExist()
        {
            // Arrange
            var chartData = new ProjectionChartData
            {
                HistoricalPrices = new List<double> { 100.0, 101.0 }
            };

            // Act & Assert
            Assert.IsTrue(chartData.IsValid);
        }

        [TestMethod]
        public void ProjectionChartData_GeneratedAt_DefaultsToNow()
        {
            // Arrange & Act
            var chartData = new ProjectionChartData();

            // Assert
            Assert.IsTrue((DateTime.Now - chartData.GeneratedAt).TotalSeconds < 5);
        }

        #endregion

        #region ChartRequestParameters Tests

        [TestMethod]
        public void ChartRequestParameters_DefaultValues()
        {
            // Arrange & Act
            var parameters = new ChartRequestParameters();

            // Assert
            Assert.IsNotNull(parameters.Symbols);
            Assert.AreEqual(0, parameters.Symbols.Count);
            Assert.IsNull(parameters.StartDate);
            Assert.AreEqual(30, parameters.ForecastDays);
            Assert.IsTrue(parameters.IncludeBollingerBands);
            Assert.IsTrue(parameters.IncludeSupportResistance);
            Assert.AreEqual(60, parameters.HistoricalDays);
        }

        #endregion
    }
}
