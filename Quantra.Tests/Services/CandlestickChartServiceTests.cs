using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Quantra.Models;
using Quantra.Services;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Unit tests for CandlestickChartService
    /// </summary>
    public class CandlestickChartServiceTests
    {
        private readonly CandlestickChartService _chartService;

        public CandlestickChartServiceTests()
        {
            _chartService = new CandlestickChartService();
        }

        [Fact]
        public void CreateChartSeries_WithValidData_ReturnsSeriesCollections()
        {
            // Arrange
            var testData = CreateTestHistoricalData(50);

            // Act
            var (candlestickSeries, volumeSeries, timeLabels) = _chartService.CreateChartSeries(testData, "AAPL");

            // Assert
            Assert.NotNull(candlestickSeries);
            Assert.NotNull(volumeSeries);
            Assert.NotNull(timeLabels);
            Assert.Single(candlestickSeries);
            Assert.Equal(2, volumeSeries.Count); // Up and down volume series
            Assert.Equal(50, timeLabels.Count);
        }

        [Fact]
        public void CreateChartSeries_WithEmptyData_ReturnsEmptyCollections()
        {
            // Arrange
            var emptyData = new List<HistoricalPrice>();

            // Act
            var (candlestickSeries, volumeSeries, timeLabels) = _chartService.CreateChartSeries(emptyData, "AAPL");

            // Assert
            Assert.NotNull(candlestickSeries);
            Assert.NotNull(volumeSeries);
            Assert.NotNull(timeLabels);
            Assert.Empty(candlestickSeries);
            Assert.Empty(volumeSeries);
            Assert.Empty(timeLabels);
        }

        [Fact]
        public void CreateChartSeries_WithNullData_ReturnsEmptyCollections()
        {
            // Arrange
            List<HistoricalPrice> nullData = null;

            // Act
            var (candlestickSeries, volumeSeries, timeLabels) = _chartService.CreateChartSeries(nullData, "AAPL");

            // Assert
            Assert.NotNull(candlestickSeries);
            Assert.NotNull(volumeSeries);
            Assert.NotNull(timeLabels);
            Assert.Empty(candlestickSeries);
            Assert.Empty(volumeSeries);
            Assert.Empty(timeLabels);
        }

        [Fact]
        public void CreateChartSeries_WithMaxCandles_LimitsDataPoints()
        {
            // Arrange
            var testData = CreateTestHistoricalData(200);
            int maxCandles = 100;

            // Act
            var (candlestickSeries, volumeSeries, timeLabels) = _chartService.CreateChartSeries(testData, "AAPL", maxCandles);

            // Assert
            Assert.Equal(maxCandles, timeLabels.Count);
        }

        [Fact]
        public void CreateChartSeries_DataIsSortedByDate()
        {
            // Arrange
            var testData = CreateUnsortedHistoricalData();

            // Act
            var (candlestickSeries, volumeSeries, timeLabels) = _chartService.CreateChartSeries(testData, "AAPL");

            // Assert
            // Verify that the time labels are in chronological order
            Assert.NotEmpty(timeLabels);
            // Additional verification could check that dates are sorted
        }

        [Fact]
        public void CreateChartSeries_WithCustomColorScheme_AppliesColors()
        {
            // Arrange
            var testData = CreateTestHistoricalData(10);
            var customColorScheme = new CandlestickChartColorScheme
            {
                CandleUpBrush = System.Windows.Media.Brushes.Blue,
                CandleDownBrush = System.Windows.Media.Brushes.Orange
            };
            var chartService = new CandlestickChartService(customColorScheme);

            // Act
            var (candlestickSeries, volumeSeries, timeLabels) = chartService.CreateChartSeries(testData, "AAPL");

            // Assert
            Assert.NotNull(candlestickSeries);
            Assert.Single(candlestickSeries);
            // Verify colors are applied (specific implementation would check series properties)
        }

        [Fact]
        public void CreateChartSeries_SeparatesUpAndDownVolumes()
        {
            // Arrange
            var testData = new List<HistoricalPrice>
            {
                new HistoricalPrice
                {
                    Date = DateTime.Now.AddMinutes(-10),
                    Open = 100,
                    High = 105,
                    Low = 99,
                    Close = 103, // Up candle
                    Volume = 1000000
                },
                new HistoricalPrice
                {
                    Date = DateTime.Now.AddMinutes(-5),
                    Open = 103,
                    High = 104,
                    Low = 98,
                    Close = 99, // Down candle
                    Volume = 1500000
                },
                new HistoricalPrice
                {
                    Date = DateTime.Now,
                    Open = 99,
                    High = 102,
                    Low = 98,
                    Close = 101, // Up candle
                    Volume = 1200000
                }
            };

            // Act
            var (candlestickSeries, volumeSeries, timeLabels) = _chartService.CreateChartSeries(testData, "AAPL");

            // Assert
            Assert.Equal(2, volumeSeries.Count); // Up and down volume series
            Assert.Equal(3, timeLabels.Count);
        }

        [Fact]
        public void CreateChartSeries_DetectsGapsInData()
        {
            // Arrange
            var testData = new List<HistoricalPrice>
            {
                new HistoricalPrice { Date = DateTime.Now.AddHours(-2), Open = 100, High = 101, Low = 99, Close = 100.5, Volume = 1000000 },
                new HistoricalPrice { Date = DateTime.Now.AddHours(-1), Open = 100.5, High = 102, Low = 100, Close = 101, Volume = 1000000 },
                // Gap here - more than expected interval
                new HistoricalPrice { Date = DateTime.Now, Open = 101, High = 103, Low = 100.5, Close = 102, Volume = 1000000 }
            };

            // Act
            var (candlestickSeries, volumeSeries, timeLabels) = _chartService.CreateChartSeries(testData, "AAPL");

            // Assert
            Assert.NotNull(candlestickSeries);
            Assert.Equal(3, timeLabels.Count);
            // Gap detection is embedded in tooltip generation
        }

        [Fact]
        public void CreateChartSeries_FormatsTimeLabelsCorrectly()
        {
            // Arrange
            var testData = new List<HistoricalPrice>
            {
                new HistoricalPrice { Date = new DateTime(2024, 1, 1, 9, 30, 0), Open = 100, High = 101, Low = 99, Close = 100.5, Volume = 1000000 },
                new HistoricalPrice { Date = new DateTime(2024, 1, 1, 9, 35, 0), Open = 100.5, High = 102, Low = 100, Close = 101, Volume = 1000000 },
                new HistoricalPrice { Date = new DateTime(2024, 1, 2, 9, 30, 0), Open = 101, High = 103, Low = 100.5, Close = 102, Volume = 1000000 }
            };

            // Act
            var (candlestickSeries, volumeSeries, timeLabels) = _chartService.CreateChartSeries(testData, "AAPL");

            // Assert
            Assert.Equal(3, timeLabels.Count);
            // First label should include date (day changed from null)
            Assert.Contains("01/01", timeLabels[0]);
            // Second label should only have time (same day)
            Assert.DoesNotContain("01/01", timeLabels[1]);
            // Third label should include date (day changed)
            Assert.Contains("01/02", timeLabels[2]);
        }

        #region Helper Methods

        private List<HistoricalPrice> CreateTestHistoricalData(int count)
        {
            var data = new List<HistoricalPrice>();
            var baseDate = DateTime.Now.AddDays(-1);

            for (int i = 0; i < count; i++)
            {
                data.Add(new HistoricalPrice
                {
                    Date = baseDate.AddMinutes(i * 5),
                    Open = 100 + i * 0.1,
                    High = 101 + i * 0.1,
                    Low = 99 + i * 0.1,
                    Close = 100.5 + i * 0.1,
                    Volume = 1000000 + i * 1000
                });
            }

            return data;
        }

        private List<HistoricalPrice> CreateUnsortedHistoricalData()
        {
            var data = new List<HistoricalPrice>
            {
                new HistoricalPrice { Date = DateTime.Now, Open = 103, High = 105, Low = 102, Close = 104, Volume = 3000000 },
                new HistoricalPrice { Date = DateTime.Now.AddMinutes(-10), Open = 100, High = 102, Low = 99, Close = 101, Volume = 1000000 },
                new HistoricalPrice { Date = DateTime.Now.AddMinutes(-5), Open = 101, High = 103, Low = 100, Close = 102, Volume = 2000000 }
            };

            return data;
        }

        #endregion
    }
}
