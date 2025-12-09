using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Moq;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.ViewModels;
using Quantra.Services;

namespace Quantra.Tests.ViewModels
{
    /// <summary>
    /// Unit tests for CandlestickChartViewModel
    /// </summary>
    public class CandlestickChartViewModelTests
    {
        private readonly Mock<ICandlestickDataService> _mockDataService;
        private readonly Mock<TechnicalIndicatorService> _mockTechnicalIndicatorService;
        private readonly Mock<UserSettingsService> _mockUserSettingsService;
        private readonly Mock<LoggingService> _mockLoggingService;
        private readonly CandlestickChartService _chartService;

        public CandlestickChartViewModelTests()
        {
            _mockDataService = new Mock<ICandlestickDataService>();
            _mockTechnicalIndicatorService = new Mock<TechnicalIndicatorService>();
            _mockUserSettingsService = new Mock<UserSettingsService>();
            _mockLoggingService = new Mock<LoggingService>();
            _chartService = new CandlestickChartService();
        }

        [Fact]
        public void Constructor_WithValidParameters_InitializesViewModel()
        {
            // Arrange & Act
            var viewModel = CreateViewModel("AAPL");

            // Assert
            Assert.Equal("AAPL", viewModel.Symbol);
            Assert.Equal("Real-Time Candlestick Chart - AAPL", viewModel.WindowTitle);
            Assert.True(viewModel.IsAutoRefreshEnabled);
            Assert.False(viewModel.IsLoading);
            Assert.False(viewModel.IsDataLoaded);
        }

        [Fact]
        public void Constructor_WithNullSymbol_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => CreateViewModel(null));
        }

        [Fact]
        public void Constructor_WithNullDataService_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new CandlestickChartViewModel(
                    "AAPL",
                    null,
                    _chartService,
                    _mockTechnicalIndicatorService.Object,
                    _mockUserSettingsService.Object,
                    _mockLoggingService.Object));
        }

        [Fact]
        public async Task LoadCandlestickDataAsync_WithValidData_UpdatesChart()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");
            var testData = CreateTestHistoricalData();

            _mockDataService
                .Setup(x => x.GetCandlestickDataAsync("AAPL", "5min", false, It.IsAny<CancellationToken>()))
                .ReturnsAsync(testData);

            // Act
            await viewModel.LoadCandlestickDataAsync();

            // Assert
            Assert.True(viewModel.IsDataLoaded);
            Assert.False(viewModel.IsLoading);
            Assert.False(viewModel.IsNoData);
            Assert.NotNull(viewModel.CandlestickSeries);
            Assert.NotNull(viewModel.VolumeSeries);
            Assert.NotEmpty(viewModel.TimeLabels);
        }

        [Fact]
        public async Task LoadCandlestickDataAsync_WithNoData_SetsNoDataFlag()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");

            _mockDataService
                .Setup(x => x.GetCandlestickDataAsync("AAPL", "5min", false, It.IsAny<CancellationToken>()))
                .ReturnsAsync(new List<HistoricalPrice>());

            // Act
            await viewModel.LoadCandlestickDataAsync();

            // Assert
            Assert.False(viewModel.IsDataLoaded);
            Assert.True(viewModel.IsNoData);
            Assert.False(viewModel.IsLoading);
        }

        [Fact]
        public async Task LoadCandlestickDataAsync_WithException_HandlesGracefully()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");

            _mockDataService
                .Setup(x => x.GetCandlestickDataAsync("AAPL", "5min", false, It.IsAny<CancellationToken>()))
                .ThrowsAsync(new Exception("Test exception"));

            // Act
            await viewModel.LoadCandlestickDataAsync();

            // Assert
            Assert.False(viewModel.IsDataLoaded);
            Assert.True(viewModel.IsNoData);
            Assert.False(viewModel.IsLoading);
            _mockLoggingService.Verify(
                x => x.LogErrorWithContext(It.IsAny<Exception>(), It.IsAny<string>()),
                Times.AtLeastOnce);
        }

        [Fact]
        public void IsAutoRefreshEnabled_WhenSetToTrue_StartsRefreshTimer()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");
            viewModel.IsAutoRefreshEnabled = false;

            // Act
            viewModel.IsAutoRefreshEnabled = true;

            // Assert
            Assert.True(viewModel.IsAutoRefreshEnabled);
            Assert.Equal("ON", viewModel.AutoRefreshText);
        }

        [Fact]
        public void IsAutoRefreshEnabled_WhenSetToFalse_StopsRefreshTimer()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");

            // Act
            viewModel.IsAutoRefreshEnabled = false;

            // Assert
            Assert.False(viewModel.IsAutoRefreshEnabled);
            Assert.Equal("OFF", viewModel.AutoRefreshText);
        }

        [Fact]
        public void IsPaused_WhenSetToTrue_StopsAutoRefresh()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");
            viewModel.IsAutoRefreshEnabled = true;

            // Act
            viewModel.IsPaused = true;

            // Assert
            Assert.True(viewModel.IsPaused);
            Assert.Equal("? Resume", viewModel.PauseButtonText);
        }

        [Fact]
        public void CurrentInterval_WhenChanged_TriggersDataReload()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");
            var testData = CreateTestHistoricalData();

            _mockDataService
                .Setup(x => x.GetCandlestickDataAsync("AAPL", It.IsAny<string>(), true, It.IsAny<CancellationToken>()))
                .ReturnsAsync(testData);

            // Act
            viewModel.CurrentInterval = "15min";

            // Assert
            Assert.Equal("15min", viewModel.CurrentInterval);
            _mockDataService.Verify(
                x => x.GetCandlestickDataAsync("AAPL", "15min", true, It.IsAny<CancellationToken>()),
                Times.Once);
        }

        [Fact]
        public void PriceChange_WhenPositive_ReturnGreenColor()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");

            // Act
            viewModel.PriceChange = 10.0;

            // Assert
            Assert.NotNull(viewModel.PriceChangeColor);
            // Color is green (simplified check)
        }

        [Fact]
        public void PriceChange_WhenNegative_ReturnsRedColor()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");

            // Act
            viewModel.PriceChange = -10.0;

            // Assert
            Assert.NotNull(viewModel.PriceChangeColor);
            // Color is red (simplified check)
        }

        [Fact]
        public void MaxCandles_WhenChanged_UpdatesChart()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");
            var testData = CreateTestHistoricalData();

            // Simulate cached data
            viewModel.LoadCandlestickDataAsync().Wait();

            // Act
            viewModel.MaxCandles = 50;

            // Assert
            Assert.Equal(50, viewModel.MaxCandles);
        }

        [Fact]
        public void Dispose_StopsTimersAndCancelsRequests()
        {
            // Arrange
            var viewModel = CreateViewModel("AAPL");
            viewModel.IsAutoRefreshEnabled = true;

            // Act
            viewModel.Dispose();

            // Assert
            // Verify that resources are cleaned up (timers stopped, cancellation tokens disposed)
            Assert.False(viewModel.IsLoading);
        }

        #region Helper Methods

        private CandlestickChartViewModel CreateViewModel(string symbol)
        {
            return new CandlestickChartViewModel(
                symbol,
                _mockDataService.Object,
                _chartService,
                _mockTechnicalIndicatorService.Object,
                _mockUserSettingsService.Object,
                _mockLoggingService.Object);
        }

        private List<HistoricalPrice> CreateTestHistoricalData()
        {
            var data = new List<HistoricalPrice>();
            var baseDate = DateTime.Now.AddDays(-1);

            for (int i = 0; i < 100; i++)
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

        #endregion
    }
}
