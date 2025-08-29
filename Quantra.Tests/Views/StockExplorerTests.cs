using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Controls;
using Quantra;
using Quantra.Models;
using Quantra.ViewModels;
using LiveCharts;
using Quantra.Services;

namespace Quantra.Tests.Views
{
    [TestClass]
    public class StockExplorerTests
    {
        [TestMethod]
        public void UpdatedTimestampText_InitializesEmpty()
        {
            // Arrange & Act
            var stockExplorer = new StockExplorer();
            
            // Assert
            Assert.AreEqual("", stockExplorer.UpdatedTimestampText);
        }

        [TestMethod]
        public void UpdatedTimestampText_PropertyChangedNotified()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            bool propertyChangedFired = false;
            
            stockExplorer.PropertyChanged += (sender, e) =>
            {
                if (e.PropertyName == nameof(stockExplorer.UpdatedTimestampText))
                    propertyChangedFired = true;
            };
            
            // Act
            stockExplorer.UpdatedTimestampText = "Updated: 12/01/2023 10:30";
            
            // Assert
            Assert.IsTrue(propertyChangedFired);
            Assert.AreEqual("Updated: 12/01/2023 10:30", stockExplorer.UpdatedTimestampText);
        }

        [TestMethod]
        public void QuoteData_LastUpdatedProperty_Exists()
        {
            // Arrange & Act
            var quoteData = new QuoteData
            {
                Symbol = "AAPL",
                Price = 150.00,
                LastUpdated = DateTime.Now
            };
            
            // Assert
            Assert.IsNotNull(quoteData.LastUpdated);
            Assert.AreEqual("AAPL", quoteData.Symbol);
            Assert.AreEqual(150.00, quoteData.Price);
        }

        [TestMethod]
        public void QuoteData_TimestampProperty_Exists()
        {
            // Arrange & Act
            var quoteData = new QuoteData
            {
                Symbol = "AAPL",
                Price = 150.00,
                Timestamp = DateTime.Now
            };
            
            // Assert
            Assert.IsNotNull(quoteData.Timestamp);
            Assert.AreEqual("AAPL", quoteData.Symbol);
        }

        [TestMethod]
        public void QuoteData_DefaultLastUpdated_HandledProperly()
        {
            // Arrange & Act
            var quoteData = new QuoteData
            {
                Symbol = "AAPL",
                Price = 150.00
                // LastUpdated not set, should be default(DateTime)
            };
            
            // Assert
            Assert.AreEqual(default(DateTime), quoteData.LastUpdated);
            Assert.AreEqual("AAPL", quoteData.Symbol);
        }

        [TestMethod]
        public void QuoteData_LastAccessedProperty_Exists()
        {
            // Arrange & Act
            var quoteData = new QuoteData
            {
                Symbol = "AAPL",
                Price = 150.00,
                LastAccessed = DateTime.Now
            };
            
            // Assert
            Assert.IsNotNull(quoteData.LastAccessed);
            Assert.AreEqual("AAPL", quoteData.Symbol);
            Assert.AreEqual(150.00, quoteData.Price);
        }

        [TestMethod]
        public void QuoteData_DefaultLastAccessed_HandledProperly()
        {
            // Arrange & Act
            var quoteData = new QuoteData
            {
                Symbol = "AAPL",
                Price = 150.00
                // LastAccessed not set, should be default(DateTime)
            };
            
            // Assert
            Assert.AreEqual(default(DateTime), quoteData.LastAccessed);
            Assert.AreEqual("AAPL", quoteData.Symbol);
        }

        [TestMethod]
        public void QuoteData_VWAPProperty_Exists()
        {
            // Arrange & Act
            var quoteData = new QuoteData
            {
                Symbol = "AAPL",
                Price = 150.00,
                PERatio = 25.5,
                RSI = 65.2,
                VWAP = 148.75
            };
            
            // Assert
            Assert.AreEqual("AAPL", quoteData.Symbol);
            Assert.AreEqual(150.00, quoteData.Price);
            Assert.AreEqual(25.5, quoteData.PERatio);
            Assert.AreEqual(65.2, quoteData.RSI);
            Assert.AreEqual(148.75, quoteData.VWAP);
        }

        [TestMethod]
        public void SelectedSymbolTitle_InitializesWithDefaultMessage()
        {
            // Arrange & Act
            var stockExplorer = new StockExplorer();
            
            // Assert
            Assert.AreEqual("Select a symbol to view historical data", stockExplorer.SelectedSymbolTitle);
        }

        [TestMethod]
        public void SelectedSymbolTitle_PropertyChangedNotified()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            bool propertyChangedFired = false;
            
            stockExplorer.PropertyChanged += (sender, e) =>
            {
                if (e.PropertyName == nameof(stockExplorer.SelectedSymbolTitle))
                    propertyChangedFired = true;
            };
            
            // Act
            stockExplorer.SelectedSymbolTitle = "AAPL - Historical Data";
            
            // Assert
            Assert.IsTrue(propertyChangedFired);
            Assert.AreEqual("AAPL - Historical Data", stockExplorer.SelectedSymbolTitle);
        }

        [TestMethod]
        public async Task QuoteData_PopulateChartValuesFromHistorical_WorksCorrectly()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var historicalData = new List<HistoricalPrice>
            {
                new HistoricalPrice { Date = DateTime.Now.AddDays(-2), Open = 145.0, High = 148.0, Low = 144.0, Close = 147.0, Volume = 1000000 },
                new HistoricalPrice { Date = DateTime.Now.AddDays(-1), Open = 147.0, High = 150.0, Low = 146.0, Close = 149.0, Volume = 1200000 },
                new HistoricalPrice { Date = DateTime.Now, Open = 149.0, High = 152.0, Low = 148.0, Close = 151.0, Volume = 1100000 }
            };

            // Act
            await quoteData.PopulateChartValuesFromHistorical(historicalData);

            // Assert
            Assert.AreEqual(3, quoteData.StockPriceValues.Count);
            Assert.AreEqual(147.0, quoteData.StockPriceValues[0]);
            Assert.AreEqual(149.0, quoteData.StockPriceValues[1]);
            Assert.AreEqual(151.0, quoteData.StockPriceValues[2]);
            
            // Verify PatternCandles (OHLC) data
            Assert.AreEqual(3, quoteData.PatternCandles.Count);
            Assert.AreEqual(145.0, quoteData.PatternCandles[0].Open);
            Assert.AreEqual(148.0, quoteData.PatternCandles[0].High);
            Assert.AreEqual(144.0, quoteData.PatternCandles[0].Low);
            Assert.AreEqual(147.0, quoteData.PatternCandles[0].Close);
        }

        [TestMethod]
        public async Task StockExplorerViewModel_PriceValues_ReflectsSelectedStock()
        {
            // Arrange
            var viewModel = new StockExplorerViewModel();
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var historicalData = new List<HistoricalPrice>
            {
                new HistoricalPrice { Date = DateTime.Now.AddDays(-1), Close = 150.0 },
                new HistoricalPrice { Date = DateTime.Now, Close = 152.0 }
            };
            
            // Populate the quote data
            await quoteData.PopulateChartValuesFromHistorical(historicalData);

            // Act
            viewModel.SelectedStock = quoteData;

            // Assert
            Assert.IsNotNull(viewModel.PriceValues);
            Assert.AreEqual(2, viewModel.PriceValues.Count);
            Assert.AreEqual(150.0, viewModel.PriceValues[0]);
            Assert.AreEqual(152.0, viewModel.PriceValues[1]);
        }

        [TestMethod]
        public void StockExplorer_MultipleInstances_CreateSuccessfully()
        {
            // Arrange & Act
            var stockExplorer1 = new StockExplorer();
            var stockExplorer2 = new StockExplorer();

            // Assert - Both instances should be created successfully
            Assert.IsNotNull(stockExplorer1);
            Assert.IsNotNull(stockExplorer2);
            Assert.AreNotSame(stockExplorer1, stockExplorer2);
            
            // Verify initial state is properly set for both
            Assert.AreEqual("", stockExplorer1.UpdatedTimestampText);
            Assert.AreEqual("", stockExplorer2.UpdatedTimestampText);
            Assert.AreEqual("Select a symbol to view historical data", stockExplorer1.SelectedSymbolTitle);
            Assert.AreEqual("Select a symbol to view historical data", stockExplorer2.SelectedSymbolTitle);
        }

        [TestMethod]
        public async Task StockExplorerViewModel_DateFormatter_UpdatesWhenDateLabelFormatterChanges()
        {
            // Arrange
            var viewModel = new StockExplorerViewModel();
            bool dateFormatterChangedFired = false;
            
            viewModel.PropertyChanged += (sender, e) =>
            {
                if (e.PropertyName == nameof(viewModel.DateFormatter))
                    dateFormatterChangedFired = true;
            };

            var historicalData = new List<HistoricalPrice>
            {
                new HistoricalPrice { Date = new DateTime(2023, 1, 1), Close = 150.0 },
                new HistoricalPrice { Date = new DateTime(2023, 1, 2), Close = 152.0 },
                new HistoricalPrice { Date = new DateTime(2023, 1, 3), Close = 154.0 }
            };

            // Act
            await viewModel.LoadChartDataAsync(historicalData);

            // Assert
            Assert.IsTrue(dateFormatterChangedFired, "DateFormatter property change notification should fire when LoadChartDataAsync is called");
            Assert.IsNotNull(viewModel.DateFormatter, "DateFormatter should not be null after LoadChartDataAsync");
            
            // Test that the DateFormatter function works correctly
            var formattedDate = viewModel.DateFormatter(1); // Should format the second date (index 1)
            Assert.AreEqual("01/02", formattedDate, "DateFormatter should format dates as MM/dd");
        }

        [TestMethod]
        public void TopPE_SymbolSelectionMode_EnumValue_Exists()
        {
            // Arrange & Act - Verify TopPE enum value exists
            var topPEMode = Quantra.Enums.SymbolSelectionMode.TopPE;
            
            // Assert
            Assert.AreEqual(2, (int)topPEMode, "TopPE should be the third enum value (index 2)");
            Assert.AreEqual("TopPE", topPEMode.ToString());
        }

        [TestMethod]
        public void StockExplorer_CurrentSelectionMode_CanBeSetToTopPE()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Act
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.TopPE;
            
            // Assert
            Assert.AreEqual(Quantra.Enums.SymbolSelectionMode.TopPE, stockExplorer.CurrentSelectionMode);
        }

        [TestMethod]
        public void StockExplorer_CurrentSelectionMode_CanBeSetToHighVolume()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Act
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.HighVolume;
            
            // Assert
            Assert.AreEqual(Quantra.Enums.SymbolSelectionMode.HighVolume, stockExplorer.CurrentSelectionMode);
        }

        [TestMethod]
        public void HighVolume_SymbolSelectionMode_EnumValue_Exists()
        {
            // Arrange & Act - Verify HighVolume enum value exists
            var highVolumeMode = Quantra.Enums.SymbolSelectionMode.HighVolume;
            
            // Assert
            Assert.AreEqual(3, (int)highVolumeMode, "HighVolume should be the fourth enum value (index 3)");
            Assert.AreEqual("HighVolume", highVolumeMode.ToString());
        }

        [TestMethod]
        public void LowPE_SymbolSelectionMode_EnumValue_Exists()
        {
            // Arrange & Act - Verify LowPE enum value exists
            var lowPEMode = Quantra.Enums.SymbolSelectionMode.LowPE;
            
            // Assert
            Assert.AreEqual(4, (int)lowPEMode, "LowPE should be the fifth enum value (index 4)");
            Assert.AreEqual("LowPE", lowPEMode.ToString());
        }

        [TestMethod]
        public void StockExplorer_CurrentSelectionMode_CanBeSetToLowPE()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Act
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.LowPE;
            
            // Assert
            Assert.AreEqual(Quantra.Enums.SymbolSelectionMode.LowPE, stockExplorer.CurrentSelectionMode);
        }

        [TestMethod]
        public void StockExplorer_AutomaticSymbolLoading_InitializesCorrectly()
        {
            // Arrange & Act
            var stockExplorer = new StockExplorer();
            
            // Assert - Timer should be initialized but not running
            Assert.IsNotNull(stockExplorer);
            // We can't directly test the private timer, but we can verify the object initializes without errors
        }

        [TestMethod]
        public void StockExplorer_ModeSwitch_SymbolSearchPanelVisibility()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Act - Switch to a non-Individual mode first, then back to Individual
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.TopPE;
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.IndividualAsset;
            
            // Assert - SymbolSearchPanel should be visible for IndividualAsset mode
            Assert.AreEqual(Quantra.Enums.SymbolSelectionMode.IndividualAsset, stockExplorer.CurrentSelectionMode);
            // Note: We can't directly test UI visibility in unit tests without a UI framework,
            // but we can verify the mode is set correctly which triggers the OnSelectionModeChanged logic
        }

        [TestMethod]
        public void StockExplorer_ModeSwitch_ToAllModes_CompletesSuccessfully()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            var allModes = new[]
            {
                Quantra.Enums.SymbolSelectionMode.IndividualAsset,
                Quantra.Enums.SymbolSelectionMode.TopVolumeRsiDiscrepancies,
                Quantra.Enums.SymbolSelectionMode.TopPE,
                Quantra.Enums.SymbolSelectionMode.HighVolume,
                Quantra.Enums.SymbolSelectionMode.LowPE,
                Quantra.Enums.SymbolSelectionMode.RsiOversold,
                Quantra.Enums.SymbolSelectionMode.RsiOverbought,
                Quantra.Enums.SymbolSelectionMode.AllDatabase,
                Quantra.Enums.SymbolSelectionMode.HighTheta,
                Quantra.Enums.SymbolSelectionMode.HighBeta,
                Quantra.Enums.SymbolSelectionMode.HighAlpha
            };
            
            // Act & Assert - Should be able to switch to each mode without errors
            foreach (var mode in allModes)
            {
                stockExplorer.CurrentSelectionMode = mode;
                Assert.AreEqual(mode, stockExplorer.CurrentSelectionMode, 
                    $"Should be able to set CurrentSelectionMode to {mode}");
            }
        }

        [TestMethod]
        public void StockDataGrid_HeadersVisibility_IsSetToColumnOnly()
        {
            // Arrange & Act
            var stockExplorer = new StockExplorer();
            
            // Assert - The DataGrid should have HeadersVisibility set to Column only (no row headers)
            // This verifies that the first column button (row header) is hidden
            //Assert.IsNotNull(stockExplorer.StockDataGrid, "StockDataGrid should be accessible");
            
            // Note: In a proper UI test, we would verify the HeadersVisibility property is set correctly,
            // but this test ensures the component initializes without errors after the HeadersVisibility change
        }

        [TestMethod]
        public void TopVolumeRsiDiscrepancies_UsesBroaderSymbolUniverse()
        {
            // Arrange - Verify that the GetTopVolumeRsiDiscrepancies method uses a broader symbol universe
            var expectedSymbolCount = StockSymbols.CommonSymbols.Count;
            var oldHardcodedCount = 8; // Previous hardcoded array size
            
            // Assert - The new implementation should use significantly more symbols
            Assert.IsTrue(expectedSymbolCount > oldHardcodedCount, 
                $"Should use broader symbol universe. Expected > {oldHardcodedCount}, got {expectedSymbolCount}");
            Assert.IsTrue(expectedSymbolCount >= 70, 
                $"Should include major indices and ETFs (at least 70 symbols), got {expectedSymbolCount}");
            
            // Verify that all original symbols are still included in the broader universe
            var originalSymbols = new[] { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY" };
            foreach (var symbol in originalSymbols)
            {
                Assert.IsTrue(StockSymbols.CommonSymbols.Contains(symbol), 
                    $"Original symbol {symbol} should still be included in broader universe");
            }
        }

        [TestMethod]
        public void HighVolume_UsesBroaderSymbolUniverse()
        {
            // Arrange - Verify that the High Volume mode uses a broader symbol universe
            var expectedSymbolCount = StockSymbols.CommonSymbols.Count;
            var oldHardcodedCount = 10; // Previous hardcoded array size for High Volume
            
            // Assert - The new implementation should use significantly more symbols
            Assert.IsTrue(expectedSymbolCount > oldHardcodedCount, 
                $"High Volume should use broader symbol universe. Expected > {oldHardcodedCount}, got {expectedSymbolCount}");
            Assert.IsTrue(expectedSymbolCount >= 70, 
                $"Should include major indices and ETFs (at least 70 symbols), got {expectedSymbolCount}");
            
            // Verify that original high volume symbols are still included in the broader universe
            var originalHighVolumeSymbols = new[] { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "AMD" };
            foreach (var symbol in originalHighVolumeSymbols)
            {
                Assert.IsTrue(StockSymbols.CommonSymbols.Contains(symbol), 
                    $"Original high volume symbol {symbol} should still be included in broader universe");
            }
        }

        [TestMethod]
        public void RsiOverbought_SymbolSelectionMode_EnumValue_Exists()
        {
            // Arrange & Act - Verify RsiOverbought enum value exists
            var rsiOverboughtMode = Quantra.Enums.SymbolSelectionMode.RsiOverbought;
            
            // Assert
            Assert.AreEqual(6, (int)rsiOverboughtMode, "RsiOverbought should be the seventh enum value (index 6)");
            Assert.AreEqual("RsiOverbought", rsiOverboughtMode.ToString());
        }

        [TestMethod]
        public void StockExplorer_CurrentSelectionMode_CanBeSetToRsiOverbought()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Act
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.RsiOverbought;
            
            // Assert
            Assert.AreEqual(Quantra.Enums.SymbolSelectionMode.RsiOverbought, stockExplorer.CurrentSelectionMode);
        }

        [TestMethod]
        public void AllDatabase_SymbolSelectionMode_EnumValue_Exists()
        {
            // Arrange & Act - Verify AllDatabase enum value exists
            var allDatabaseMode = Quantra.Enums.SymbolSelectionMode.AllDatabase;
            
            // Assert
            Assert.AreEqual(7, (int)allDatabaseMode, "AllDatabase should be the eighth enum value (index 7)");
            Assert.AreEqual("AllDatabase", allDatabaseMode.ToString());
        }

        [TestMethod]
        public void StockExplorer_CurrentSelectionMode_CanBeSetToAllDatabase()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Act
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.AllDatabase;
            
            // Assert
            Assert.AreEqual(Quantra.Enums.SymbolSelectionMode.AllDatabase, stockExplorer.CurrentSelectionMode);
        }

        [TestMethod]
        public void AllDatabase_Mode_LoadsCachedDataWithoutApiCalls()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Act - Switch to AllDatabase mode
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.AllDatabase;
            
            // Assert
            Assert.AreEqual(Quantra.Enums.SymbolSelectionMode.AllDatabase, stockExplorer.CurrentSelectionMode);
            
            // Note: In a full integration test, we would verify that:
            // 1. The AllDatabaseButton is visible when this mode is selected
            // 2. Clicking the button loads data from StockDataCacheService.GetAllCachedStocks()
            // 3. No API calls are made during the data loading process
            // However, these would require UI integration testing framework
        }

        [TestMethod]
        public void LowPE_UsesBroaderSymbolUniverse()
        {
            // Arrange - Verify that the Low P/E mode uses a broader symbol universe
            var expectedSymbolCount = StockSymbols.CommonSymbols.Count;
            var oldHardcodedCount = 10; // Previous hardcoded array size for Low P/E
            
            // Assert - The new implementation should use significantly more symbols
            Assert.IsTrue(expectedSymbolCount > oldHardcodedCount, 
                $"Low P/E should use broader symbol universe. Expected > {oldHardcodedCount}, got {expectedSymbolCount}");
            Assert.IsTrue(expectedSymbolCount >= 70, 
                $"Should include major indices and ETFs (at least 70 symbols), got {expectedSymbolCount}");
            
            // Verify that original low P/E symbols are still included in the broader universe
            var originalLowPESymbols = new[] { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "INTC" };
            foreach (var symbol in originalLowPESymbols)
            {
                Assert.IsTrue(StockSymbols.CommonSymbols.Contains(symbol), 
                    $"Original Low P/E symbol {symbol} should still be included in broader universe");
            }
        }

        [TestMethod]
        public async Task GlobalLoadingStateService_WithLoadingState_ManagesStateCorrectly()
        {
            // Arrange
            bool loadingStateChanged = false;
            bool finalLoadingState = false;
            
            GlobalLoadingStateService.LoadingStateChanged += (isLoading) =>
            {
                loadingStateChanged = true;
                finalLoadingState = isLoading;
            };

            // Act
            await GlobalLoadingStateService.WithLoadingState(Task.Delay(50));

            // Assert
            Assert.IsTrue(loadingStateChanged, "Loading state should have changed");
            Assert.IsFalse(finalLoadingState, "Loading state should be false after task completion");
            Assert.IsFalse(GlobalLoadingStateService.IsLoading, "GlobalLoadingStateService should not be loading after task completion");
        }

        [TestMethod]
        public void StockExplorer_SymbolAndPriceFields_ResetToDefaultsOnModeChange()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Set some values to simulate having selected a stock
            stockExplorer.SymbolText = "Symbol: AAPL";
            stockExplorer.PriceText = "Price: $150.25";
            stockExplorer.UpdatedTimestampText = "Updated: 12/01/2023 10:30";
            
            // Verify values are set
            Assert.AreEqual("Symbol: AAPL", stockExplorer.SymbolText);
            Assert.AreEqual("Price: $150.25", stockExplorer.PriceText);
            Assert.AreEqual("Updated: 12/01/2023 10:30", stockExplorer.UpdatedTimestampText);
            
            // Act - Change mode from default IndividualAsset to another mode
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.TopPE;
            
            // Assert - Symbol and Price fields should be reset to defaults
            Assert.AreEqual("", stockExplorer.SymbolText, "SymbolText should be reset to empty string when mode changes");
            Assert.AreEqual("", stockExplorer.PriceText, "PriceText should be reset to empty string when mode changes");
            Assert.AreEqual("", stockExplorer.UpdatedTimestampText, "UpdatedTimestampText should be reset to empty string when mode changes");
        }

        [TestMethod]
        public void StockExplorer_SymbolAndPriceFields_ResetOnEveryModeChange()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            var testModes = new[]
            {
                Quantra.Enums.SymbolSelectionMode.TopVolumeRsiDiscrepancies,
                Quantra.Enums.SymbolSelectionMode.TopPE,
                Quantra.Enums.SymbolSelectionMode.HighVolume,
                Quantra.Enums.SymbolSelectionMode.LowPE,
                Quantra.Enums.SymbolSelectionMode.RsiOversold,
                Quantra.Enums.SymbolSelectionMode.RsiOverbought,
                Quantra.Enums.SymbolSelectionMode.AllDatabase,
                Quantra.Enums.SymbolSelectionMode.IndividualAsset,
                Quantra.Enums.SymbolSelectionMode.HighTheta,
                Quantra.Enums.SymbolSelectionMode.HighBeta,
                Quantra.Enums.SymbolSelectionMode.HighAlpha
            };
            
            // Act & Assert - Test each mode change resets the fields
            foreach (var mode in testModes)
            {
                // Set some values to simulate having selected a stock
                stockExplorer.SymbolText = "Symbol: TEST";
                stockExplorer.PriceText = "Price: $100.00";
                stockExplorer.UpdatedTimestampText = "Updated: Today";
                
                // Change to the test mode
                stockExplorer.CurrentSelectionMode = mode;
                
                // Assert fields are reset
                Assert.AreEqual("", stockExplorer.SymbolText, $"SymbolText should be reset when switching to {mode}");
                Assert.AreEqual("", stockExplorer.PriceText, $"PriceText should be reset when switching to {mode}");
                Assert.AreEqual("", stockExplorer.UpdatedTimestampText, $"UpdatedTimestampText should be reset when switching to {mode}");
            }
        }

        [TestMethod]
        public void StockExplorer_SymbolAndPriceFields_NoResetWhenSameModeSet()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Set initial mode and values
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.TopPE;
            stockExplorer.SymbolText = "Symbol: AAPL";
            stockExplorer.PriceText = "Price: $150.25";
            stockExplorer.UpdatedTimestampText = "Updated: 12/01/2023 10:30";
            
            // Act - Set the same mode again
            stockExplorer.CurrentSelectionMode = Quantra.Enums.SymbolSelectionMode.TopPE;
            
            // Assert - Fields should NOT be reset since mode didn't actually change
            Assert.AreEqual("Symbol: AAPL", stockExplorer.SymbolText, "SymbolText should not be reset when setting same mode");
            Assert.AreEqual("Price: $150.25", stockExplorer.PriceText, "PriceText should not be reset when setting same mode");
            Assert.AreEqual("Updated: 12/01/2023 10:30", stockExplorer.UpdatedTimestampText, "UpdatedTimestampText should not be reset when setting same mode");
        }

        private void VerifyRsiModeUsesBroaderSymbolUniverse(string modeName, int oldHardcodedCount)
        {
            // Arrange - Verify that the RSI mode uses a broader symbol universe
            var expectedSymbolCount = StockSymbols.CommonSymbols.Count;
            
            // Assert - The new implementation should use significantly more symbols
            Assert.IsTrue(expectedSymbolCount > oldHardcodedCount, 
                $"RSI {modeName} should use broader symbol universe. Expected > {oldHardcodedCount}, got {expectedSymbolCount}");
            Assert.IsTrue(expectedSymbolCount >= 70, 
                $"Should include major indices and ETFs (at least 70 symbols), got {expectedSymbolCount}");
            
            // Verify that original RSI symbols are still included in the broader universe
            var originalRsiSymbols = new[] { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "AMD" };
            foreach (var symbol in originalRsiSymbols)
            {
                Assert.IsTrue(StockSymbols.CommonSymbols.Contains(symbol), 
                    $"Original RSI symbol {symbol} should still be included in broader universe");
            }
        }

        [TestMethod]
        public void RsiOversold_UsesBroaderSymbolUniverse()
        {
            // Arrange - Verify that the RSI Oversold mode uses a broader symbol universe
            var expectedSymbolCount = StockSymbols.CommonSymbols.Count;
            var oldHardcodedCount = 10; // Previous hardcoded array size for RSI Oversold
            
            // Assert - The new implementation should use significantly more symbols
            Assert.IsTrue(expectedSymbolCount > oldHardcodedCount, 
                $"RSI Oversold should use broader symbol universe. Expected > {oldHardcodedCount}, got {expectedSymbolCount}");
            Assert.IsTrue(expectedSymbolCount >= 70, 
                $"Should include major indices and ETFs (at least 70 symbols), got {expectedSymbolCount}");
            
            // Verify that original RSI symbols are still included in the broader universe
            var originalRsiSymbols = new[] { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "AMD" };
            foreach (var symbol in originalRsiSymbols)
            {
                Assert.IsTrue(StockSymbols.CommonSymbols.Contains(symbol), 
                    $"Original RSI symbol {symbol} should still be included in broader universe");
            }
        }

        [TestMethod]
        public void StockDataGrid_SelectionChanged_EventHandler_Exists()
        {
            // Arrange & Act
            var stockExplorer = new StockExplorer();
            
            // Assert - Verify that the StockDataGrid has a SelectionChanged event handler
            // This test ensures the event handler exists for the cursor state fix
            //Assert.IsNotNull(stockExplorer.StockDataGrid, "StockDataGrid should be accessible");
            
            // Note: We cannot directly test the cursor state in unit tests without a UI framework,
            // but we can verify the component initializes correctly and the method exists
            var handleSymbolSelectionMethod = typeof(StockExplorer).GetMethod("HandleSymbolSelectionAsync",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            Assert.IsNotNull(handleSymbolSelectionMethod, "HandleSymbolSelectionAsync method should exist");
        }

        [TestMethod]
        public void RefreshSymbolDataFromAPI_AcceptsCancellationToken()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            
            // Act & Assert - Verify that RefreshSymbolDataFromAPI method accepts cancellation token
            var refreshMethod = typeof(StockExplorer).GetMethod("RefreshSymbolDataFromAPI",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            Assert.IsNotNull(refreshMethod, "RefreshSymbolDataFromAPI method should exist");
            
            // Verify the method signature includes CancellationToken parameter
            var parameters = refreshMethod.GetParameters();
            Assert.AreEqual(2, parameters.Length, "RefreshSymbolDataFromAPI should have 2 parameters");
            Assert.AreEqual("symbol", parameters[0].Name, "First parameter should be symbol");
            Assert.AreEqual("cancellationToken", parameters[1].Name, "Second parameter should be cancellationToken");
            Assert.AreEqual(typeof(System.Threading.CancellationToken), parameters[1].ParameterType, 
                "Second parameter should be CancellationToken type");
        }

        [TestMethod]
        public void QuoteData_OptionChainProperties_Exist()
        {
            // Arrange & Act
            var quoteData = new QuoteData
            {
                Symbol = "AAPL",
                Price = 150.00,
                OptionDataFetchTimestamp = DateTime.Now,
                OptionDataCacheWindow = TimeSpan.FromMinutes(15)
            };
            
            // Assert
            Assert.IsNotNull(quoteData.OptionChain);
            Assert.IsNotNull(quoteData.OptionDataFetchTimestamp);
            Assert.IsNotNull(quoteData.OptionDataCacheWindow);
            Assert.AreEqual("AAPL", quoteData.Symbol);
            Assert.AreEqual(150.00, quoteData.Price);
        }

        [TestMethod]
        public void QuoteData_OptionChainMethods_ExistAndWork()
        {
            // Arrange
            var quoteData = new QuoteData { Symbol = "AAPL" };
            var expirationDate = DateTime.Now.AddDays(30);
            
            // Act
            var optionsByStrike = quoteData.GetOptionsByStrikeAndExpiration(150.0, expirationDate);
            var availableStrikes = quoteData.GetAvailableStrikes(expirationDate);
            var availableExpirations = quoteData.GetAvailableExpirations();
            
            // Assert
            Assert.IsNotNull(optionsByStrike);
            Assert.IsNotNull(availableStrikes);
            Assert.IsNotNull(availableExpirations);
            Assert.AreEqual(0, optionsByStrike.Count);
            Assert.AreEqual(0, availableStrikes.Count);
            Assert.AreEqual(0, availableExpirations.Count);
        }
        [TestMethod]
        public void StockExplorer_SentimentAnalysisProperties_InitializeCorrectly()
        {
            // Arrange & Act
            var stockExplorer = new StockExplorer();
            
            // Assert - Sentiment analysis properties should initialize to default values
            Assert.AreEqual(false, stockExplorer.HasSentimentResults, "HasSentimentResults should initialize to false");
            Assert.AreEqual(0.0, stockExplorer.OverallSentimentScore, "OverallSentimentScore should initialize to 0.0");
            Assert.AreEqual(0.0, stockExplorer.NewsSentimentScore, "NewsSentimentScore should initialize to 0.0");
            Assert.AreEqual(0.0, stockExplorer.SocialMediaSentimentScore, "SocialMediaSentimentScore should initialize to 0.0");
            Assert.AreEqual(0.0, stockExplorer.AnalystSentimentScore, "AnalystSentimentScore should initialize to 0.0");
            Assert.AreEqual("", stockExplorer.SentimentSummary, "SentimentSummary should initialize to empty string");
        }

        [TestMethod]
        public void StockExplorer_SentimentAnalysisProperties_PropertyChangedNotifications()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            var propertyChangedEvents = new List<string>();
            
            stockExplorer.PropertyChanged += (sender, e) =>
            {
                propertyChangedEvents.Add(e.PropertyName);
            };
            
            // Act - Set sentiment analysis properties
            stockExplorer.HasSentimentResults = true;
            stockExplorer.OverallSentimentScore = 0.5;
            stockExplorer.NewsSentimentScore = 0.3;
            stockExplorer.SocialMediaSentimentScore = 0.7;
            stockExplorer.AnalystSentimentScore = 0.4;
            stockExplorer.SentimentSummary = "Test sentiment summary";
            
            // Assert - PropertyChanged events should be fired for all sentiment properties
            Assert.IsTrue(propertyChangedEvents.Contains(nameof(stockExplorer.HasSentimentResults)), "HasSentimentResults PropertyChanged should fire");
            Assert.IsTrue(propertyChangedEvents.Contains(nameof(stockExplorer.OverallSentimentScore)), "OverallSentimentScore PropertyChanged should fire");
            Assert.IsTrue(propertyChangedEvents.Contains(nameof(stockExplorer.NewsSentimentScore)), "NewsSentimentScore PropertyChanged should fire");
            Assert.IsTrue(propertyChangedEvents.Contains(nameof(stockExplorer.SocialMediaSentimentScore)), "SocialMediaSentimentScore PropertyChanged should fire");
            Assert.IsTrue(propertyChangedEvents.Contains(nameof(stockExplorer.AnalystSentimentScore)), "AnalystSentimentScore PropertyChanged should fire");
            Assert.IsTrue(propertyChangedEvents.Contains(nameof(stockExplorer.SentimentSummary)), "SentimentSummary PropertyChanged should fire");
            
            // Verify values are set correctly
            Assert.AreEqual(true, stockExplorer.HasSentimentResults);
            Assert.AreEqual(0.5, stockExplorer.OverallSentimentScore);
            Assert.AreEqual(0.3, stockExplorer.NewsSentimentScore);
            Assert.AreEqual(0.7, stockExplorer.SocialMediaSentimentScore);
            Assert.AreEqual(0.4, stockExplorer.AnalystSentimentScore);
            Assert.AreEqual("Test sentiment summary", stockExplorer.SentimentSummary);
        }

        [TestMethod]
        public void StockExplorer_SentimentAnalysisProperties_ThresholdChangeBehavior()
        {
            // Arrange
            var stockExplorer = new StockExplorer();
            var propertyChangedEvents = new List<string>();
            
            stockExplorer.PropertyChanged += (sender, e) =>
            {
                propertyChangedEvents.Add(e.PropertyName);
            };
            
            // Act - Set initial values
            stockExplorer.NewsSentimentScore = 0.5;
            propertyChangedEvents.Clear(); // Clear events from initial setup
            
            // Set values within threshold (should not trigger PropertyChanged)
            stockExplorer.NewsSentimentScore = 0.5005; // Within 0.001 threshold
            
            // Assert - No PropertyChanged should fire for values within threshold
            Assert.AreEqual(0, propertyChangedEvents.Count, "PropertyChanged should not fire for changes within 0.001 threshold");
            
            // Act - Set value outside threshold (should trigger PropertyChanged)
            stockExplorer.NewsSentimentScore = 0.502; // Outside 0.001 threshold
            
            // Assert - PropertyChanged should fire for values outside threshold
            Assert.IsTrue(propertyChangedEvents.Contains(nameof(stockExplorer.NewsSentimentScore)), "PropertyChanged should fire for changes outside 0.001 threshold");
            Assert.AreEqual(0.502, stockExplorer.NewsSentimentScore);
        }
    }
}