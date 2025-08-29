using System;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Quantra.Tests
{
    [TestClass]
    public class StockExplorerPredictionTests
    {
        [TestMethod]
        public void TestPredictionSummaryProperty()
        {
            // Arrange
            var viewModel = new Quantra.ViewModels.StockExplorerViewModel();
            
            // Act
            var initialSummary = viewModel.PredictionSummary;
            viewModel.PredictionSummary = "Test prediction summary";
            var updatedSummary = viewModel.PredictionSummary;
            
            // Assert
            Assert.AreEqual("Ready to run predictions...", initialSummary);
            Assert.AreEqual("Test prediction summary", updatedSummary);
        }

        [TestMethod]
        public void TestPredictionLoadingState()
        {
            // Arrange
            var viewModel = new Quantra.ViewModels.StockExplorerViewModel();
            
            // Act
            var initialState = viewModel.IsPredictionLoading;
            viewModel.IsPredictionLoading = true;
            var loadingState = viewModel.IsPredictionLoading;
            viewModel.IsPredictionLoading = false;
            var finalState = viewModel.IsPredictionLoading;
            
            // Assert
            Assert.IsFalse(initialState);
            Assert.IsTrue(loadingState);
            Assert.IsFalse(finalState);
        }

        [TestMethod]
        public void TestCanRunPredictionsLogic()
        {
            // Arrange
            var viewModel = new Quantra.ViewModels.StockExplorerViewModel();
            
            // Act & Assert - No stocks initially
            Assert.IsFalse(viewModel.CanRunPredictions);
            
            // Add a mock stock
            var mockStock = new Quantra.Models.QuoteData
            {
                Symbol = "AAPL",
                Price = 150.0,
                Volume = 1000000
            };
            viewModel.CachedStocks.Add(mockStock);
            
            // Should be able to run predictions now
            Assert.IsTrue(viewModel.CanRunPredictions);
            
            // When loading is true, should not be able to run
            viewModel.IsPredictionLoading = true;
            Assert.IsFalse(viewModel.CanRunPredictions);
        }
    }
}