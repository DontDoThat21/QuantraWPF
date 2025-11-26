using System;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Services;
using Quantra.ViewModels;
using Quantra; // For QuoteData

namespace Quantra.Tests
{
    /// <summary>
    /// Tests for StockExplorerViewModel prediction-related functionality.
    /// Note: These tests use stub services since the ViewModel requires dependency injection.
    /// </summary>
    [TestClass]
    public class StockExplorerPredictionTests
    {
        /// <summary>
        /// Creates a test ViewModel with stub services.
        /// The stub services provide minimal implementations for testing ViewModel logic.
        /// </summary>
        private StockExplorerViewModel CreateTestViewModel()
        {
            // Create stub services using test helper factory method
            var stubCacheService = TestServiceFactory.CreateStubStockDataCacheService();
            var stubAlphaVantageService = TestServiceFactory.CreateStubAlphaVantageService();
            var stubInferenceService = TestServiceFactory.CreateStubRealTimeInferenceService();
            var stubPredictionCacheService = TestServiceFactory.CreateStubPredictionCacheService();

            return new StockExplorerViewModel(
                stubCacheService,
                stubAlphaVantageService,
                stubInferenceService,
                stubPredictionCacheService);
        }

        [TestMethod]
        public void TestPredictionSummaryProperty()
        {
            // Arrange
            var viewModel = CreateTestViewModel();
            
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
            var viewModel = CreateTestViewModel();
            
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
            var viewModel = CreateTestViewModel();
            
            // Act & Assert - No stocks initially
            Assert.IsFalse(viewModel.CanRunPredictions);
            
            // Add a mock stock
            var mockStock = new QuoteData
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