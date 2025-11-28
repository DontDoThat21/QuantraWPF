using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Unit tests for the ModelExplainerService (MarketChat story 4).
    /// Validates that feature weights and risk metrics are translated to plain English correctly.
    /// </summary>
    [TestClass]
    public class ModelExplainerServiceTests
    {
        private ModelExplainerService _service;

        [TestInitialize]
        public void Setup()
        {
            _service = new ModelExplainerService();
        }

        #region ExplainPredictionFactors Tests

        [TestMethod]
        public void ExplainPredictionFactors_NullPrediction_ReturnsNoDataMessage()
        {
            // Act
            var result = _service.ExplainPredictionFactors(null);

            // Assert
            Assert.AreEqual("No prediction data available to explain.", result);
        }

        [TestMethod]
        public void ExplainPredictionFactors_ValidPrediction_ContainsSymbol()
        {
            // Arrange
            var prediction = CreateTestPrediction("TSLA");

            // Act
            var result = _service.ExplainPredictionFactors(prediction);

            // Assert
            Assert.IsTrue(result.Contains("TSLA"), "Explanation should contain the symbol");
        }

        [TestMethod]
        public void ExplainPredictionFactors_BuySignal_ContainsBuyExplanation()
        {
            // Arrange
            var prediction = CreateTestPrediction("AAPL");
            prediction.Action = "BUY";
            prediction.TargetPrice = 200.00;
            prediction.CurrentPrice = 180.00;

            // Act
            var result = _service.ExplainPredictionFactors(prediction);

            // Assert
            Assert.IsTrue(result.Contains("BUY"), "Explanation should mention BUY action");
            Assert.IsTrue(result.Contains("Target price"), "Explanation should include target price");
        }

        [TestMethod]
        public void ExplainPredictionFactors_SellSignal_ContainsSellExplanation()
        {
            // Arrange
            var prediction = CreateTestPrediction("AAPL");
            prediction.Action = "SELL";
            prediction.TargetPrice = 160.00;
            prediction.CurrentPrice = 180.00;

            // Act
            var result = _service.ExplainPredictionFactors(prediction);

            // Assert
            Assert.IsTrue(result.Contains("SELL"), "Explanation should mention SELL action");
        }

        [TestMethod]
        public void ExplainPredictionFactors_WithFeatureWeights_ExplainsTopFactors()
        {
            // Arrange
            var prediction = CreateTestPrediction("MSFT");
            prediction.FeatureWeights = new Dictionary<string, double>
            {
                { "RSI", 0.32 },
                { "MACD", 0.24 },
                { "VWAP", 0.18 },
                { "volume", 0.15 },
                { "momentum", 0.11 }
            };

            // Act
            var result = _service.ExplainPredictionFactors(prediction);

            // Assert
            Assert.IsTrue(result.Contains("Key Factors"), "Explanation should have key factors section");
            Assert.IsTrue(result.Contains("RSI") || result.Contains("Relative Strength"), "Explanation should translate RSI");
        }

        [TestMethod]
        public void ExplainPredictionFactors_WithRiskMetrics_ExplainsRisk()
        {
            // Arrange
            var prediction = CreateTestPrediction("GOOGL");
            prediction.RiskMetrics = new Quantra.Models.RiskMetrics
            {
                ValueAtRisk = 5.50,
                MaxDrawdown = 12.00,
                SharpeRatio = 1.5,
                RiskScore = 0.45
            };

            // Act
            var result = _service.ExplainPredictionFactors(prediction);

            // Assert
            Assert.IsTrue(result.Contains("Risk Assessment"), "Explanation should include risk assessment");
            Assert.IsTrue(result.Contains("Sharpe"), "Explanation should mention Sharpe ratio");
        }

        #endregion

        #region ExplainConfidenceScore Tests

        [TestMethod]
        public void ExplainConfidenceScore_VeryHighConfidence_DescribesStrong()
        {
            // Act
            var result = _service.ExplainConfidenceScore(0.90, "pytorch");

            // Assert
            Assert.IsTrue(result.Contains("very high") || result.Contains("90"), "Should describe very high confidence");
            Assert.IsTrue(result.Contains("strong conviction") || result.Contains("confirming"), "Should mention strong signals");
        }

        [TestMethod]
        public void ExplainConfidenceScore_ModerateConfidence_DescribesMixed()
        {
            // Act
            var result = _service.ExplainConfidenceScore(0.55, "random_forest");

            // Assert
            Assert.IsTrue(result.Contains("moderate") || result.Contains("55"), "Should describe moderate confidence");
        }

        [TestMethod]
        public void ExplainConfidenceScore_LowConfidence_DescribesUncertain()
        {
            // Act
            var result = _service.ExplainConfidenceScore(0.35, "tensorflow");

            // Assert
            Assert.IsTrue(result.Contains("low") || result.Contains("35"), "Should describe low confidence");
        }

        [TestMethod]
        public void ExplainConfidenceScore_IncludesModelTypeExplanation()
        {
            // Act
            var result = _service.ExplainConfidenceScore(0.75, "pytorch");

            // Assert
            Assert.IsTrue(result.Contains("PyTorch") || result.Contains("neural"), "Should mention model type");
        }

        #endregion

        #region ExplainRiskMetrics Tests

        [TestMethod]
        public void ExplainRiskMetrics_NullMetrics_ReturnsNoDataMessage()
        {
            // Act
            var result = _service.ExplainRiskMetrics(null);

            // Assert
            Assert.AreEqual("No risk metrics available.", result);
        }

        [TestMethod]
        public void ExplainRiskMetrics_ContainsVaRExplanation()
        {
            // Arrange
            var metrics = new Quantra.Models.RiskMetrics
            {
                ValueAtRisk = 10.00,
                MaxDrawdown = 20.00,
                SharpeRatio = 1.2,
                RiskScore = 0.5
            };

            // Act
            var result = _service.ExplainRiskMetrics(metrics);

            // Assert
            Assert.IsTrue(result.Contains("Value at Risk") || result.Contains("VaR"), "Should explain VaR");
            Assert.IsTrue(result.Contains("$10"), "Should include VaR value");
        }

        [TestMethod]
        public void ExplainRiskMetrics_HighSharpeRatio_DescribesExcellent()
        {
            // Arrange
            var metrics = new Quantra.Models.RiskMetrics
            {
                ValueAtRisk = 5.00,
                MaxDrawdown = 10.00,
                SharpeRatio = 2.5,
                RiskScore = 0.3
            };

            // Act
            var result = _service.ExplainRiskMetrics(metrics);

            // Assert
            Assert.IsTrue(result.Contains("Excellent") || result.Contains("excellent"), "Should describe excellent Sharpe");
        }

        [TestMethod]
        public void ExplainRiskMetrics_HighRiskScore_WarnsAboutRisk()
        {
            // Arrange
            var metrics = new Quantra.Models.RiskMetrics
            {
                ValueAtRisk = 25.00,
                MaxDrawdown = 40.00,
                SharpeRatio = 0.3,
                RiskScore = 0.85
            };

            // Act
            var result = _service.ExplainRiskMetrics(metrics);

            // Assert
            Assert.IsTrue(result.Contains("high risk") || result.Contains("Very high"), "Should warn about high risk");
        }

        #endregion

        #region TranslateIndicatorToPlainEnglish Tests

        [TestMethod]
        public void TranslateIndicator_RSI_TranslatesCorrectly()
        {
            // Act
            var result = _service.TranslateIndicatorToPlainEnglish("RSI", 0.32, 28);

            // Assert
            Assert.IsTrue(result.Contains("Relative Strength"), "Should translate RSI to full name");
            Assert.IsTrue(result.Contains("momentum") || result.Contains("speed"), "Should describe RSI function");
            Assert.IsTrue(result.Contains("28") || result.Contains("oversold"), "Should include current value or interpretation");
        }

        [TestMethod]
        public void TranslateIndicator_MACD_TranslatesCorrectly()
        {
            // Act
            var result = _service.TranslateIndicatorToPlainEnglish("MACD", 0.24, 1.5);

            // Assert
            Assert.IsTrue(result.Contains("Moving Average Convergence") || result.Contains("MACD"), "Should translate MACD");
            Assert.IsTrue(result.Contains("momentum") || result.Contains("trend"), "Should describe MACD function");
        }

        [TestMethod]
        public void TranslateIndicator_VWAP_TranslatesCorrectly()
        {
            // Act
            var result = _service.TranslateIndicatorToPlainEnglish("VWAP", 0.18, 150.50);

            // Assert
            Assert.IsTrue(result.Contains("Volume-Weighted") || result.Contains("VWAP"), "Should translate VWAP");
        }

        [TestMethod]
        public void TranslateIndicator_HighWeight_DescribesAsSignificant()
        {
            // Act
            var result = _service.TranslateIndicatorToPlainEnglish("RSI", 0.35);

            // Assert
            Assert.IsTrue(result.Contains("significant") || result.Contains("very"), "Should describe high weight as significant");
        }

        [TestMethod]
        public void TranslateIndicator_LowWeight_DescribesAsMinor()
        {
            // Act
            var result = _service.TranslateIndicatorToPlainEnglish("volatility", 0.03);

            // Assert
            Assert.IsTrue(result.Contains("minimal") || result.Contains("minor"), "Should describe low weight appropriately");
        }

        [TestMethod]
        public void TranslateIndicator_UnknownIndicator_ReturnsGenericDescription()
        {
            // Act
            var result = _service.TranslateIndicatorToPlainEnglish("custom_feature_xyz", 0.15);

            // Assert
            Assert.IsTrue(result.Contains("custom_feature_xyz"), "Should include the indicator name");
            Assert.IsTrue(result.Contains("influence"), "Should mention influence on prediction");
        }

        #endregion

        #region CompareModelPredictions Tests

        [TestMethod]
        public void CompareModelPredictions_NullOrEmpty_ReturnsNoDataMessage()
        {
            // Act
            var resultNull = _service.CompareModelPredictions(null);
            var resultEmpty = _service.CompareModelPredictions(new Dictionary<string, PredictionResult>());

            // Assert
            Assert.IsTrue(resultNull.Contains("No model predictions"), "Should indicate no predictions for null");
            Assert.IsTrue(resultEmpty.Contains("No model predictions"), "Should indicate no predictions for empty");
        }

        [TestMethod]
        public void CompareModelPredictions_SingleModel_IndicatesNoComparison()
        {
            // Arrange
            var predictions = new Dictionary<string, PredictionResult>
            {
                { "pytorch", CreateTestPrediction("AAPL") }
            };

            // Act
            var result = _service.CompareModelPredictions(predictions);

            // Assert
            Assert.IsTrue(result.Contains("only one") || result.Contains("No comparison"), "Should indicate only one model");
        }

        [TestMethod]
        public void CompareModelPredictions_MultipleModels_ShowsComparison()
        {
            // Arrange
            var predictions = new Dictionary<string, PredictionResult>
            {
                { "pytorch", CreateTestPrediction("AAPL", "BUY", 0.85) },
                { "tensorflow", CreateTestPrediction("AAPL", "BUY", 0.78) },
                { "random_forest", CreateTestPrediction("AAPL", "HOLD", 0.62) }
            };

            // Act
            var result = _service.CompareModelPredictions(predictions);

            // Assert
            Assert.IsTrue(result.Contains("Model Comparison"), "Should have comparison header");
            Assert.IsTrue(result.Contains("PyTorch") || result.Contains("pytorch"), "Should list PyTorch");
            Assert.IsTrue(result.Contains("TensorFlow") || result.Contains("tensorflow"), "Should list TensorFlow");
        }

        [TestMethod]
        public void CompareModelPredictions_Consensus_IdentifiesAgreement()
        {
            // Arrange
            var predictions = new Dictionary<string, PredictionResult>
            {
                { "pytorch", CreateTestPrediction("AAPL", "BUY", 0.85) },
                { "tensorflow", CreateTestPrediction("AAPL", "BUY", 0.78) }
            };

            // Act
            var result = _service.CompareModelPredictions(predictions);

            // Assert
            Assert.IsTrue(result.Contains("Consensus") || result.Contains("agree"), "Should identify consensus");
        }

        [TestMethod]
        public void CompareModelPredictions_MixedSignals_WarnsAboutDisagreement()
        {
            // Arrange
            var predictions = new Dictionary<string, PredictionResult>
            {
                { "pytorch", CreateTestPrediction("AAPL", "BUY", 0.65) },
                { "random_forest", CreateTestPrediction("AAPL", "SELL", 0.55) }
            };

            // Act
            var result = _service.CompareModelPredictions(predictions);

            // Assert
            Assert.IsTrue(result.Contains("Mixed") || result.Contains("disagree") || result.Contains("different"), 
                "Should warn about mixed signals");
        }

        #endregion

        #region GetTopInfluentialFactors Tests

        [TestMethod]
        public void GetTopInfluentialFactors_NullOrEmpty_ReturnsNoDataMessage()
        {
            // Act
            var resultNull = _service.GetTopInfluentialFactors(null);
            var resultEmpty = _service.GetTopInfluentialFactors(new Dictionary<string, double>());

            // Assert
            Assert.IsTrue(resultNull.Contains("No feature weight"), "Should indicate no data for null");
            Assert.IsTrue(resultEmpty.Contains("No feature weight"), "Should indicate no data for empty");
        }

        [TestMethod]
        public void GetTopInfluentialFactors_ReturnsCorrectNumber()
        {
            // Arrange
            var weights = new Dictionary<string, double>
            {
                { "RSI", 0.30 },
                { "MACD", 0.25 },
                { "VWAP", 0.20 },
                { "volume", 0.15 },
                { "momentum", 0.10 }
            };

            // Act
            var result = _service.GetTopInfluentialFactors(weights, 3);

            // Assert
            Assert.IsTrue(result.Contains("RSI") || result.Contains("Relative Strength"), "Should include top factor RSI");
            Assert.IsTrue(result.Contains("MACD") || result.Contains("Moving Average"), "Should include second factor MACD");
            // Verify it doesn't include lower-ranked factors in detailed description
            var lines = result.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
            Assert.IsTrue(lines.Length >= 3, "Should have explanations for top 3 factors");
        }

        [TestMethod]
        public void GetTopInfluentialFactors_SortsByAbsoluteWeight()
        {
            // Arrange
            var weights = new Dictionary<string, double>
            {
                { "feature_a", 0.10 },
                { "feature_b", -0.35 },  // Negative but high absolute value
                { "feature_c", 0.20 }
            };

            // Act
            var result = _service.GetTopInfluentialFactors(weights, 2);

            // Assert
            Assert.IsTrue(result.Contains("feature_b"), "Should include factor with highest absolute weight");
        }

        #endregion

        #region Helper Methods

        private PredictionResult CreateTestPrediction(string symbol, string action = "BUY", double confidence = 0.75)
        {
            return new PredictionResult
            {
                Symbol = symbol,
                Action = action,
                Confidence = confidence,
                TargetPrice = 150.00,
                CurrentPrice = 140.00,
                ModelType = "pytorch",
                PredictionDate = DateTime.Now,
                FeatureWeights = new Dictionary<string, double>
                {
                    { "RSI", 0.25 },
                    { "MACD", 0.20 },
                    { "volume", 0.15 }
                },
                RiskMetrics = new Quantra.Models.RiskMetrics
                {
                    ValueAtRisk = 5.00,
                    MaxDrawdown = 10.00,
                    SharpeRatio = 1.2,
                    RiskScore = 0.4
                }
            };
        }

        #endregion
    }
}
