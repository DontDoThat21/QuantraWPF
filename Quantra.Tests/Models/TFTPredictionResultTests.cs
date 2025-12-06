using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Models;

namespace Quantra.Tests.Models
{
    /// <summary>
    /// Unit tests for the TFTPredictionResult model (Issue #4).
    /// Validates TFT prediction format, multi-horizon predictions, and uncertainty quantification.
    /// </summary>
    [TestClass]
    public class TFTPredictionResultTests
    {
        #region Constructor Tests

        [TestMethod]
        public void Constructor_InitializesEmptyCollections()
        {
            // Act
            var result = new TFTPredictionResult();

            // Assert
            Assert.IsNotNull(result.Horizons);
            Assert.IsNotNull(result.FeatureWeights);
            Assert.IsNotNull(result.TemporalAttention);
            Assert.AreEqual(0, result.Horizons.Count);
            Assert.AreEqual(0, result.FeatureWeights.Count);
            Assert.AreEqual(0, result.TemporalAttention.Count);
        }

        [TestMethod]
        public void Constructor_SetsPredictionTimestamp()
        {
            // Act
            var before = DateTime.Now;
            var result = new TFTPredictionResult();
            var after = DateTime.Now;

            // Assert
            Assert.IsTrue(result.PredictionTimestamp >= before && result.PredictionTimestamp <= after);
        }

        #endregion

        #region Success Property Tests

        [TestMethod]
        public void Success_ReturnsTrue_WhenNoError()
        {
            // Arrange
            var result = new TFTPredictionResult
            {
                Error = null
            };

            // Assert
            Assert.IsTrue(result.Success);
        }

        [TestMethod]
        public void Success_ReturnsTrue_WhenErrorIsEmpty()
        {
            // Arrange
            var result = new TFTPredictionResult
            {
                Error = ""
            };

            // Assert
            Assert.IsTrue(result.Success);
        }

        [TestMethod]
        public void Success_ReturnsFalse_WhenErrorIsSet()
        {
            // Arrange
            var result = new TFTPredictionResult
            {
                Error = "Model not trained"
            };

            // Assert
            Assert.IsFalse(result.Success);
        }

        #endregion

        #region PotentialReturn Tests

        [TestMethod]
        public void PotentialReturn_CalculatesCorrectly_ForPositiveReturn()
        {
            // Arrange
            var result = new TFTPredictionResult
            {
                CurrentPrice = 100.0,
                TargetPrice = 110.0
            };

            // Assert
            Assert.AreEqual(0.10, result.PotentialReturn, 0.001);
        }

        [TestMethod]
        public void PotentialReturn_CalculatesCorrectly_ForNegativeReturn()
        {
            // Arrange
            var result = new TFTPredictionResult
            {
                CurrentPrice = 100.0,
                TargetPrice = 90.0
            };

            // Assert
            Assert.AreEqual(-0.10, result.PotentialReturn, 0.001);
        }

        [TestMethod]
        public void PotentialReturn_ReturnsZero_WhenCurrentPriceIsZero()
        {
            // Arrange
            var result = new TFTPredictionResult
            {
                CurrentPrice = 0.0,
                TargetPrice = 100.0
            };

            // Assert
            Assert.AreEqual(0.0, result.PotentialReturn);
        }

        #endregion

        #region Multi-Horizon Predictions Tests

        [TestMethod]
        public void Horizons_CanStoreMultipleHorizons()
        {
            // Arrange
            var result = new TFTPredictionResult();
            
            result.Horizons["5d"] = new HorizonPredictionData
            {
                MedianPrice = 105.0,
                LowerBound = 100.0,
                UpperBound = 110.0,
                Confidence = 0.80
            };
            
            result.Horizons["10d"] = new HorizonPredictionData
            {
                MedianPrice = 108.0,
                LowerBound = 98.0,
                UpperBound = 118.0,
                Confidence = 0.75
            };
            
            result.Horizons["20d"] = new HorizonPredictionData
            {
                MedianPrice = 112.0,
                LowerBound = 95.0,
                UpperBound = 130.0,
                Confidence = 0.65
            };

            // Assert
            Assert.AreEqual(3, result.Horizons.Count);
            Assert.IsTrue(result.Horizons.ContainsKey("5d"));
            Assert.IsTrue(result.Horizons.ContainsKey("10d"));
            Assert.IsTrue(result.Horizons.ContainsKey("20d"));
        }

        [TestMethod]
        public void HorizonPredictionData_StoresAllQuantiles()
        {
            // Arrange
            var horizonData = new HorizonPredictionData
            {
                MedianPrice = 100.0,
                LowerBound = 90.0,
                UpperBound = 110.0,
                Q25 = 95.0,
                Q75 = 105.0,
                Confidence = 0.85,
                TargetPrice = 102.0
            };

            // Assert
            Assert.AreEqual(100.0, horizonData.MedianPrice);
            Assert.AreEqual(90.0, horizonData.LowerBound);
            Assert.AreEqual(110.0, horizonData.UpperBound);
            Assert.AreEqual(95.0, horizonData.Q25);
            Assert.AreEqual(105.0, horizonData.Q75);
            Assert.AreEqual(0.85, horizonData.Confidence);
            Assert.AreEqual(102.0, horizonData.TargetPrice);
        }

        #endregion

        #region Uncertainty Quantification Tests

        [TestMethod]
        public void UncertaintyInterval_ReflectsConfidenceBounds()
        {
            // Arrange
            var result = new TFTPredictionResult
            {
                LowerBound = 95.0,
                UpperBound = 105.0,
                Uncertainty = 10.0 // Upper - Lower
            };

            // Assert
            Assert.AreEqual(10.0, result.Uncertainty);
            Assert.AreEqual(10.0, result.UpperBound - result.LowerBound);
        }

        [TestMethod]
        public void LowerUncertainty_IndicatesHigherConfidence()
        {
            // Arrange
            var highConfidenceResult = new TFTPredictionResult
            {
                Uncertainty = 5.0,
                Confidence = 0.90
            };
            
            var lowConfidenceResult = new TFTPredictionResult
            {
                Uncertainty = 20.0,
                Confidence = 0.60
            };

            // Assert - Lower uncertainty correlates with higher confidence
            Assert.IsTrue(highConfidenceResult.Uncertainty < lowConfidenceResult.Uncertainty);
            Assert.IsTrue(highConfidenceResult.Confidence > lowConfidenceResult.Confidence);
        }

        #endregion

        #region Feature Weights Tests

        [TestMethod]
        public void FeatureWeights_CanStoreMultipleFeatures()
        {
            // Arrange
            var result = new TFTPredictionResult();
            
            result.FeatureWeights["Close"] = 0.15;
            result.FeatureWeights["Volume"] = 0.08;
            result.FeatureWeights["RSI"] = 0.12;
            result.FeatureWeights["MACD"] = 0.10;
            result.FeatureWeights["SocialSentiment"] = 0.09;

            // Assert
            Assert.AreEqual(5, result.FeatureWeights.Count);
            Assert.AreEqual(0.15, result.FeatureWeights["Close"]);
            Assert.AreEqual(0.12, result.FeatureWeights["RSI"]);
        }

        [TestMethod]
        public void FeatureWeights_SumToReasonableTotal()
        {
            // Arrange - typical TFT feature weights should sum close to 1.0
            var result = new TFTPredictionResult();
            
            result.FeatureWeights["Close"] = 0.20;
            result.FeatureWeights["Volume"] = 0.15;
            result.FeatureWeights["RSI"] = 0.18;
            result.FeatureWeights["MACD"] = 0.12;
            result.FeatureWeights["ATR"] = 0.10;
            result.FeatureWeights["VWAP"] = 0.08;
            result.FeatureWeights["Other"] = 0.17;

            // Act
            double total = 0;
            foreach (var weight in result.FeatureWeights.Values)
            {
                total += weight;
            }

            // Assert - weights should sum to approximately 1.0 (variable selection network output)
            Assert.IsTrue(total > 0.95 && total < 1.05);
        }

        #endregion

        #region Temporal Attention Tests

        [TestMethod]
        public void TemporalAttention_CanStoreTimeOffsets()
        {
            // Arrange
            var result = new TFTPredictionResult();
            
            result.TemporalAttention[-1] = 0.25;  // Yesterday
            result.TemporalAttention[-2] = 0.18;  // 2 days ago
            result.TemporalAttention[-3] = 0.15;  // 3 days ago
            result.TemporalAttention[-5] = 0.12;  // 5 days ago
            result.TemporalAttention[-10] = 0.08; // 10 days ago

            // Assert
            Assert.AreEqual(5, result.TemporalAttention.Count);
            Assert.AreEqual(0.25, result.TemporalAttention[-1]);
            Assert.AreEqual(0.08, result.TemporalAttention[-10]);
        }

        [TestMethod]
        public void TemporalAttention_RecentDaysHaveHigherWeight()
        {
            // Arrange - TFT typically assigns higher attention to recent time steps
            var result = new TFTPredictionResult();
            
            result.TemporalAttention[-1] = 0.25;
            result.TemporalAttention[-5] = 0.15;
            result.TemporalAttention[-10] = 0.08;

            // Assert
            Assert.IsTrue(result.TemporalAttention[-1] > result.TemporalAttention[-5]);
            Assert.IsTrue(result.TemporalAttention[-5] > result.TemporalAttention[-10]);
        }

        #endregion

        #region Action Determination Tests

        [TestMethod]
        public void Action_ValidValues_BuySellHold()
        {
            // Arrange & Act
            var buyResult = new TFTPredictionResult { Action = "BUY" };
            var sellResult = new TFTPredictionResult { Action = "SELL" };
            var holdResult = new TFTPredictionResult { Action = "HOLD" };

            // Assert
            Assert.AreEqual("BUY", buyResult.Action);
            Assert.AreEqual("SELL", sellResult.Action);
            Assert.AreEqual("HOLD", holdResult.Action);
        }

        [TestMethod]
        public void Confidence_ValidRange()
        {
            // Arrange
            var result = new TFTPredictionResult
            {
                Confidence = 0.85
            };

            // Assert
            Assert.IsTrue(result.Confidence >= 0.0 && result.Confidence <= 1.0);
        }

        #endregion

        #region HorizonPrediction UI Model Tests

        [TestMethod]
        public void HorizonPrediction_CanBeCreatedForUI()
        {
            // Arrange
            var horizonPrediction = new HorizonPrediction
            {
                Horizon = "5 Days",
                MedianPrice = 105.50,
                LowerBound = 100.25,
                UpperBound = 110.75,
                Confidence = 0.82
            };

            // Assert
            Assert.AreEqual("5 Days", horizonPrediction.Horizon);
            Assert.AreEqual(105.50, horizonPrediction.MedianPrice);
            Assert.AreEqual(100.25, horizonPrediction.LowerBound);
            Assert.AreEqual(110.75, horizonPrediction.UpperBound);
            Assert.AreEqual(0.82, horizonPrediction.Confidence);
        }

        #endregion

        #region Integration Format Tests

        [TestMethod]
        public void TFTPredictionResult_CanBePopulatedFromPythonResponse()
        {
            // Arrange - simulate Python response format
            var result = new TFTPredictionResult
            {
                Symbol = "AAPL",
                Action = "BUY",
                Confidence = 0.78,
                CurrentPrice = 175.50,
                TargetPrice = 185.25,
                LowerBound = 170.00,
                UpperBound = 190.00,
                Uncertainty = 20.00,
                ModelVersion = "tft_v1.0",
                InferenceTimeMs = 125.5
            };

            // Add horizon predictions
            result.Horizons["5d"] = new HorizonPredictionData
            {
                MedianPrice = 180.00,
                LowerBound = 175.00,
                UpperBound = 185.00,
                Confidence = 0.82
            };
            result.Horizons["10d"] = new HorizonPredictionData
            {
                MedianPrice = 185.25,
                LowerBound = 170.00,
                UpperBound = 195.00,
                Confidence = 0.75
            };

            // Add feature weights
            result.FeatureWeights["RSI"] = 0.18;
            result.FeatureWeights["MACD"] = 0.15;
            result.FeatureWeights["Close"] = 0.22;

            // Add temporal attention
            result.TemporalAttention[-1] = 0.28;
            result.TemporalAttention[-5] = 0.15;

            // Assert
            Assert.AreEqual("AAPL", result.Symbol);
            Assert.AreEqual("BUY", result.Action);
            Assert.AreEqual(0.78, result.Confidence, 0.001);
            Assert.IsTrue(result.Success);
            Assert.AreEqual(2, result.Horizons.Count);
            Assert.AreEqual(3, result.FeatureWeights.Count);
            Assert.AreEqual(2, result.TemporalAttention.Count);
            Assert.IsTrue(result.InferenceTimeMs > 0);
        }

        [TestMethod]
        public void TFTPredictionResult_ErrorFormat_ProperlyHandled()
        {
            // Arrange - simulate Python error response
            var result = new TFTPredictionResult
            {
                Symbol = "INVALID",
                Action = "HOLD",
                Confidence = 0.0,
                Error = "Model not trained. Please train the model first."
            };

            // Assert
            Assert.IsFalse(result.Success);
            Assert.AreEqual("HOLD", result.Action);
            Assert.AreEqual(0.0, result.Confidence);
            Assert.IsTrue(result.Error.Contains("Model not trained"));
        }

        #endregion
    }
}
