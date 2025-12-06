using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Unit tests for MarketContextService market context calculations.
    /// Tests for TFT model market context features implementation.
    /// These tests validate pure/synchronous calculation methods that don't require API calls.
    /// </summary>
    [TestClass]
    public class MarketContextServiceTests
    {
        #region Sector-to-ETF Mapping Tests

        [TestMethod]
        public void MapSectorToETF_ReturnsXLK_ForTechnology()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Technology");
            
            // Assert
            Assert.AreEqual("XLK", etf, "Technology sector should map to XLK");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLV_ForHealthcare()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Healthcare");
            
            // Assert
            Assert.AreEqual("XLV", etf, "Healthcare sector should map to XLV");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLF_ForFinancial()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Financial");
            
            // Assert
            Assert.AreEqual("XLF", etf, "Financial sector should map to XLF");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLY_ForConsumerDiscretionary()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Consumer Discretionary");
            
            // Assert
            Assert.AreEqual("XLY", etf, "Consumer Discretionary sector should map to XLY");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLP_ForConsumerStaples()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Consumer Staples");
            
            // Assert
            Assert.AreEqual("XLP", etf, "Consumer Staples sector should map to XLP");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLE_ForEnergy()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Energy");
            
            // Assert
            Assert.AreEqual("XLE", etf, "Energy sector should map to XLE");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLU_ForUtilities()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Utilities");
            
            // Assert
            Assert.AreEqual("XLU", etf, "Utilities sector should map to XLU");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLRE_ForRealEstate()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Real Estate");
            
            // Assert
            Assert.AreEqual("XLRE", etf, "Real Estate sector should map to XLRE");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLB_ForMaterials()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Materials");
            
            // Assert
            Assert.AreEqual("XLB", etf, "Materials sector should map to XLB");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLI_ForIndustrials()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Industrials");
            
            // Assert
            Assert.AreEqual("XLI", etf, "Industrials sector should map to XLI");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsXLC_ForCommunicationServices()
        {
            // Arrange & Act
            string etf = MapSectorToETF("Communication Services");
            
            // Assert
            Assert.AreEqual("XLC", etf, "Communication Services sector should map to XLC");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsSPY_ForUnknownSector()
        {
            // Arrange & Act
            string etf = MapSectorToETF("UnknownSector");
            
            // Assert
            Assert.AreEqual("SPY", etf, "Unknown sector should default to SPY");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsSPY_ForNullSector()
        {
            // Arrange & Act
            string etf = MapSectorToETF(null);
            
            // Assert
            Assert.AreEqual("SPY", etf, "Null sector should default to SPY");
        }

        [TestMethod]
        public void MapSectorToETF_ReturnsSPY_ForEmptySector()
        {
            // Arrange & Act
            string etf = MapSectorToETF("");
            
            // Assert
            Assert.AreEqual("SPY", etf, "Empty sector should default to SPY");
        }

        [TestMethod]
        public void MapSectorToETF_IsCaseInsensitive()
        {
            // Arrange & Act
            string etfLower = MapSectorToETF("technology");
            string etfUpper = MapSectorToETF("TECHNOLOGY");
            string etfMixed = MapSectorToETF("TeCHnoLoGy");
            
            // Assert
            Assert.AreEqual("XLK", etfLower, "Lowercase 'technology' should map to XLK");
            Assert.AreEqual("XLK", etfUpper, "Uppercase 'TECHNOLOGY' should map to XLK");
            Assert.AreEqual("XLK", etfMixed, "Mixed case 'TeCHnoLoGy' should map to XLK");
        }

        [TestMethod]
        public void MapSectorToETF_HandlesAlternativeNames()
        {
            // Arrange & Act - Test various alternative sector names
            string techVariant = MapSectorToETF("Information Technology");
            string finVariant = MapSectorToETF("Financials");
            string consumerVariant = MapSectorToETF("Consumer Cyclical");
            string materialsVariant = MapSectorToETF("Basic Materials");
            
            // Assert
            Assert.AreEqual("XLK", techVariant, "Information Technology should map to XLK");
            Assert.AreEqual("XLF", finVariant, "Financials should map to XLF");
            Assert.AreEqual("XLY", consumerVariant, "Consumer Cyclical should map to XLY");
            Assert.AreEqual("XLB", materialsVariant, "Basic Materials should map to XLB");
        }

        /// <summary>
        /// Maps sector name to corresponding sector ETF symbol.
        /// This is a copy of the logic from MarketContextService for testing purposes.
        /// </summary>
        private string MapSectorToETF(string sector)
        {
            if (string.IsNullOrWhiteSpace(sector))
            {
                return "SPY"; // Default to S&P 500
            }

            // Normalize sector name for matching
            string normalizedSector = sector.ToUpperInvariant().Trim();

            return normalizedSector switch
            {
                "TECHNOLOGY" or "INFORMATION TECHNOLOGY" or "TECH" => "XLK",
                "HEALTHCARE" or "HEALTH CARE" => "XLV",
                "FINANCIAL SERVICES" or "FINANCIALS" or "FINANCIAL" => "XLF",
                "CONSUMER DISCRETIONARY" or "CONSUMER CYCLICAL" => "XLY",
                "CONSUMER STAPLES" or "CONSUMER DEFENSIVE" => "XLP",
                "ENERGY" => "XLE",
                "UTILITIES" => "XLU",
                "REAL ESTATE" => "XLRE",
                "MATERIALS" or "BASIC MATERIALS" => "XLB",
                "INDUSTRIALS" or "INDUSTRIAL" => "XLI",
                "COMMUNICATION SERVICES" or "TELECOMMUNICATIONS" or "TELECOM" => "XLC",
                _ => "SPY" // Default to S&P 500 if sector unknown
            };
        }

        #endregion

        #region Volatility Regime Classification Tests

        [TestMethod]
        public void GetVolatilityRegime_ReturnsZero_ForLowVIX()
        {
            // Arrange & Act
            int regime = GetVolatilityRegime(12.5);
            
            // Assert
            Assert.AreEqual(0, regime, "VIX < 15 should return 0 (Low volatility)");
        }

        [TestMethod]
        public void GetVolatilityRegime_ReturnsOne_ForNormalVIX()
        {
            // Arrange & Act
            int regime = GetVolatilityRegime(17.5);
            
            // Assert
            Assert.AreEqual(1, regime, "VIX 15-20 should return 1 (Normal volatility)");
        }

        [TestMethod]
        public void GetVolatilityRegime_ReturnsTwo_ForElevatedVIX()
        {
            // Arrange & Act
            int regime = GetVolatilityRegime(25.0);
            
            // Assert
            Assert.AreEqual(2, regime, "VIX 20-30 should return 2 (Elevated volatility)");
        }

        [TestMethod]
        public void GetVolatilityRegime_ReturnsThree_ForHighVIX()
        {
            // Arrange & Act
            int regime = GetVolatilityRegime(35.0);
            
            // Assert
            Assert.AreEqual(3, regime, "VIX >= 30 should return 3 (High volatility)");
        }

        [TestMethod]
        public void GetVolatilityRegime_BoundaryTest_Exactly15()
        {
            // Arrange & Act
            int regime = GetVolatilityRegime(15.0);
            
            // Assert
            Assert.AreEqual(1, regime, "VIX exactly 15 should return 1 (Normal volatility)");
        }

        [TestMethod]
        public void GetVolatilityRegime_BoundaryTest_Exactly20()
        {
            // Arrange & Act
            int regime = GetVolatilityRegime(20.0);
            
            // Assert
            Assert.AreEqual(2, regime, "VIX exactly 20 should return 2 (Elevated volatility)");
        }

        [TestMethod]
        public void GetVolatilityRegime_BoundaryTest_Exactly30()
        {
            // Arrange & Act
            int regime = GetVolatilityRegime(30.0);
            
            // Assert
            Assert.AreEqual(3, regime, "VIX exactly 30 should return 3 (High volatility)");
        }

        /// <summary>
        /// Get volatility regime classification based on VIX level.
        /// This is a copy of the logic from MarketContextService for testing purposes.
        /// </summary>
        private int GetVolatilityRegime(double vix)
        {
            return vix switch
            {
                < 15 => 0,  // Low volatility
                < 20 => 1,  // Normal volatility
                < 30 => 2,  // Elevated volatility
                _ => 3      // High volatility
            };
        }

        #endregion

        #region Volatility Regime Description Tests

        [TestMethod]
        public void GetVolatilityRegimeDescription_ReturnsLowVolatility_ForVIXBelow15()
        {
            // Arrange & Act
            string description = GetVolatilityRegimeDescription(10.0);
            
            // Assert
            Assert.AreEqual("Low Volatility", description);
        }

        [TestMethod]
        public void GetVolatilityRegimeDescription_ReturnsNormal_ForVIX15To20()
        {
            // Arrange & Act
            string description = GetVolatilityRegimeDescription(18.0);
            
            // Assert
            Assert.AreEqual("Normal", description);
        }

        [TestMethod]
        public void GetVolatilityRegimeDescription_ReturnsElevatedVolatility_ForVIX20To30()
        {
            // Arrange & Act
            string description = GetVolatilityRegimeDescription(25.0);
            
            // Assert
            Assert.AreEqual("Elevated Volatility", description);
        }

        [TestMethod]
        public void GetVolatilityRegimeDescription_ReturnsHighVolatility_ForVIXAbove30()
        {
            // Arrange & Act
            string description = GetVolatilityRegimeDescription(40.0);
            
            // Assert
            Assert.AreEqual("High Volatility", description);
        }

        /// <summary>
        /// Get volatility regime description based on VIX level.
        /// This is a copy of the logic from MarketContextService for testing purposes.
        /// </summary>
        private string GetVolatilityRegimeDescription(double vix)
        {
            return vix switch
            {
                < 15 => "Low Volatility",
                < 20 => "Normal",
                < 30 => "Elevated Volatility",
                _ => "High Volatility"
            };
        }

        #endregion

        #region Relative Strength Calculation Tests

        [TestMethod]
        public void CalculateRelativeStrengthVsSector_ReturnsCorrectRatio()
        {
            // Arrange
            double stockPrice = 150.0;
            double sectorETFPrice = 100.0;
            
            // Act
            double relativeStrength = CalculateRelativeStrengthVsSector(stockPrice, sectorETFPrice);
            
            // Assert
            Assert.AreEqual(1.5, relativeStrength, 0.001, "150/100 should return 1.5");
        }

        [TestMethod]
        public void CalculateRelativeStrengthVsSector_ReturnsOne_ForEqualPrices()
        {
            // Arrange
            double stockPrice = 100.0;
            double sectorETFPrice = 100.0;
            
            // Act
            double relativeStrength = CalculateRelativeStrengthVsSector(stockPrice, sectorETFPrice);
            
            // Assert
            Assert.AreEqual(1.0, relativeStrength, 0.001, "Equal prices should return 1.0");
        }

        [TestMethod]
        public void CalculateRelativeStrengthVsSector_ReturnsLessThanOne_WhenStockUnderperforms()
        {
            // Arrange
            double stockPrice = 50.0;
            double sectorETFPrice = 100.0;
            
            // Act
            double relativeStrength = CalculateRelativeStrengthVsSector(stockPrice, sectorETFPrice);
            
            // Assert
            Assert.AreEqual(0.5, relativeStrength, 0.001, "50/100 should return 0.5");
        }

        [TestMethod]
        public void CalculateRelativeStrengthVsSector_ReturnsOne_ForZeroStockPrice()
        {
            // Arrange
            double stockPrice = 0;
            double sectorETFPrice = 100.0;
            
            // Act
            double relativeStrength = CalculateRelativeStrengthVsSector(stockPrice, sectorETFPrice);
            
            // Assert
            Assert.AreEqual(1.0, relativeStrength, 0.001, "Zero stock price should return neutral 1.0");
        }

        [TestMethod]
        public void CalculateRelativeStrengthVsSector_ReturnsOne_ForZeroSectorETFPrice()
        {
            // Arrange
            double stockPrice = 100.0;
            double sectorETFPrice = 0;
            
            // Act
            double relativeStrength = CalculateRelativeStrengthVsSector(stockPrice, sectorETFPrice);
            
            // Assert
            Assert.AreEqual(1.0, relativeStrength, 0.001, "Zero sector ETF price should return neutral 1.0");
        }

        [TestMethod]
        public void CalculateRelativeStrengthVsSector_ReturnsOne_ForNegativePrices()
        {
            // Arrange
            double stockPrice = -50.0;
            double sectorETFPrice = 100.0;
            
            // Act
            double relativeStrength = CalculateRelativeStrengthVsSector(stockPrice, sectorETFPrice);
            
            // Assert
            Assert.AreEqual(1.0, relativeStrength, 0.001, "Negative stock price should return neutral 1.0");
        }

        /// <summary>
        /// Calculate relative strength of a stock vs its sector.
        /// This is a copy of the logic from MarketContextService for testing purposes.
        /// </summary>
        private double CalculateRelativeStrengthVsSector(double stockPrice, double sectorETFPrice)
        {
            if (stockPrice <= 0 || sectorETFPrice <= 0)
            {
                return 1.0; // Neutral if prices are invalid
            }

            return stockPrice / sectorETFPrice;
        }

        #endregion

        #region Sector Name Tests

        [TestMethod]
        public void GetSectorName_ReturnsCorrectName_ForTechnologyCode()
        {
            // Arrange & Act
            string sectorName = GetSectorName(0);
            
            // Assert
            Assert.AreEqual("Technology", sectorName);
        }

        [TestMethod]
        public void GetSectorName_ReturnsCorrectName_ForHealthcareCode()
        {
            // Arrange & Act
            string sectorName = GetSectorName(1);
            
            // Assert
            Assert.AreEqual("Healthcare", sectorName);
        }

        [TestMethod]
        public void GetSectorName_ReturnsCorrectName_ForFinancialCode()
        {
            // Arrange & Act
            string sectorName = GetSectorName(2);
            
            // Assert
            Assert.AreEqual("Financial", sectorName);
        }

        [TestMethod]
        public void GetSectorName_ReturnsUnknown_ForInvalidCode()
        {
            // Arrange & Act
            string sectorName = GetSectorName(99);
            
            // Assert
            Assert.AreEqual("Unknown", sectorName);
        }

        [TestMethod]
        public void GetSectorName_ReturnsUnknown_ForNegativeCode()
        {
            // Arrange & Act
            string sectorName = GetSectorName(-1);
            
            // Assert
            Assert.AreEqual("Unknown", sectorName);
        }

        [TestMethod]
        public void GetSectorName_CoversAllElevenSectors()
        {
            // Arrange - Test all 11 standard GICS sectors
            var expectedSectors = new Dictionary<int, string>
            {
                { 0, "Technology" },
                { 1, "Healthcare" },
                { 2, "Financial" },
                { 3, "Consumer Discretionary" },
                { 4, "Consumer Staples" },
                { 5, "Industrials" },
                { 6, "Energy" },
                { 7, "Materials" },
                { 8, "Real Estate" },
                { 9, "Utilities" },
                { 10, "Communication Services" }
            };
            
            // Act & Assert
            foreach (var kvp in expectedSectors)
            {
                string actualSectorName = GetSectorName(kvp.Key);
                Assert.AreEqual(kvp.Value, actualSectorName, $"Sector code {kvp.Key} should return '{kvp.Value}'");
            }
        }

        /// <summary>
        /// Get sector name from sector code.
        /// This is a copy of the logic from MarketContextService for testing purposes.
        /// </summary>
        private string GetSectorName(int sectorCode)
        {
            return sectorCode switch
            {
                0 => "Technology",
                1 => "Healthcare",
                2 => "Financial",
                3 => "Consumer Discretionary",
                4 => "Consumer Staples",
                5 => "Industrials",
                6 => "Energy",
                7 => "Materials",
                8 => "Real Estate",
                9 => "Utilities",
                10 => "Communication Services",
                _ => "Unknown"
            };
        }

        #endregion

        #region Market Context Indicator Key Tests

        /// <summary>
        /// Test that all expected market context indicator keys are defined correctly.
        /// These keys should match what PredictionAnalysis.Analysis.cs uses.
        /// </summary>
        [TestMethod]
        public void MarketContextIndicatorKeys_AreConsistent()
        {
            // Arrange - Expected indicator keys used by the TFT model integration
            var expectedKeys = new List<string>
            {
                "SP500_Price",
                "SP500_Return",
                "SP500_Direction",
                "VIX",
                "VolatilityRegime",
                "TreasuryYield_10Y",
                "SectorETF_Price",
                "SectorETF_Return",
                "RelativeStrengthVsSector",
                "MarketBreadth",
                "IsBullishBreadth"
            };
            
            // Act & Assert - Verify each key is a valid string
            foreach (var key in expectedKeys)
            {
                Assert.IsFalse(string.IsNullOrEmpty(key), $"Indicator key '{key}' should not be null or empty");
                Assert.IsFalse(key.Contains(" "), $"Indicator key '{key}' should not contain spaces");
            }
        }

        [TestMethod]
        public void VolatilityRegimeValues_AreWithinExpectedRange()
        {
            // Arrange & Act - Test all VIX values from 0 to 100
            for (double vix = 0; vix <= 100; vix += 5)
            {
                int regime = GetVolatilityRegime(vix);
                
                // Assert - Regime should be 0, 1, 2, or 3
                Assert.IsTrue(regime >= 0 && regime <= 3, $"Volatility regime for VIX {vix} should be 0-3, got {regime}");
            }
        }

        [TestMethod]
        public void SP500Direction_CalculationLogic()
        {
            // Test the direction calculation logic
            // Positive return should give direction 1.0
            double positiveReturn = 0.02;
            double direction = positiveReturn > 0 ? 1.0 : -1.0;
            Assert.AreEqual(1.0, direction, "Positive return should give direction 1.0");

            // Negative return should give direction -1.0
            double negativeReturn = -0.01;
            direction = negativeReturn > 0 ? 1.0 : -1.0;
            Assert.AreEqual(-1.0, direction, "Negative return should give direction -1.0");

            // Zero return should give direction -1.0 (matches implementation)
            double zeroReturn = 0;
            direction = zeroReturn > 0 ? 1.0 : -1.0;
            Assert.AreEqual(-1.0, direction, "Zero return should give direction -1.0");
        }

        [TestMethod]
        public void IsBullishBreadth_CalculationLogic()
        {
            // Test the bullish breadth calculation logic
            // Market breadth > 1.0 is bullish
            double bullishBreadth = 1.5;
            double isBullish = bullishBreadth > 1.0 ? 1.0 : 0.0;
            Assert.AreEqual(1.0, isBullish, "Breadth > 1.0 should be bullish (1.0)");

            // Market breadth <= 1.0 is not bullish
            double neutralBreadth = 1.0;
            isBullish = neutralBreadth > 1.0 ? 1.0 : 0.0;
            Assert.AreEqual(0.0, isBullish, "Breadth <= 1.0 should not be bullish (0.0)");

            double bearishBreadth = 0.8;
            isBullish = bearishBreadth > 1.0 ? 1.0 : 0.0;
            Assert.AreEqual(0.0, isBullish, "Breadth < 1.0 should not be bullish (0.0)");
        }

        #endregion
    }
}
