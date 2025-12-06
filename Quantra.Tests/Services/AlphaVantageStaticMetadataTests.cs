using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Services;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Unit tests for static metadata feature methods in AlphaVantageService.
    /// Tests sector code, market cap category, and exchange code mappings for TFT model.
    /// These tests validate the numeric code mappings used for ML/TFT model static features.
    /// </summary>
    [TestClass]
    public class AlphaVantageStaticMetadataTests
    {
        private AlphaVantageService _alphaVantageService;

        [TestInitialize]
        public void Setup()
        {
            // Initialize the database to ensure user settings service works
            DatabaseMonolith.Initialize();
            
            // Create service instance with real dependencies (using DatabaseMonolith for settings)
            var userSettingsService = new UserSettingsService();
            var loggingService = new LoggingService();
            
            _alphaVantageService = new AlphaVantageService(userSettingsService, loggingService);
        }

        #region GetSectorCode Tests

        [TestMethod]
        public void GetSectorCode_Technology_ReturnsZero()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Technology");

            // Assert
            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void GetSectorCode_InformationTechnology_ReturnsZero()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Information Technology");

            // Assert
            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void GetSectorCode_Healthcare_ReturnsOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Healthcare");

            // Assert
            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void GetSectorCode_HealthCareWithSpace_ReturnsOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Health Care");

            // Assert
            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void GetSectorCode_FinancialServices_ReturnsTwo()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Financial Services");

            // Assert
            Assert.AreEqual(2, result);
        }

        [TestMethod]
        public void GetSectorCode_Financials_ReturnsTwo()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Financials");

            // Assert
            Assert.AreEqual(2, result);
        }

        [TestMethod]
        public void GetSectorCode_ConsumerCyclical_ReturnsThree()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Consumer Cyclical");

            // Assert
            Assert.AreEqual(3, result);
        }

        [TestMethod]
        public void GetSectorCode_ConsumerDefensive_ReturnsFour()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Consumer Defensive");

            // Assert
            Assert.AreEqual(4, result);
        }

        [TestMethod]
        public void GetSectorCode_Industrials_ReturnsFive()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Industrials");

            // Assert
            Assert.AreEqual(5, result);
        }

        [TestMethod]
        public void GetSectorCode_Energy_ReturnsSix()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Energy");

            // Assert
            Assert.AreEqual(6, result);
        }

        [TestMethod]
        public void GetSectorCode_BasicMaterials_ReturnsSeven()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Basic Materials");

            // Assert
            Assert.AreEqual(7, result);
        }

        [TestMethod]
        public void GetSectorCode_RealEstate_ReturnsEight()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Real Estate");

            // Assert
            Assert.AreEqual(8, result);
        }

        [TestMethod]
        public void GetSectorCode_Utilities_ReturnsNine()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Utilities");

            // Assert
            Assert.AreEqual(9, result);
        }

        [TestMethod]
        public void GetSectorCode_CommunicationServices_ReturnsTen()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Communication Services");

            // Assert
            Assert.AreEqual(10, result);
        }

        [TestMethod]
        public void GetSectorCode_UnknownSector_ReturnsNegativeOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode("Unknown Sector");

            // Assert
            Assert.AreEqual(-1, result);
        }

        [TestMethod]
        public void GetSectorCode_NullSector_ReturnsNegativeOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode(null);

            // Assert
            Assert.AreEqual(-1, result);
        }

        [TestMethod]
        public void GetSectorCode_EmptySector_ReturnsNegativeOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetSectorCode(string.Empty);

            // Assert
            Assert.AreEqual(-1, result);
        }

        [TestMethod]
        public void GetSectorCode_CaseInsensitive_Technology()
        {
            // Arrange & Act
            int resultLower = _alphaVantageService.GetSectorCode("technology");
            int resultUpper = _alphaVantageService.GetSectorCode("TECHNOLOGY");
            int resultMixed = _alphaVantageService.GetSectorCode("TeChnOLogY");

            // Assert
            Assert.AreEqual(0, resultLower);
            Assert.AreEqual(0, resultUpper);
            Assert.AreEqual(0, resultMixed);
        }

        #endregion

        #region GetMarketCapCategory Tests

        [TestMethod]
        public void GetMarketCapCategory_SmallCap_ReturnsZero()
        {
            // Small-cap: < $2 billion
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(1_500_000_000m); // $1.5B

            // Assert
            Assert.AreEqual(0, result, "Small-cap should return 0");
        }

        [TestMethod]
        public void GetMarketCapCategory_MidCap_ReturnsOne()
        {
            // Mid-cap: $2B - $10B
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(5_000_000_000m); // $5B

            // Assert
            Assert.AreEqual(1, result, "Mid-cap should return 1");
        }

        [TestMethod]
        public void GetMarketCapCategory_LargeCap_ReturnsTwo()
        {
            // Large-cap: $10B - $200B
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(50_000_000_000m); // $50B

            // Assert
            Assert.AreEqual(2, result, "Large-cap should return 2");
        }

        [TestMethod]
        public void GetMarketCapCategory_MegaCap_ReturnsThree()
        {
            // Mega-cap: > $200B
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(500_000_000_000m); // $500B

            // Assert
            Assert.AreEqual(3, result, "Mega-cap should return 3");
        }

        [TestMethod]
        public void GetMarketCapCategory_BoundarySmallToMid_ReturnsOne()
        {
            // At exactly $2B threshold
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(2_000_000_000m); // $2B

            // Assert
            Assert.AreEqual(1, result, "At $2B boundary should return Mid-cap (1)");
        }

        [TestMethod]
        public void GetMarketCapCategory_BoundaryMidToLarge_ReturnsTwo()
        {
            // At exactly $10B threshold
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(10_000_000_000m); // $10B

            // Assert
            Assert.AreEqual(2, result, "At $10B boundary should return Large-cap (2)");
        }

        [TestMethod]
        public void GetMarketCapCategory_BoundaryLargeToMega_ReturnsThree()
        {
            // At exactly $200B threshold
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(200_000_000_000m); // $200B

            // Assert
            Assert.AreEqual(3, result, "At $200B boundary should return Mega-cap (3)");
        }

        [TestMethod]
        public void GetMarketCapCategory_NullValue_ReturnsNegativeOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory((decimal?)null);

            // Assert
            Assert.AreEqual(-1, result, "Null market cap should return -1");
        }

        [TestMethod]
        public void GetMarketCapCategory_ZeroValue_ReturnsNegativeOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(0m);

            // Assert
            Assert.AreEqual(-1, result, "Zero market cap should return -1");
        }

        [TestMethod]
        public void GetMarketCapCategory_NegativeValue_ReturnsNegativeOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(-1_000_000_000m);

            // Assert
            Assert.AreEqual(-1, result, "Negative market cap should return -1");
        }

        [TestMethod]
        public void GetMarketCapCategory_LongOverload_SmallCap_ReturnsZero()
        {
            // Test the long overload
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(1_500_000_000L); // $1.5B

            // Assert
            Assert.AreEqual(0, result, "Small-cap (long) should return 0");
        }

        [TestMethod]
        public void GetMarketCapCategory_LongOverload_MegaCap_ReturnsThree()
        {
            // Test the long overload with mega-cap value
            // Arrange & Act
            int result = _alphaVantageService.GetMarketCapCategory(300_000_000_000L); // $300B

            // Assert
            Assert.AreEqual(3, result, "Mega-cap (long) should return 3");
        }

        #endregion

        #region GetExchangeCode Tests

        [TestMethod]
        public void GetExchangeCode_NYSE_ReturnsZero()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("NYSE");

            // Assert
            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void GetExchangeCode_NewYorkStockExchange_ReturnsZero()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("New York Stock Exchange");

            // Assert
            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void GetExchangeCode_NASDAQ_ReturnsOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("NASDAQ");

            // Assert
            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void GetExchangeCode_NASDAQGlobalSelect_ReturnsOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("NASDAQ Global Select");

            // Assert
            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void GetExchangeCode_AMEX_ReturnsTwo()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("AMEX");

            // Assert
            Assert.AreEqual(2, result);
        }

        [TestMethod]
        public void GetExchangeCode_NYSEAmerican_ReturnsTwo()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("NYSE American");

            // Assert
            Assert.AreEqual(2, result);
        }

        [TestMethod]
        public void GetExchangeCode_BATS_ReturnsThree()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("BATS");

            // Assert
            Assert.AreEqual(3, result, "BATS should be categorized as Other (3)");
        }

        [TestMethod]
        public void GetExchangeCode_UnknownExchange_ReturnsThree()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("London Stock Exchange");

            // Assert
            Assert.AreEqual(3, result, "Unknown exchanges should be categorized as Other (3)");
        }

        [TestMethod]
        public void GetExchangeCode_NullExchange_ReturnsNegativeOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode(null);

            // Assert
            Assert.AreEqual(-1, result, "Null exchange should return -1");
        }

        [TestMethod]
        public void GetExchangeCode_EmptyExchange_ReturnsNegativeOne()
        {
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode(string.Empty);

            // Assert
            Assert.AreEqual(-1, result, "Empty exchange should return -1");
        }

        [TestMethod]
        public void GetExchangeCode_CaseInsensitive_NYSE()
        {
            // Arrange & Act
            int resultLower = _alphaVantageService.GetExchangeCode("nyse");
            int resultUpper = _alphaVantageService.GetExchangeCode("NYSE");
            int resultMixed = _alphaVantageService.GetExchangeCode("NySe");

            // Assert
            Assert.AreEqual(0, resultLower);
            Assert.AreEqual(0, resultUpper);
            Assert.AreEqual(0, resultMixed);
        }

        [TestMethod]
        public void GetExchangeCode_PartialMatch_NYSEInName()
        {
            // Test fallback partial matching for NYSE
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("NYSE Composite");

            // Assert
            Assert.AreEqual(0, result, "Exchange containing NYSE should return 0");
        }

        [TestMethod]
        public void GetExchangeCode_PartialMatch_NASDAQInName()
        {
            // Test fallback partial matching for NASDAQ
            // Arrange & Act
            int result = _alphaVantageService.GetExchangeCode("NASDAQ Composite");

            // Assert
            Assert.AreEqual(1, result, "Exchange containing NASDAQ should return 1");
        }

        #endregion
    }
}
