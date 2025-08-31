using NUnit.Framework;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using System;
using static NUnit.Framework.Assert;

namespace Quantra.Tests.Services
{
    [TestFixture]
    public class GreekCalculationEngineTests
    {
        private GreekCalculationEngine _greekEngine;
        private Position _testPosition;
        private MarketConditions _testMarket;
        
        [SetUp]
        public void Setup()
        {
            _greekEngine = new GreekCalculationEngine();
            
            // Standard test position: At-the-money call option
            _testPosition = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 150.0,
                TimeToExpiration = 30.0 / 365.0, // 30 days
                RiskFreeRate = 0.05, // 5%
                Volatility = 0.25, // 25%
                IsCall = true,
                Quantity = 1,
                OptionPrice = 5.0
            };
            
            _testMarket = new MarketConditions
            {
                InterestRate = 0.05,
                VolatilityIndex = 20.0,
                MarketTrend = 0.0,
                EconomicGrowth = 0.0
            };
        }
        
        [Test]
        public void CalculateTheta_ValidCallOption_ReturnsNegativeTheta()
        {
            // Act
            double theta = _greekEngine.CalculateTheta(_testPosition, _testMarket);
            // Assert
            Assert.That(theta, Is.LessThan(0), "Call option Theta should be negative (time decay)");
            Assert.That(Math.Abs(theta), Is.GreaterThan(0), "Theta should not be zero for valid option");
        }
        [Test]
        public void CalculateTheta_ValidPutOption_ReturnsNegativeTheta()
        {
            // Arrange
            _testPosition.IsCall = false;
            // Act
            double theta = _greekEngine.CalculateTheta(_testPosition, _testMarket);
            // Assert
            Assert.That(theta, Is.LessThan(0), "Put option Theta should be negative (time decay)");
            Assert.That(Math.Abs(theta), Is.GreaterThan(0), "Theta should not be zero for valid option");
        }
        [Test]
        public void CalculateTheta_ShortPosition_ReturnsPositiveTheta()
        {
            // Arrange
            _testPosition.Quantity = -1; // Short position
            // Act
            double theta = _greekEngine.CalculateTheta(_testPosition, _testMarket);
            // Assert
            Assert.That(theta, Is.GreaterThan(0), "Short option position should have positive Theta (benefits from time decay)");
        }
        [Test]
        public void CalculateTheta_ZeroTimeToExpiration_ReturnsZero()
        {
            // Arrange
            _testPosition.TimeToExpiration = 0.0;
            // Act
            double theta = _greekEngine.CalculateTheta(_testPosition, _testMarket);
            // Assert
            Assert.That(theta, Is.EqualTo(0.0), "Theta should be zero when time to expiration is zero");
        }
        [Test]
        public void CalculateTheta_NullPosition_ReturnsZero()
        {
            // Act
            double theta = _greekEngine.CalculateTheta(null, _testMarket);
            // Assert
            Assert.That(theta, Is.EqualTo(0.0), "Theta should be zero for null position");
        }
        [Test]
        public void CalculateTheta_HigherVolatility_ReturnsHigherAbsoluteTheta()
        {
            // Arrange
            var lowVolPosition = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 150.0,
                TimeToExpiration = 30.0 / 365.0,
                RiskFreeRate = 0.05,
                Volatility = 0.15, // 15% volatility
                IsCall = true,
                Quantity = 1
            };
            var highVolPosition = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 150.0,
                TimeToExpiration = 30.0 / 365.0,
                RiskFreeRate = 0.05,
                Volatility = 0.35, // 35% volatility
                IsCall = true,
                Quantity = 1
            };
            // Act
            double lowVolTheta = _greekEngine.CalculateTheta(lowVolPosition, _testMarket);
            double highVolTheta = _greekEngine.CalculateTheta(highVolPosition, _testMarket);
            // Assert
            Assert.That(Math.Abs(highVolTheta), Is.GreaterThan(Math.Abs(lowVolTheta)),
                "Higher volatility should result in higher absolute Theta value");
        }
        [Test]
        public void CalculateTheta_LongerTimeToExpiration_ReturnsLowerAbsoluteTheta()
        {
            // Arrange
            var shortTimePosition = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 150.0,
                TimeToExpiration = 7.0 / 365.0, // 1 week
                RiskFreeRate = 0.05,
                Volatility = 0.25,
                IsCall = true,
                Quantity = 1
            };
            var longTimePosition = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 150.0,
                TimeToExpiration = 90.0 / 365.0, // 3 months
                RiskFreeRate = 0.05,
                Volatility = 0.25,
                IsCall = true,
                Quantity = 1
            };
            // Act
            double shortTimeTheta = _greekEngine.CalculateTheta(shortTimePosition, _testMarket);
            double longTimeTheta = _greekEngine.CalculateTheta(longTimePosition, _testMarket);
            // Assert
            Assert.That(Math.Abs(shortTimeTheta), Is.GreaterThan(Math.Abs(longTimeTheta)),
                "Shorter time to expiration should result in higher absolute Theta value (accelerating time decay)");
        }
        [Test]
        public void CalculateGreeks_ValidPosition_IncludesTheta()
        {
            // Act
            var greeks = _greekEngine.CalculateGreeks(_testPosition, _testMarket);
            // Assert
            Assert.That(greeks, Is.Not.Null, "Greek metrics should not be null");
            Assert.That(greeks.Theta, Is.LessThan(0), "Theta should be negative for long call position");
            Assert.That(greeks.CalculatedAt, Is.LessThanOrEqualTo(DateTime.Now), "Calculated timestamp should be valid");
        }
        [Test]
        public void CalculateGreeks_NullPosition_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                _greekEngine.CalculateGreeks(null, _testMarket));
        }
        [Test]
        public void CalculateGreeks_NullMarket_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                _greekEngine.CalculateGreeks(_testPosition, null));
        }
        [Test]
        public void CalculateTheta_AtTheMoneyOptions_ReturnsReasonableTheta()
        {
            // Arrange - At-the-money option with 30 days to expiration
            _testPosition.UnderlyingPrice = 100.0;
            _testPosition.StrikePrice = 100.0;
            _testPosition.TimeToExpiration = 30.0 / 365.0;
            _testPosition.Volatility = 0.20;
            _testPosition.RiskFreeRate = 0.05;
            // Act
            double theta = _greekEngine.CalculateTheta(_testPosition, _testMarket);
            // Assert
            Assert.That(theta, Is.LessThan(0), "Call option theta should be negative");
            Assert.That(Math.Abs(theta), Is.GreaterThan(0.001), "Theta should be meaningful for 30-day option");
            Assert.That(Math.Abs(theta), Is.LessThan(1.0), "Theta should not be extreme for reasonable parameters");
        }
        [Test]
        public void CalculateTheta_DeepInTheMoneyCall_ReturnsDifferentThetaThanATM()
        {
            // Arrange - Deep ITM call
            var itmPosition = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 120.0, // Deep ITM
                TimeToExpiration = 30.0 / 365.0,
                RiskFreeRate = 0.05,
                Volatility = 0.25,
                IsCall = true,
                Quantity = 1
            };
            var atmPosition = new Position
            {
                Symbol = "AAPL",
                UnderlyingPrice = 150.0,
                StrikePrice = 150.0, // ATM
                TimeToExpiration = 30.0 / 365.0,
                RiskFreeRate = 0.05,
                Volatility = 0.25,
                IsCall = true,
                Quantity = 1
            };
            // Act
            double itmTheta = _greekEngine.CalculateTheta(itmPosition, _testMarket);
            double atmTheta = _greekEngine.CalculateTheta(atmPosition, _testMarket);
            // Assert
            Assert.That(itmTheta, Is.Not.EqualTo(atmTheta), "ITM and ATM options should have different theta values");
            Assert.That(itmTheta, Is.LessThan(0), "ITM call theta should be negative");
            Assert.That(atmTheta, Is.LessThan(0), "ATM call theta should be negative");
        }
    }
}