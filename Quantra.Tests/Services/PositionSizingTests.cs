using System;
using Quantra.Enums;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Xunit;
using Quantra.DAL.Services;

namespace Quantra.Tests.Services
{
    public class PositionSizingTests
    {
        [Fact]
        public void CalculatePositionSizeByFixedRisk_ShouldReturnCorrectShares()
        {
            // Arrange
            var bot = new WebullTradingBot();
            string symbol = "AAPL";
            double price = 200.0;
            double stopLossPrice = 190.0;
            double riskPercentage = 0.01; // 1%
            double accountSize = 100000.0;
            
            // Expected calculation:
            // Risk per share = $200 - $190 = $10
            // Risk amount = $100,000 * 0.01 = $1,000
            // Shares = $1,000 / $10 = 100 shares
            int expectedShares = 100;
            
            // Act
            int shares = bot.CalculatePositionSizeByRisk(
                symbol, price, stopLossPrice, riskPercentage, accountSize);
            
            // Assert
            Assert.Equal(expectedShares, shares);
        }
        
        [Fact]
        public void CalculatePositionSize_UsingPercentageOfEquity_ShouldReturnCorrectShares()
        {
            // Arrange
            var bot = new WebullTradingBot();
            var parameters = new PositionSizingParameters
            {
                Symbol = "MSFT",
                Price = 400.0,
                StopLossPrice = 380.0,
                RiskPercentage = 0.05, // 5%
                AccountSize = 100000.0,
                Method = PositionSizingMethod.PercentageOfEquity
            };
            
            // Expected calculation:
            // Position amount = $100,000 * 0.05 = $5,000
            // Shares = $5,000 / $400 = 12 shares
            int expectedShares = 12;
            
            // Act
            int shares = bot.CalculatePositionSize(parameters);
            
            // Assert
            Assert.Equal(expectedShares, shares);
        }
        
        [Fact]
        public void CalculatePositionSize_UsingFixedAmount_ShouldReturnCorrectShares()
        {
            // Arrange
            var bot = new WebullTradingBot();
            var parameters = new PositionSizingParameters
            {
                Symbol = "TSLA",
                Price = 250.0,
                StopLossPrice = 240.0,
                AccountSize = 100000.0,
                Method = PositionSizingMethod.FixedAmount,
                FixedAmount = 10000.0 // $10,000 fixed amount
            };
            
            // Expected calculation:
            // Shares = $10,000 / $250 = 40 shares
            int expectedShares = 40;
            
            // Act
            int shares = bot.CalculatePositionSize(parameters);
            
            // Assert
            Assert.Equal(expectedShares, shares);
        }
        
        [Fact]
        public void CalculatePositionSize_WithConservativeRiskMode_ShouldReducePositionSize()
        {
            // Arrange
            var bot = new WebullTradingBot();
            
            // Normal risk mode parameters
            var normalParameters = new PositionSizingParameters
            {
                Symbol = "SPY",
                Price = 500.0,
                StopLossPrice = 490.0,
                RiskPercentage = 0.01, // 1%
                AccountSize = 100000.0,
                Method = PositionSizingMethod.FixedRisk,
                RiskMode = RiskMode.Normal
            };
            
            // Conservative risk mode parameters (same except for risk mode)
            var conservativeParameters = new PositionSizingParameters
            {
                Symbol = "SPY",
                Price = 500.0,
                StopLossPrice = 490.0,
                RiskPercentage = 0.01, // 1%
                AccountSize = 100000.0,
                Method = PositionSizingMethod.FixedRisk,
                RiskMode = RiskMode.Conservative
            };
            
            // Act
            int normalShares = bot.CalculatePositionSize(normalParameters);
            int conservativeShares = bot.CalculatePositionSize(conservativeParameters);
            
            // Assert
            // Conservative mode should use half the risk percentage -> half the shares
            Assert.True(conservativeShares < normalShares);
            Assert.Equal(normalShares / 2, conservativeShares);
        }
        
        [Fact]
        public void CalculatePositionSize_WithAggressiveRiskMode_ShouldIncreasePositionSize()
        {
            // Arrange
            var bot = new WebullTradingBot();
            
            // Normal risk mode parameters
            var normalParameters = new PositionSizingParameters
            {
                Symbol = "QQQ",
                Price = 400.0,
                StopLossPrice = 390.0,
                RiskPercentage = 0.01, // 1%
                AccountSize = 100000.0,
                Method = PositionSizingMethod.FixedRisk,
                RiskMode = RiskMode.Normal
            };
            
            // Aggressive risk mode parameters (same except for risk mode)
            var aggressiveParameters = new PositionSizingParameters
            {
                Symbol = "QQQ",
                Price = 400.0,
                StopLossPrice = 390.0,
                RiskPercentage = 0.01, // 1%
                AccountSize = 100000.0,
                Method = PositionSizingMethod.FixedRisk,
                RiskMode = RiskMode.Aggressive
            };
            
            // Act
            int normalShares = bot.CalculatePositionSize(normalParameters);
            int aggressiveShares = bot.CalculatePositionSize(aggressiveParameters);
            
            // Assert
            // Aggressive mode should use 1.5x the risk percentage -> 1.5x the shares
            Assert.True(aggressiveShares > normalShares);
            // Expected aggressive shares is 1.5x normal shares
            Assert.Equal((int)(normalShares * 1.5), aggressiveShares);
        }
        
        [Fact]
        public void CalculatePositionSize_UsingTierBased_ShouldAdjustBasedOnConfidence()
        {
            // Arrange
            var bot = new WebullTradingBot();
            
            // Base parameters with different confidence levels
            var lowConfidenceParams = new PositionSizingParameters
            {
                Symbol = "NVDA",
                Price = 800.0,
                StopLossPrice = 780.0,
                RiskPercentage = 0.01, // 1%
                AccountSize = 100000.0,
                Method = PositionSizingMethod.TierBased,
                Confidence = 0.55 // Low confidence
            };
            
            var highConfidenceParams = new PositionSizingParameters
            {
                Symbol = "NVDA",
                Price = 800.0,
                StopLossPrice = 780.0,
                RiskPercentage = 0.01, // 1%
                AccountSize = 100000.0,
                Method = PositionSizingMethod.TierBased,
                Confidence = 0.95 // High confidence
            };
            
            // Act
            int lowConfidenceShares = bot.CalculatePositionSize(lowConfidenceParams);
            int highConfidenceShares = bot.CalculatePositionSize(highConfidenceParams);
            
            // Assert
            // High confidence should get more shares than low confidence
            Assert.True(highConfidenceShares > lowConfidenceShares);
        }
        
        [Fact]
        public void CalculatePositionSize_WithInvalidParameters_ShouldReturnZero()
        {
            // Arrange
            var bot = new WebullTradingBot();
            
            // Invalid parameters (price is zero)
            var invalidParameters = new PositionSizingParameters
            {
                Symbol = "XYZ",
                Price = 0, // Invalid price
                StopLossPrice = 10.0,
                RiskPercentage = 0.01,
                AccountSize = 100000.0,
                Method = PositionSizingMethod.FixedRisk
            };
            
            // Act
            int shares = bot.CalculatePositionSize(invalidParameters);
            
            // Assert
            Assert.Equal(0, shares);
        }
        
        [Fact]
        public void CalculatePositionSize_WhenPositionExceedsMaxSize_ShouldLimitToMaxSize()
        {
            // Arrange
            var bot = new WebullTradingBot();
            
            // Parameters for a position that would exceed the max size percentage
            var parameters = new PositionSizingParameters
            {
                Symbol = "GME",
                Price = 20.0,
                StopLossPrice = 19.9, // Small stop loss distance -> large position
                RiskPercentage = 0.05, // 5%
                AccountSize = 100000.0,
                MaxPositionSizePercent = 0.10, // 10% max position
                Method = PositionSizingMethod.FixedRisk
            };
            
            // Expected max position value = $100,000 * 0.10 = $10,000
            // Max shares at $20 per share = $10,000 / $20 = 500 shares
            int expectedMaxShares = 500;
            
            // Act
            int shares = bot.CalculatePositionSize(parameters);
            
            // Assert
            Assert.Equal(expectedMaxShares, shares);
        }
        
        [Fact]
        public void CalculatePositionSize_UsingAdaptiveRisk_ShouldAdjustBasedOnMarketFactors()
        {
            // Arrange
            var bot = new WebullTradingBot();
            
            double accountSize = 100000.0;
            double price = 300.0;
            double stopLossPrice = 290.0;
            double basePositionPercentage = 0.01; // 1% base risk
            
            // Scenario 1: Normal Market Conditions
            var normalParams = new PositionSizingParameters
            {
                Symbol = "AAPL",
                Price = price,
                StopLossPrice = stopLossPrice,
                AccountSize = accountSize,
                BasePositionPercentage = basePositionPercentage,
                Method = PositionSizingMethod.AdaptiveRisk,
                MarketVolatilityFactor = 0.0,
                PerformanceFactor = 0.0,
                TrendStrengthFactor = 0.5
            };
            
            // Scenario 2: High Volatility, Recent Losses, Weak Trend (should reduce size)
            var highRiskParams = new PositionSizingParameters
            {
                Symbol = "AAPL",
                Price = price,
                StopLossPrice = stopLossPrice,
                AccountSize = accountSize,
                BasePositionPercentage = basePositionPercentage,
                Method = PositionSizingMethod.AdaptiveRisk,
                MarketVolatilityFactor = 0.8, // High volatility
                PerformanceFactor = -0.7, // Recent losses
                TrendStrengthFactor = 0.2 // Weak trend
            };
            
            // Scenario 3: Low Volatility, Recent Gains, Strong Trend (should increase size)
            var lowRiskParams = new PositionSizingParameters
            {
                Symbol = "AAPL",
                Price = price,
                StopLossPrice = stopLossPrice,
                AccountSize = accountSize,
                BasePositionPercentage = basePositionPercentage,
                Method = PositionSizingMethod.AdaptiveRisk,
                MarketVolatilityFactor = -0.6, // Low volatility
                PerformanceFactor = 0.5, // Recent gains
                TrendStrengthFactor = 0.9 // Strong trend
            };
            
            // Act
            int normalShares = bot.CalculatePositionSize(normalParams);
            int highRiskShares = bot.CalculatePositionSize(highRiskParams);
            int lowRiskShares = bot.CalculatePositionSize(lowRiskParams);
            
            // Assert
            // 1. High risk scenario should use fewer shares than normal
            Assert.True(highRiskShares < normalShares);
            
            // 2. Low risk scenario should use more shares than normal
            Assert.True(lowRiskShares > normalShares);
            
            // 3. High risk should be significantly less than low risk
            Assert.True(highRiskShares < lowRiskShares * 0.5); // Less than 50% of low risk position
        }
        
        [Fact]
        public void CalculatePositionSizeByAdaptiveRisk_PublicMethod_ShouldCalculateCorrectly()
        {
            // Arrange
            var bot = new WebullTradingBot();
            string symbol = "MSFT";
            double price = 350.0;
            double stopLossPrice = 340.0;
            double basePositionPercentage = 0.01; // 1% base
            double accountSize = 100000.0;
            
            // 1. Favorable market conditions - should increase size
            double volatilityFactorFavorable = -0.5; // Low volatility
            double performanceFactorFavorable = 0.3; // Good performance 
            double trendStrengthFactorFavorable = 0.8; // Strong trend
            
            // 2. Unfavorable market conditions - should reduce size
            double volatilityFactorUnfavorable = 0.5; // High volatility
            double performanceFactorUnfavorable = -0.3; // Poor performance
            double trendStrengthFactorUnfavorable = 0.3; // Weak trend
            
            // Act
            int favorableShares = bot.CalculatePositionSizeByAdaptiveRisk(
                symbol, price, stopLossPrice, basePositionPercentage, accountSize,
                volatilityFactorFavorable, performanceFactorFavorable, trendStrengthFactorFavorable);
                
            int unfavorableShares = bot.CalculatePositionSizeByAdaptiveRisk(
                symbol, price, stopLossPrice, basePositionPercentage, accountSize,
                volatilityFactorUnfavorable, performanceFactorUnfavorable, trendStrengthFactorUnfavorable);
            
            // Also test the default values
            int defaultShares = bot.CalculatePositionSizeByAdaptiveRisk(
                symbol, price, stopLossPrice, basePositionPercentage, accountSize);
                
            // Assert
            Assert.True(favorableShares > unfavorableShares);
            // Default should be between favorable and unfavorable
            Assert.True(defaultShares > unfavorableShares);
            Assert.True(defaultShares < favorableShares);
        }
    }
}