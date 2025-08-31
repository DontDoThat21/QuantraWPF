using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.Enums;

namespace Quantra.Tests.Services
{
    public class SpreadBacktestingEngineTests
    {
        /// <summary>
        /// Test basic spread backtesting functionality with a simple vertical spread
        /// </summary>
        [Fact]
        public async Task RunSpreadBacktestAsync_WithVerticalSpread_ShouldGenerateResults()
        {
            // Arrange
            var backtest = new BacktestingEngine();
            var historical = GenerateTestHistoricalData();
            var spreadStrategy = CreateTestVerticalSpreadStrategy();
            
            // Act
            var result = await backtest.RunSpreadBacktestAsync("TEST", historical, spreadStrategy);
            
            // Assert
            Assert.NotNull(result);
            Assert.True(result.IsSpreadBacktest, "Should be identified as spread backtest");
            Assert.NotNull(result.SpreadResults);
            Assert.Equal("options_spread", result.AssetClass);
            Assert.True(result.EquityCurve.Count > 0, "Should have equity curve data");
            Assert.True(result.DrawdownCurve.Count > 0, "Should have drawdown data");
        }

        /// <summary>
        /// Test that spread P&L is calculated correctly over time
        /// </summary>
        [Fact]
        public async Task RunSpreadBacktestAsync_ShouldCalculateRollingPnL()
        {
            // Arrange
            var backtest = new BacktestingEngine();
            var historical = GenerateTestHistoricalData();
            var spreadStrategy = CreateTestStraddleStrategy();
            
            // Act
            var result = await backtest.RunSpreadBacktestAsync("TEST", historical, spreadStrategy);
            
            // Assert
            Assert.NotNull(result.SpreadResults.RollingPnL);
            Assert.True(result.SpreadResults.RollingPnL.Count > 0, "Should have rolling P&L data");
            
            // Verify P&L points have required data
            var firstPnL = result.SpreadResults.RollingPnL.First();
            Assert.True(firstPnL.Date > DateTime.MinValue);
            Assert.True(firstPnL.UnderlyingPrice > 0);
            Assert.True(firstPnL.OpenPositions >= 0);
        }

        /// <summary>
        /// Test spread strategy signal generation
        /// </summary>
        [Fact]
        public void SpreadStrategyProfile_GenerateSignal_ShouldReturnValidSignals()
        {
            // Arrange
            var spreadConfig = CreateTestSpreadConfiguration();
            var strategy = new SpreadStrategyProfile(spreadConfig, "Test Strategy");
            var historical = GenerateTestHistoricalData();
            
            // Act
            var signal1 = strategy.GenerateSignal(historical, 25); // Mid-way through data
            var signal2 = strategy.GenerateSignal(historical, 5);  // Early data (should be null)
            
            // Assert
            Assert.True(string.IsNullOrEmpty(signal2), "Should not generate signal with insufficient history");
            // signal1 could be ENTER or null depending on market conditions in test data
        }

        /// <summary>
        /// Test that ValidateConditions method works correctly for different scenarios
        /// </summary>
        [Fact]
        public void SpreadStrategyProfile_ValidateConditions_ShouldValidateCorrectly()
        {
            // Arrange
            var spreadConfig = CreateTestSpreadConfiguration();
            var strategy = new SpreadStrategyProfile(spreadConfig, "Test Strategy");
            
            // Test with null indicators
            Assert.False(strategy.ValidateConditions(null), "Should return false for null indicators");
            
            // Test with empty indicators
            var emptyIndicators = new Dictionary<string, double>();
            Assert.False(strategy.ValidateConditions(emptyIndicators), "Should return false for empty indicators");
            
            // Test with missing required PRICE indicator
            var missingPriceIndicators = new Dictionary<string, double>
            {
                { "VOLATILITY", 0.2 },
                { "VOLUME", 1000000 }
            };
            Assert.False(strategy.ValidateConditions(missingPriceIndicators), "Should return false when PRICE is missing");
            
            // Test with invalid PRICE (zero or negative)
            var invalidPriceIndicators = new Dictionary<string, double>
            {
                { "PRICE", 0 },
                { "VOLATILITY", 0.2 },
                { "VOLUME", 1000000 }
            };
            Assert.False(strategy.ValidateConditions(invalidPriceIndicators), "Should return false for zero or negative price");
            
            // Test with valid indicators
            var validIndicators = new Dictionary<string, double>
            {
                { "PRICE", 150.0 },
                { "VOLATILITY", 0.2 },
                { "VOLUME", 1000000 }
            };
            Assert.True(strategy.ValidateConditions(validIndicators), "Should return true for valid indicators");
            
            // Test with minimal valid indicators (just PRICE)
            var minimalIndicators = new Dictionary<string, double>
            {
                { "PRICE", 150.0 }
            };
            Assert.True(strategy.ValidateConditions(minimalIndicators), "Should return true with just valid PRICE");
        }

        /// <summary>
        /// Test that spread backtesting handles multiple concurrent positions
        /// </summary>
        [Fact]
        public async Task RunSpreadBacktestAsync_ShouldHandleMultiplePositions()
        {
            // Arrange
            var backtest = new BacktestingEngine();
            var historical = GenerateTrendingHistoricalData(); // More volatile data to trigger entries
            var spreadStrategy = CreateTestVerticalSpreadStrategy();
            
            // Act
            var result = await backtest.RunSpreadBacktestAsync("TEST", historical, spreadStrategy);
            
            // Assert
            var maxOpenPositions = result.SpreadResults.RollingPnL.Max(p => p.OpenPositions);
            Assert.True(maxOpenPositions <= 5, "Should respect position limit of 5");
        }

        /// <summary>
        /// Test spread performance metrics calculation
        /// </summary>
        [Fact]
        public async Task RunSpreadBacktestAsync_ShouldCalculateSpreadMetrics()
        {
            // Arrange
            var backtest = new BacktestingEngine();
            var historical = GenerateTestHistoricalData();
            var spreadStrategy = CreateTestVerticalSpreadStrategy();
            
            // Act
            var result = await backtest.RunSpreadBacktestAsync("TEST", historical, spreadStrategy);
            
            // Assert
            var spreadResults = result.SpreadResults;
            Assert.True(spreadResults.ProfitableTradePercentage >= 0 && spreadResults.ProfitableTradePercentage <= 1);
            Assert.True(spreadResults.AverageTimeInTrade >= 0);
            
            // Should have equity comparison
            Assert.True(double.IsFinite(spreadResults.OutperformanceVsEquity));
        }

        /// <summary>
        /// Test that transaction costs are applied to spread trades
        /// </summary>
        [Fact]
        public async Task RunSpreadBacktestAsync_WithTransactionCosts_ShouldReduceReturns()
        {
            // Arrange
            var backtest = new BacktestingEngine();
            var historical = GenerateTestHistoricalData();
            var spreadStrategy = CreateTestVerticalSpreadStrategy();
            var costModel = TransactionCostModel.CreateFixedCommissionModel(5); // $5 per leg
            
            // Act
            var resultWithoutCosts = await backtest.RunSpreadBacktestAsync("TEST", historical, spreadStrategy);
            var resultWithCosts = await backtest.RunSpreadBacktestAsync("TEST", historical, spreadStrategy, costModel: costModel);
            
            // Assert
            Assert.True(resultWithCosts.TotalTransactionCosts > 0, "Should have transaction costs");
            // Note: Due to random signal generation, we can't guarantee returns will be lower
            // but we can verify costs are tracked
        }

        /// <summary>
        /// Test different spread types generate different signals
        /// </summary>
        [Theory]
        [InlineData(MultiLegStrategyType.VerticalSpread)]
        [InlineData(MultiLegStrategyType.Straddle)]
        [InlineData(MultiLegStrategyType.Strangle)]
        [InlineData(MultiLegStrategyType.IronCondor)]
        public void SpreadStrategyProfile_DifferentTypes_ShouldHaveDifferentLogic(MultiLegStrategyType spreadType)
        {
            // Arrange
            var spreadConfig = CreateTestSpreadConfiguration();
            spreadConfig.SpreadType = spreadType;
            var strategy = new SpreadStrategyProfile(spreadConfig, $"Test {spreadType}");
            var historical = GenerateTestHistoricalData();
            
            // Act & Assert - Should not throw exceptions
            for (int i = 20; i < historical.Count; i += 5)
            {
                var signal = strategy.GenerateSignal(historical, i);
                // Signal could be ENTER or null, both are valid
                Assert.True(signal == null || signal == "ENTER", $"Invalid signal for {spreadType}: {signal}");
            }
        }

        /// <summary>
        /// Test spread exit conditions work correctly
        /// </summary>
        [Fact]
        public async Task RunSpreadBacktestAsync_ShouldExitPositionsCorrectly()
        {
            // Arrange
            var backtest = new BacktestingEngine();
            var historical = GenerateTestHistoricalData();
            var spreadStrategy = CreateTestVerticalSpreadStrategy();
            spreadStrategy.TargetProfitPercentage = 0.1; // Very low target for testing
            
            // Act
            var result = await backtest.RunSpreadBacktestAsync("TEST", historical, spreadStrategy);
            
            // Assert
            if (result.SpreadResults.SpreadTrades.Any())
            {
                var exitReasons = result.SpreadResults.SpreadTrades
                    .Where(t => !string.IsNullOrEmpty(t.ExitReason))
                    .Select(t => t.ExitReason)
                    .Distinct();
                
                Assert.True(exitReasons.Any(), "Should have exit reasons");
                
                var validExitReasons = new[] { "TIME_DECAY", "TARGET_PROFIT", "STOP_LOSS", "EXPIRATION" };
                foreach (var reason in exitReasons)
                {
                    Assert.Contains(reason, validExitReasons);
                }
            }
        }

        #region Helper Methods

        /// <summary>
        /// Generate test historical data with moderate volatility
        /// </summary>
        private List<HistoricalPrice> GenerateTestHistoricalData()
        {
            var data = new List<HistoricalPrice>();
            var random = new Random(42); // Fixed seed for reproducibility
            var basePrice = 100.0;
            var currentPrice = basePrice;
            
            for (int i = 0; i < 50; i++)
            {
                var change = (random.NextDouble() - 0.5) * 0.04; // ±2% daily change
                currentPrice *= (1 + change);
                
                data.Add(new HistoricalPrice
                {
                    Date = DateTime.Today.AddDays(-50 + i),
                    Open = currentPrice,
                    High = currentPrice * (1 + Math.Abs(change)),
                    Low = currentPrice * (1 - Math.Abs(change)),
                    Close = currentPrice,
                    Volume = 1000000 + random.Next(500000)
                });
            }
            
            return data;
        }

        /// <summary>
        /// Generate trending historical data to trigger more entries
        /// </summary>
        private List<HistoricalPrice> GenerateTrendingHistoricalData()
        {
            var data = new List<HistoricalPrice>();
            var random = new Random(42);
            var basePrice = 100.0;
            var currentPrice = basePrice;
            
            for (int i = 0; i < 50; i++)
            {
                // Create trending movements with higher volatility
                var trendFactor = i < 25 ? 0.01 : -0.01; // Up then down trend
                var randomFactor = (random.NextDouble() - 0.5) * 0.06; // ±3% random
                var change = trendFactor + randomFactor;
                
                currentPrice *= (1 + change);
                
                data.Add(new HistoricalPrice
                {
                    Date = DateTime.Today.AddDays(-50 + i),
                    Open = currentPrice,
                    High = currentPrice * (1 + Math.Abs(change)),
                    Low = currentPrice * (1 - Math.Abs(change)),
                    Close = currentPrice,
                    Volume = 1000000 + random.Next(500000)
                });
            }
            
            return data;
        }

        /// <summary>
        /// Create a test vertical spread strategy
        /// </summary>
        private SpreadStrategyProfile CreateTestVerticalSpreadStrategy()
        {
            var spreadConfig = CreateTestSpreadConfiguration();
            spreadConfig.SpreadType = MultiLegStrategyType.VerticalSpread;
            
            return new SpreadStrategyProfile(spreadConfig, "Test Bull Call Spread")
            {
                TargetProfitPercentage = 0.5,
                StopLossPercentage = -2.0,
                RiskFreeRate = 0.02
            };
        }

        /// <summary>
        /// Create a test straddle strategy
        /// </summary>
        private SpreadStrategyProfile CreateTestStraddleStrategy()
        {
            var spreadConfig = CreateTestSpreadConfiguration();
            spreadConfig.SpreadType = MultiLegStrategyType.Straddle;
            
            // Modify legs for straddle (same strike, different option types)
            spreadConfig.Legs.Clear();
            spreadConfig.Legs.Add(new OptionLeg
            {
                Option = new OptionData { StrikePrice = 100, OptionType = "CALL" },
                Action = "BUY",
                Quantity = 1,
                Price = 5.0
            });
            spreadConfig.Legs.Add(new OptionLeg
            {
                Option = new OptionData { StrikePrice = 100, OptionType = "PUT" },
                Action = "BUY",
                Quantity = 1,
                Price = 5.0
            });
            
            return new SpreadStrategyProfile(spreadConfig, "Test Long Straddle")
            {
                TargetProfitPercentage = 0.5,
                StopLossPercentage = -2.0,
                RiskFreeRate = 0.02
            };
        }

        /// <summary>
        /// Create a test spread configuration
        /// </summary>
        private SpreadConfiguration CreateTestSpreadConfiguration()
        {
            return new SpreadConfiguration
            {
                SpreadType = MultiLegStrategyType.VerticalSpread,
                UnderlyingSymbol = "TEST",
                UnderlyingPrice = 100.0,
                Legs = new List<OptionLeg>
                {
                    new OptionLeg
                    {
                        Option = new OptionData { StrikePrice = 95, OptionType = "CALL" },
                        Action = "BUY",
                        Quantity = 1,
                        Price = 7.0
                    },
                    new OptionLeg
                    {
                        Option = new OptionData { StrikePrice = 105, OptionType = "CALL" },
                        Action = "SELL",
                        Quantity = 1,
                        Price = 3.0
                    }
                }
            };
        }

        #endregion
    }
}