using System;
using System.Linq;
using Quantra.Enums;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Xunit;

namespace Quantra.Tests.Services
{
    public class MultiLegStrategyTests
    {
        [Fact]
        public void MultiLegStrategy_Validate_VerticalSpread_ValidStrategy_ReturnsTrue()
        {
            // Arrange
            var strategy = new MultiLegStrategy
            {
                Name = "Test Vertical Spread",
                StrategyType = MultiLegStrategyType.VerticalSpread,
                Legs = new System.Collections.Generic.List<ScheduledOrder>()
                {
                    new ScheduledOrder
                    {
                        Symbol = "AAPL",
                        OrderType = "BUY",
                        Quantity = 1,
                        StrikePrice = 180,
                        OptionType = "CALL",
                        IsOption = true
                    },
                    new ScheduledOrder
                    {
                        Symbol = "AAPL",
                        OrderType = "SELL",
                        Quantity = 1,
                        StrikePrice = 185,
                        OptionType = "CALL",
                        IsOption = true
                    }
                }
            };

            // Act
            bool isValid = strategy.Validate();

            // Assert
            Assert.True(isValid);
        }

        [Fact]
        public void MultiLegStrategy_Validate_VerticalSpread_InvalidSingleLeg_ReturnsFalse()
        {
            // Arrange
            var strategy = new MultiLegStrategy
            {
                Name = "Invalid Vertical Spread",
                StrategyType = MultiLegStrategyType.VerticalSpread,
                Legs = new System.Collections.Generic.List<ScheduledOrder>()
                {
                    new ScheduledOrder
                    {
                        Symbol = "AAPL",
                        OrderType = "BUY",
                        Quantity = 1
                    }
                }
            };

            // Act
            bool isValid = strategy.Validate();

            // Assert
            Assert.False(isValid);
        }

        [Fact]
        public void MultiLegStrategy_Validate_VerticalSpread_DifferentSymbols_ReturnsFalse()
        {
            // Arrange
            var strategy = new MultiLegStrategy
            {
                Name = "Invalid Symbol Vertical Spread",
                StrategyType = MultiLegStrategyType.VerticalSpread,
                Legs = new System.Collections.Generic.List<ScheduledOrder>()
                {
                    new ScheduledOrder
                    {
                        Symbol = "AAPL",
                        OrderType = "BUY",
                        Quantity = 1
                    },
                    new ScheduledOrder
                    {
                        Symbol = "MSFT",
                        OrderType = "SELL",
                        Quantity = 1
                    }
                }
            };

            // Act
            bool isValid = strategy.Validate();

            // Assert
            Assert.False(isValid);
        }

        [Fact]
        public void MultiLegStrategy_Validate_PairsTrade_SameSymbol_ReturnsFalse()
        {
            // Arrange
            var strategy = new MultiLegStrategy
            {
                Name = "Invalid Pairs Trade",
                StrategyType = MultiLegStrategyType.PairsTrade,
                Legs = new System.Collections.Generic.List<ScheduledOrder>()
                {
                    new ScheduledOrder
                    {
                        Symbol = "AAPL",
                        OrderType = "BUY",
                        Quantity = 100
                    },
                    new ScheduledOrder
                    {
                        Symbol = "AAPL",
                        OrderType = "SELL",
                        Quantity = 100
                    }
                }
            };

            // Act
            bool isValid = strategy.Validate();

            // Assert
            Assert.False(isValid);
        }

        [Fact]
        public void MultiLegStrategy_Validate_Straddle_Valid_ReturnsTrue()
        {
            // Arrange
            var strategy = new MultiLegStrategy
            {
                Name = "Valid Straddle",
                StrategyType = MultiLegStrategyType.Straddle,
                Legs = new System.Collections.Generic.List<ScheduledOrder>()
                {
                    new ScheduledOrder
                    {
                        Symbol = "AAPL",
                        OrderType = "BUY",
                        Quantity = 1,
                        StrikePrice = 180,
                        OptionType = "CALL",
                        IsOption = true
                    },
                    new ScheduledOrder
                    {
                        Symbol = "AAPL",
                        OrderType = "BUY",
                        Quantity = 1,
                        StrikePrice = 180,
                        OptionType = "PUT",
                        IsOption = true
                    }
                }
            };

            // Act
            bool isValid = strategy.Validate();

            // Assert
            Assert.True(isValid);
        }

        [Fact]
        public void MultiLegStrategy_CalculateNetCost_ReturnsSumOfLegCosts()
        {
            // Arrange
            var strategy = new MultiLegStrategy
            {
                Name = "Test Strategy",
                StrategyType = MultiLegStrategyType.Custom,
                Legs = new System.Collections.Generic.List<ScheduledOrder>()
                {
                    new ScheduledOrder
                    {
                        Symbol = "AAPL",
                        OrderType = "BUY",
                        Quantity = 10,
                        Price = 150.50
                    },
                    new ScheduledOrder
                    {
                        Symbol = "MSFT",
                        OrderType = "SELL",
                        Quantity = 5,
                        Price = 300.75
                    }
                }
            };

            // Expected: (10 * 150.50) - (5 * 300.75) = 1505 - 1503.75 = 1.25
            double expected = 1.25;

            // Act
            double netCost = strategy.CalculateNetCost();

            // Assert
            Assert.Equal(expected, netCost, 2); // Precise to 2 decimal places
        }

        [Fact]
        public void CreateVerticalSpread_WithValidParameters_ReturnsValidStrategy()
        {
            // Arrange
            var bot = new WebullTradingBot();

            // Act
            var strategy = bot.CreateVerticalSpread(
                "AAPL",      // symbol 
                1,           // quantity
                true,        // isBullish (bull call spread)
                180.0,       // lowerStrike
                185.0,       // upperStrike
                DateTime.Today.AddDays(30),  // expiration 
                1.25         // total price
            );

            // Assert
            Assert.NotNull(strategy);
            Assert.Equal(MultiLegStrategyType.VerticalSpread, strategy.StrategyType);
            Assert.Equal(2, strategy.Legs.Count);
            Assert.Equal("BUY", strategy.Legs[0].OrderType);
            Assert.Equal("SELL", strategy.Legs[1].OrderType);
            Assert.Equal(180.0, strategy.Legs[0].StrikePrice);
            Assert.Equal(185.0, strategy.Legs[1].StrikePrice);
        }

        [Fact]
        public void CreatePairsTrade_WithValidParameters_ReturnsValidStrategy()
        {
            // Arrange
            var bot = new WebullTradingBot();

            // Act
            var strategy = bot.CreatePairsTrade(
                "AAPL",      // longSymbol
                "MSFT",      // shortSymbol
                100,         // longQuantity
                50,          // shortQuantity
                0.85         // correlation
            );

            // Assert
            Assert.NotNull(strategy);
            Assert.Equal(MultiLegStrategyType.PairsTrade, strategy.StrategyType);
            Assert.Equal(2, strategy.Legs.Count);
            Assert.Equal("BUY", strategy.Legs[0].OrderType);
            Assert.Equal("SELL", strategy.Legs[1].OrderType);
            Assert.Equal("AAPL", strategy.Legs[0].Symbol);
            Assert.Equal("MSFT", strategy.Legs[1].Symbol);
            Assert.Equal(100, strategy.Legs[0].Quantity);
            Assert.Equal(50, strategy.Legs[1].Quantity);
            Assert.Contains("Correlation: 0.85", strategy.Notes);
        }
    }
}