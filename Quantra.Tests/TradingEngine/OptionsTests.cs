using System;
using Xunit;
using Quantra.DAL.TradingEngine.Options;
using Quantra.DAL.TradingEngine.Orders;

namespace Quantra.Tests.TradingEngine
{
    /// <summary>
    /// Unit tests for the Options Calculator (Black-Scholes pricing and Greeks)
    /// </summary>
    public class OptionsCalculatorTests
    {
        // Test parameters
        private const decimal UnderlyingPrice = 100m;
        private const decimal StrikePrice = 100m;
        private const double TimeToExpiry = 0.25; // 3 months
        private const double RiskFreeRate = 0.05;  // 5%
        private const double Volatility = 0.20;    // 20%

        [Fact]
        public void CalculatePrice_AtTheMoneyCall_ReturnsReasonableValue()
        {
            // Act
            var price = OptionsCalculator.CalculatePrice(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Assert - ATM call should be around 4-5% of underlying for these parameters
            Assert.True(price > 0);
            Assert.True(price < UnderlyingPrice * 0.15m); // Should be less than 15% of stock price
        }

        [Fact]
        public void CalculatePrice_AtTheMoneyPut_ReturnsReasonableValue()
        {
            // Act
            var price = OptionsCalculator.CalculatePrice(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, false);

            // Assert
            Assert.True(price > 0);
            Assert.True(price < UnderlyingPrice * 0.15m);
        }

        [Fact]
        public void CalculatePrice_DeepInTheMoneyCall_ReturnsNearlyIntrinsicValue()
        {
            // Arrange - Call with strike 80 when price is 100
            decimal strikePrice = 80m;

            // Act
            var price = OptionsCalculator.CalculatePrice(
                UnderlyingPrice, strikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Assert - Should be at least the intrinsic value (100 - 80 = 20)
            Assert.True(price >= 20m);
        }

        [Fact]
        public void CalculatePrice_DeepOutOfTheMoneyCall_ReturnsNearZero()
        {
            // Arrange - Call with strike 150 when price is 100
            decimal strikePrice = 150m;

            // Act
            var price = OptionsCalculator.CalculatePrice(
                UnderlyingPrice, strikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Assert - Should be very close to zero
            Assert.True(price < 1m);
        }

        [Fact]
        public void CalculatePrice_InvalidInputs_ReturnsZero()
        {
            // Act & Assert
            Assert.Equal(0m, OptionsCalculator.CalculatePrice(0, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, true));
            Assert.Equal(0m, OptionsCalculator.CalculatePrice(UnderlyingPrice, 0, TimeToExpiry, RiskFreeRate, Volatility, true));
            Assert.Equal(0m, OptionsCalculator.CalculatePrice(UnderlyingPrice, StrikePrice, 0, RiskFreeRate, Volatility, true));
            Assert.Equal(0m, OptionsCalculator.CalculatePrice(UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, 0, true));
        }

        [Fact]
        public void CalculateDelta_AtTheMoneyCall_ReturnsNearHalf()
        {
            // Act
            var delta = OptionsCalculator.CalculateDelta(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Assert - ATM call delta should be around 0.5
            Assert.True(delta > 0.4m && delta < 0.6m);
        }

        [Fact]
        public void CalculateDelta_AtTheMoneyPut_ReturnsNearNegativeHalf()
        {
            // Act
            var delta = OptionsCalculator.CalculateDelta(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, false);

            // Assert - ATM put delta should be around -0.5
            Assert.True(delta < -0.4m && delta > -0.6m);
        }

        [Fact]
        public void CalculateDelta_DeepInTheMoneyCall_ReturnsNearOne()
        {
            // Arrange
            decimal strikePrice = 50m;

            // Act
            var delta = OptionsCalculator.CalculateDelta(
                UnderlyingPrice, strikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Assert
            Assert.True(delta > 0.9m);
        }

        [Fact]
        public void CalculateDelta_DeepOutOfTheMoneyCall_ReturnsNearZero()
        {
            // Arrange
            decimal strikePrice = 150m;

            // Act
            var delta = OptionsCalculator.CalculateDelta(
                UnderlyingPrice, strikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Assert
            Assert.True(delta < 0.1m);
        }

        [Fact]
        public void CalculateGamma_AtTheMoney_ReturnsMaximumValue()
        {
            // Act - Gamma is highest ATM
            var gammaAtm = OptionsCalculator.CalculateGamma(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility);

            var gammaItm = OptionsCalculator.CalculateGamma(
                UnderlyingPrice, 80m, TimeToExpiry, RiskFreeRate, Volatility);

            var gammaOtm = OptionsCalculator.CalculateGamma(
                UnderlyingPrice, 120m, TimeToExpiry, RiskFreeRate, Volatility);

            // Assert - ATM gamma should be highest
            Assert.True(gammaAtm > 0);
            Assert.True(gammaAtm >= gammaItm);
            Assert.True(gammaAtm >= gammaOtm);
        }

        [Fact]
        public void CalculateTheta_Call_ReturnsNegativeValue()
        {
            // Act
            var theta = OptionsCalculator.CalculateTheta(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Assert - Long options have negative theta (time decay)
            Assert.True(theta < 0);
        }

        [Fact]
        public void CalculateTheta_Put_ReturnsNegativeValue()
        {
            // Act
            var theta = OptionsCalculator.CalculateTheta(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, false);

            // Assert
            Assert.True(theta < 0);
        }

        [Fact]
        public void CalculateVega_AtTheMoney_ReturnsMaximumValue()
        {
            // Act - Vega is highest ATM
            var vegaAtm = OptionsCalculator.CalculateVega(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility);

            var vegaItm = OptionsCalculator.CalculateVega(
                UnderlyingPrice, 80m, TimeToExpiry, RiskFreeRate, Volatility);

            var vegaOtm = OptionsCalculator.CalculateVega(
                UnderlyingPrice, 120m, TimeToExpiry, RiskFreeRate, Volatility);

            // Assert
            Assert.True(vegaAtm > 0);
            Assert.True(vegaAtm >= vegaItm);
            Assert.True(vegaAtm >= vegaOtm);
        }

        [Fact]
        public void CalculateRho_Call_ReturnsPositiveValue()
        {
            // Act
            var rho = OptionsCalculator.CalculateRho(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Assert - Call rho is positive (higher rates benefit calls)
            Assert.True(rho > 0);
        }

        [Fact]
        public void CalculateRho_Put_ReturnsNegativeValue()
        {
            // Act
            var rho = OptionsCalculator.CalculateRho(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, false);

            // Assert - Put rho is negative (higher rates hurt puts)
            Assert.True(rho < 0);
        }

        [Fact]
        public void CalculateAllGreeks_ReturnsAllValues()
        {
            // Act
            var greeks = OptionsCalculator.CalculateAllGreeks(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Assert
            Assert.True(greeks.Price > 0);
            Assert.NotEqual(0m, greeks.Delta);
            Assert.True(greeks.Gamma > 0);
            Assert.True(greeks.Theta < 0);
            Assert.True(greeks.Vega > 0);
            Assert.NotEqual(0m, greeks.Rho);
        }

        [Fact]
        public void CalculateImpliedVolatility_KnownPrice_ReturnsApproximateVolatility()
        {
            // Arrange - Calculate a price using known volatility, then reverse it
            var targetPrice = OptionsCalculator.CalculatePrice(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            // Act
            var impliedVol = OptionsCalculator.CalculateImpliedVolatility(
                targetPrice, UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, true);

            // Assert - Should be close to the original volatility
            Assert.True(Math.Abs(impliedVol - Volatility) < 0.01); // Within 1%
        }

        [Fact]
        public void PutCallParity_HoldsApproximately()
        {
            // Arrange
            var callPrice = OptionsCalculator.CalculatePrice(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, true);

            var putPrice = OptionsCalculator.CalculatePrice(
                UnderlyingPrice, StrikePrice, TimeToExpiry, RiskFreeRate, Volatility, false);

            // Put-Call Parity: C - P = S - K * e^(-rT)
            var leftSide = callPrice - putPrice;
            var rightSide = UnderlyingPrice - StrikePrice * (decimal)Math.Exp(-RiskFreeRate * TimeToExpiry);

            // Assert - Should be approximately equal
            Assert.True(Math.Abs(leftSide - rightSide) < 0.01m);
        }
    }

    /// <summary>
    /// Unit tests for the OptionsStrategy class
    /// </summary>
    public class OptionsStrategyTests
    {
        [Fact]
        public void CreateBullCallSpread_SetsPropertiesCorrectly()
        {
            // Act
            var strategy = OptionsStrategy.CreateBullCallSpread(
                "AAPL",
                longStrike: 175m,
                shortStrike: 180m,
                expiration: DateTime.Today.AddMonths(1),
                longPremium: 5.00m,
                shortPremium: 2.50m,
                contracts: 1);

            // Assert
            Assert.Equal("AAPL", strategy.UnderlyingSymbol);
            Assert.Equal(OptionStrategyType.BullCallSpread, strategy.StrategyType);
            Assert.Equal(2, strategy.Legs.Count);

            var longLeg = strategy.Legs[0];
            Assert.Equal(175m, longLeg.StrikePrice);
            Assert.Equal(OrderSide.Buy, longLeg.Side);
            Assert.Equal(OptionType.Call, longLeg.OptionType);

            var shortLeg = strategy.Legs[1];
            Assert.Equal(180m, shortLeg.StrikePrice);
            Assert.Equal(OrderSide.Sell, shortLeg.Side);
        }

        [Fact]
        public void CreateIronCondor_HasFourLegs()
        {
            // Act
            var strategy = OptionsStrategy.CreateIronCondor(
                "SPY",
                putLongStrike: 440m,
                putShortStrike: 445m,
                callShortStrike: 455m,
                callLongStrike: 460m,
                expiration: DateTime.Today.AddMonths(1),
                putLongPremium: 1.50m,
                putShortPremium: 2.50m,
                callShortPremium: 2.50m,
                callLongPremium: 1.50m);

            // Assert
            Assert.Equal(OptionStrategyType.IronCondor, strategy.StrategyType);
            Assert.Equal(4, strategy.Legs.Count);

            // Check all legs
            Assert.Contains(strategy.Legs, l => l.OptionType == OptionType.Put && l.Side == OrderSide.Buy);
            Assert.Contains(strategy.Legs, l => l.OptionType == OptionType.Put && l.Side == OrderSide.Sell);
            Assert.Contains(strategy.Legs, l => l.OptionType == OptionType.Call && l.Side == OrderSide.Sell);
            Assert.Contains(strategy.Legs, l => l.OptionType == OptionType.Call && l.Side == OrderSide.Buy);
        }

        [Fact]
        public void CreateLongStraddle_HasTwoLegsAtSameStrike()
        {
            // Act
            var strategy = OptionsStrategy.CreateLongStraddle(
                "AAPL",
                strike: 180m,
                expiration: DateTime.Today.AddMonths(1),
                callPremium: 5.00m,
                putPremium: 4.50m);

            // Assert
            Assert.Equal(OptionStrategyType.LongStraddle, strategy.StrategyType);
            Assert.Equal(2, strategy.Legs.Count);
            Assert.All(strategy.Legs, l => Assert.Equal(180m, l.StrikePrice));
            Assert.All(strategy.Legs, l => Assert.Equal(OrderSide.Buy, l.Side));
        }

        [Fact]
        public void NetPremium_DebitSpread_ReturnsPositiveValue()
        {
            // Arrange
            var strategy = OptionsStrategy.CreateBullCallSpread(
                "AAPL",
                longStrike: 175m,
                shortStrike: 180m,
                expiration: DateTime.Today.AddMonths(1),
                longPremium: 5.00m,
                shortPremium: 2.50m);

            // Act
            var netPremium = strategy.NetPremium;

            // Assert - Debit spread costs money (positive premium)
            Assert.Equal(250m, netPremium); // (5.00 - 2.50) * 100 = 250
        }

        [Fact]
        public void NetPremium_CreditSpread_ReturnsNegativeValue()
        {
            // Arrange - Iron Condor is typically a credit spread
            var strategy = OptionsStrategy.CreateIronCondor(
                "SPY",
                putLongStrike: 440m,
                putShortStrike: 445m,
                callShortStrike: 455m,
                callLongStrike: 460m,
                expiration: DateTime.Today.AddMonths(1),
                putLongPremium: 1.00m,
                putShortPremium: 2.00m,
                callShortPremium: 2.00m,
                callLongPremium: 1.00m);

            // Act
            var netPremium = strategy.NetPremium;

            // Assert - Credit spread receives money (negative premium)
            Assert.True(netPremium < 0);
        }

        [Fact]
        public void NetDelta_NeutralStrategy_ReturnsNearZero()
        {
            // Arrange - Long straddle at ATM should be roughly delta neutral
            var strategy = OptionsStrategy.CreateLongStraddle(
                "AAPL",
                strike: 180m,
                expiration: DateTime.Today.AddMonths(1),
                callPremium: 5.00m,
                putPremium: 4.50m);

            // Simulate deltas
            strategy.Legs[0].Delta = 0.5m;  // Call
            strategy.Legs[1].Delta = -0.5m; // Put

            // Act
            var netDelta = strategy.NetDelta;

            // Assert - Should be near zero when ATM
            Assert.Equal(0m, netDelta);
        }

        [Fact]
        public void NetTheta_LongOptions_ReturnsNegative()
        {
            // Arrange
            var strategy = OptionsStrategy.CreateLongStraddle(
                "AAPL",
                strike: 180m,
                expiration: DateTime.Today.AddMonths(1),
                callPremium: 5.00m,
                putPremium: 4.50m);

            // Simulate thetas
            strategy.Legs[0].Theta = -0.05m;
            strategy.Legs[1].Theta = -0.04m;

            // Act
            var netTheta = strategy.NetTheta;

            // Assert - Long options have negative theta
            Assert.True(netTheta < 0);
        }

        [Fact]
        public void UnrealizedPnL_PriceIncrease_ReturnsPositiveForLong()
        {
            // Arrange
            var leg = new OptionsLeg
            {
                UnderlyingSymbol = "AAPL",
                OptionType = OptionType.Call,
                StrikePrice = 180m,
                Side = OrderSide.Buy,
                Quantity = 1,
                EntryPrice = 5.00m,
                CurrentPrice = 7.00m
            };

            // Act
            var pnl = leg.UnrealizedPnL;

            // Assert
            Assert.Equal(200m, pnl); // (7 - 5) * 1 * 100 = 200
        }

        [Fact]
        public void UnrealizedPnL_PriceDecrease_ReturnsNegativeForLong()
        {
            // Arrange
            var leg = new OptionsLeg
            {
                UnderlyingSymbol = "AAPL",
                OptionType = OptionType.Call,
                StrikePrice = 180m,
                Side = OrderSide.Buy,
                Quantity = 1,
                EntryPrice = 5.00m,
                CurrentPrice = 3.00m
            };

            // Act
            var pnl = leg.UnrealizedPnL;

            // Assert
            Assert.Equal(-200m, pnl); // (3 - 5) * 1 * 100 = -200
        }

        [Fact]
        public void UnrealizedPnL_ShortPosition_InvertsSign()
        {
            // Arrange
            var leg = new OptionsLeg
            {
                UnderlyingSymbol = "AAPL",
                OptionType = OptionType.Call,
                StrikePrice = 180m,
                Side = OrderSide.Sell,
                Quantity = 1,
                EntryPrice = 5.00m,
                CurrentPrice = 3.00m
            };

            // Act
            var pnl = leg.UnrealizedPnL;

            // Assert - Short gains when price drops
            Assert.Equal(200m, pnl); // -(3 - 5) * 1 * 100 = 200
        }

        [Fact]
        public void DaysToExpiration_CalculatesCorrectly()
        {
            // Arrange
            var leg = new OptionsLeg
            {
                Expiration = DateTime.Today.AddDays(30)
            };

            // Act & Assert
            Assert.Equal(30, leg.DaysToExpiration);
        }

        [Fact]
        public void OptionSymbol_FormatsCorrectly()
        {
            // Arrange
            var leg = new OptionsLeg
            {
                UnderlyingSymbol = "AAPL",
                OptionType = OptionType.Call,
                StrikePrice = 180m,
                Expiration = new DateTime(2024, 3, 15)
            };

            // Act
            var symbol = leg.OptionSymbol;

            // Assert
            Assert.Contains("AAPL", symbol);
            Assert.Contains("240315", symbol);
            Assert.Contains("C", symbol);
        }

        [Fact]
        public void HasExpiredLegs_ReturnsTrueForExpiredOption()
        {
            // Arrange
            var strategy = new OptionsStrategy
            {
                Legs = new System.Collections.Generic.List<OptionsLeg>
                {
                    new OptionsLeg { Expiration = DateTime.Today.AddDays(-1) },
                    new OptionsLeg { Expiration = DateTime.Today.AddDays(30) }
                }
            };

            // Act & Assert
            Assert.True(strategy.HasExpiredLegs);
        }

        [Fact]
        public void NearestExpiration_ReturnsEarliestDate()
        {
            // Arrange
            var strategy = new OptionsStrategy
            {
                Legs = new System.Collections.Generic.List<OptionsLeg>
                {
                    new OptionsLeg { Expiration = DateTime.Today.AddDays(60) },
                    new OptionsLeg { Expiration = DateTime.Today.AddDays(30) },
                    new OptionsLeg { Expiration = DateTime.Today.AddDays(45) }
                }
            };

            // Act
            var nearest = strategy.NearestExpiration;

            // Assert
            Assert.Equal(DateTime.Today.AddDays(30), nearest);
        }
    }
}
