//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Threading.Tasks;
//using Xunit;
//using Quantra.DAL.Services.Interfaces;
//using Quantra.Models;

//namespace Quantra.Tests
//{
//    public class BacktestingEngineTests
//    {
//        /// <summary>
//        /// Test that advanced performance metrics are calculated correctly
//        /// </summary>
//        [Fact]
//        public async Task CalculateAdvancedMetrics_ShouldCalculateCorrectly()
//        {
//            // Arrange
//            var backtest = new BacktestingEngine();
//            var strategy = new TestStrategy();
//            var historical = GenerateTestHistoricalData();

//            // Act
//            var result = await backtest.RunBacktestAsync("TEST", historical, strategy);

//            // Assert
//            Assert.NotEqual(0, result.SharpeRatio);
//            Assert.NotEqual(0, result.SortinoRatio);
//            Assert.NotEqual(0, result.CalmarRatio);
//            Assert.NotEqual(0, result.ProfitFactor);
//            Assert.NotEqual(0, result.InformationRatio);

//            // Verify metrics have reasonable values
//            Assert.True(result.SharpeRatio >= -10 && result.SharpeRatio <= 10);
//            Assert.True(result.SortinoRatio >= -10 && result.SortinoRatio <= 10);
//            Assert.True(result.CalmarRatio >= 0);
//            Assert.True(result.ProfitFactor >= 0);
//            Assert.True(result.InformationRatio >= -10 && result.InformationRatio <= 10);
//        }

//        /// <summary>
//        /// Test transaction costs using a fixed commission model
//        /// </summary>
//        [Fact]
//        public async Task FixedCommissions_ShouldReduceReturns()
//        {
//            // Arrange
//            var backtest = new BacktestingEngine();
//            var strategy = new TestStrategy();
//            var historical = GenerateTestHistoricalData();

//            // Set up fixed commission model - $10 per trade
//            var costModel = TransactionCostModel.CreateFixedCommissionModel(10);

//            // Act
//            var resultWithoutCosts = await backtest.RunBacktestAsync("TEST", historical, strategy);
//            var resultWithCosts = await backtest.RunBacktestAsync("TEST", historical, strategy, costModel: costModel);

//            // Assert
//            Assert.True(resultWithCosts.TotalReturn < resultWithoutCosts.TotalReturn, 
//                "Return with costs should be less than return without costs");
//            Assert.True(resultWithCosts.TotalTransactionCosts > 0, 
//                "Total transaction costs should be greater than zero");

//            // Simple estimation of costs: $10 per trade entry and exit
//            double estimatedCosts = resultWithCosts.Trades.Count * 10 * 2; // Entry and exit
//            Assert.True(Math.Abs(resultWithCosts.TotalTransactionCosts - estimatedCosts) < 0.01,
//                "Total costs should match expected costs");
//        }

//        /// <summary>
//        /// Test transaction costs with percentage-based commission model
//        /// </summary>
//        [Fact]
//        public async Task PercentageCommissions_ShouldReduceReturns()
//        {
//            // Arrange
//            var backtest = new BacktestingEngine();
//            var strategy = new TestStrategy();
//            var historical = GenerateTestHistoricalData();

//            // Set up percentage commission model - 0.5% per trade
//            var costModel = TransactionCostModel.CreatePercentageCommissionModel(0.005);

//            // Act
//            var resultWithoutCosts = await backtest.RunBacktestAsync("TEST", historical, strategy);
//            var resultWithCosts = await backtest.RunBacktestAsync("TEST", historical, strategy, costModel: costModel);

//            // Assert
//            Assert.True(resultWithCosts.TotalReturn < resultWithoutCosts.TotalReturn, 
//                "Return with costs should be less than return without costs");
//            Assert.True(resultWithCosts.TotalTransactionCosts > 0, 
//                "Total transaction costs should be greater than zero");

//            // Ensure that costs are proportional to trade value 
//            double totalTradingValue = resultWithCosts.Trades.Sum(t => 
//                t.Quantity * t.EntryPrice + (t.ExitPrice.HasValue ? t.Quantity * t.ExitPrice.Value : 0));

//            // Rough estimation: expect costs to be around 0.5% of trading value
//            double expectedCostsPercent = totalTradingValue * 0.005;

//            // Allow for some margin due to slippage and other factors
//            Assert.True(Math.Abs(resultWithCosts.TotalTransactionCosts - expectedCostsPercent) < expectedCostsPercent * 0.2,
//                "Total costs should be approximately 0.5% of trade value");
//        }

//        /// <summary>
//        /// Test realistic transaction cost model with commission, spread and slippage
//        /// </summary>
//        [Fact]
//        public async Task RealisticCosts_ShouldImpactReturnsAndTradeMetrics()
//        {
//            // Arrange
//            var backtest = new BacktestingEngine();
//            var strategy = new TestStrategy();
//            var historical = GenerateTestHistoricalData();

//            // Create realistic retail brokerage model
//            var costModel = TransactionCostModel.CreateRetailBrokerageModel();

//            // Act
//            var resultWithoutCosts = await backtest.RunBacktestAsync("TEST", historical, strategy);
//            var resultWithCosts = await backtest.RunBacktestAsync("TEST", historical, strategy, costModel: costModel);

//            // Assert
//            Assert.True(resultWithCosts.TotalReturn < resultWithoutCosts.TotalReturn, 
//                "Return with costs should be less than return without costs");

//            // Winning trades should be fewer with costs
//            Assert.True(resultWithCosts.WinningTrades <= resultWithoutCosts.WinningTrades,
//                "Number of winning trades should decrease or stay the same when costs are included");

//            // Profit factor should be lower with costs
//            Assert.True(resultWithCosts.ProfitFactor <= resultWithoutCosts.ProfitFactor,
//                "Profit factor should decrease when costs are included");

//            // Make sure transaction costs are reported
//            Assert.True(resultWithCosts.TotalTransactionCosts > 0);
//            Assert.True(resultWithCosts.TransactionCostsPercentage > 0);
//        }

//        /// <summary>
//        /// Generate test historical data with an upward trend
//        /// </summary>
//        private List<HistoricalPrice> GenerateTestHistoricalData()
//        {
//            var data = new List<HistoricalPrice>();
//            var random = new Random(42); // Fixed seed for reproducibility

//            double price = 100.0;
//            DateTime date = DateTime.Now.AddYears(-1);

//            for (int i = 0; i < 252; i++) // One year of trading days
//            {
//                date = date.AddDays(1);
//                if (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday)
//                {
//                    continue;
//                }

//                double dailyChange = (random.NextDouble() - 0.4) * 2; // Slightly positive bias
//                price += price * (dailyChange / 100);

//                double high = price * (1 + random.NextDouble() * 0.01);
//                double low = price * (1 - random.NextDouble() * 0.01);

//                data.Add(new HistoricalPrice
//                {
//                    Date = date,
//                    Open = price,
//                    Close = price,
//                    High = high,
//                    Low = low,
//                    Volume = random.Next(10000, 1000000)
//                });
//            }

//            return data;
//        }
//    }

//    /// <summary>
//    /// Simple test strategy for backtesting
//    /// </summary>
//    public class TestStrategy : StrategyProfile
//    {
//        public TestStrategy()
//        {
//            Name = "Test Strategy";
//        }

//        public override string GenerateSignal(List<HistoricalPrice> historical, int index)
//        {
//            if (index < 20) return null;

//            // Simple moving average crossover
//            double shortMA = historical.Skip(index - 10).Take(10).Average(h => h.Close);
//            double longMA = historical.Skip(index - 20).Take(20).Average(h => h.Close);

//            if (shortMA > longMA) return "BUY";
//            if (shortMA < longMA) return "SELL";

//            return null;
//        }
//    }
//}