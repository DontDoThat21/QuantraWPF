using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.DAL.Services;

namespace Quantra.Tests.Services
{
    public class BacktestingEngineTests
    {
        /// <summary>
        /// Test that transaction costs using a fixed commission model reduce returns appropriately
        /// </summary>
        [Fact]
        public async Task FixedCommissions_ShouldReduceReturns()
        {
            // Arrange
            var backtest = new BacktestingEngine();
            var strategy = new TestStrategy();
            var historical = GenerateTestHistoricalData();
            
            // Set up fixed commission model - $10 per trade
            var costModel = TransactionCostModel.CreateFixedCommissionModel(10);
            
            // Act
            var resultWithoutCosts = await backtest.RunBacktestAsync("TEST", historical, strategy);
            var resultWithCosts = await backtest.RunBacktestAsync("TEST", historical, strategy, costModel: costModel);
            
            // Assert
            Assert.True(resultWithCosts.TotalReturn < resultWithoutCosts.TotalReturn, 
                "Return with costs should be less than return without costs");
            Assert.True(resultWithCosts.TotalTransactionCosts > 0, 
                "Total transaction costs should be greater than zero");
            
            // Verify the cost impact is tracked
            Assert.True(resultWithCosts.GrossReturn - resultWithCosts.NetReturn > 0,
                "Cost impact should be positive");
        }
        
        /// <summary>
        /// Test transaction costs with percentage-based commission model
        /// </summary>
        [Fact]
        public async Task PercentageCommissions_ShouldReduceReturns()
        {
            // Arrange
            var backtest = new BacktestingEngine();
            var strategy = new TestStrategy();
            var historical = GenerateTestHistoricalData();
            
            // Set up percentage commission model - 0.5% per trade
            var costModel = TransactionCostModel.CreatePercentageCommissionModel(0.005);
            
            // Act
            var resultWithoutCosts = await backtest.RunBacktestAsync("TEST", historical, strategy);
            var resultWithCosts = await backtest.RunBacktestAsync("TEST", historical, strategy, costModel: costModel);
            
            // Assert
            Assert.True(resultWithCosts.TotalReturn < resultWithoutCosts.TotalReturn, 
                "Return with costs should be less than return without costs");
            Assert.True(resultWithCosts.TotalTransactionCosts > 0, 
                "Total transaction costs should be greater than zero");
                
            // Verify transaction cost percentage is tracked
            Assert.True(resultWithCosts.TransactionCostsPercentage > 0,
                "Transaction costs percentage should be positive");
        }
        
        /// <summary>
        /// Test realistic transaction cost model with commission, spread and slippage
        /// </summary>
        [Fact]
        public async Task RealisticCosts_ShouldImpactReturnsAndTradeMetrics()
        {
            // Arrange
            var backtest = new BacktestingEngine();
            var strategy = new TestStrategy();
            var historical = GenerateTestHistoricalData();
            
            // Create realistic retail brokerage model
            var costModel = TransactionCostModel.CreateRetailBrokerageModel();
            
            // Act
            var resultWithoutCosts = await backtest.RunBacktestAsync("TEST", historical, strategy);
            var resultWithCosts = await backtest.RunBacktestAsync("TEST", historical, strategy, costModel: costModel);
            
            // Assert
            Assert.True(resultWithCosts.TotalReturn < resultWithoutCosts.TotalReturn, 
                "Return with costs should be less than return without costs");
            
            // Make sure transaction costs are reported
            Assert.True(resultWithCosts.TotalTransactionCosts > 0);
            Assert.True(resultWithCosts.TransactionCostsPercentage > 0);
        }
        
        /// <summary>
        /// Generate test historical data with an upward trend
        /// </summary>
        private List<HistoricalPrice> GenerateTestHistoricalData()
        {
            var data = new List<HistoricalPrice>();
            var random = new Random(42); // Fixed seed for reproducibility
            
            double price = 100.0;
            DateTime date = DateTime.Now.AddYears(-1);
            
            for (int i = 0; i < 252; i++) // One year of trading days
            {
                date = date.AddDays(1);
                if (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday)
                {
                    continue;
                }
                
                double dailyChange = (random.NextDouble() - 0.4) * 2; // Slightly positive bias
                price += price * (dailyChange / 100);
                
                double high = price * (1 + random.NextDouble() * 0.01);
                double low = price * (1 - random.NextDouble() * 0.01);
                
                data.Add(new HistoricalPrice
                {
                    Date = date,
                    Open = price,
                    Close = price,
                    High = high,
                    Low = low,
                    Volume = random.Next(10000, 1000000)
                });
            }
            
            return data;
        }
    }
    
    /// <summary>
    /// Simple test strategy for backtesting
    /// </summary>
    public class TestStrategy : Quantra.Models.StrategyProfile
    {
        public TestStrategy()
        {
            Name = "Test Strategy";
        }
        
        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            int idx = index ?? (prices.Count - 1);
            if (idx < 20) return null;
            
            // Simple moving average crossover
            double shortMA = prices.Skip(idx - 10).Take(10).Average(h => h.Close);
            double longMA = prices.Skip(idx - 20).Take(20).Average(h => h.Close);
            
            if (shortMA > longMA) return "BUY";
            if (shortMA < longMA) return "SELL";
            
            return null;
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            return true;
        }
    }
}