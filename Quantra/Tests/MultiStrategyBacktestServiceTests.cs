//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Threading.Tasks;
//using Xunit;
//using Quantra.Models; // Only import Models
//using Quantra.DAL.Services.Interfaces; // Needed for MultiStrategyBacktestService

//namespace Quantra.Tests
//{
//    public class MultiStrategyBacktestServiceTests
//    {
//        /// <summary>
//        /// Test that the multi-strategy backtest service runs multiple strategies correctly
//        /// </summary>
//        [Fact]
//        public async Task RunComparison_ShouldReturnResultsForAllStrategies()
//        {
//            // Arrange
//            var service = new MultiStrategyBacktestService();
//            var strategies = new List<Quantra.Services.StrategyProfile>
//            {
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 10, SlowPeriod = 30, Name = "SMA(10,30)" },
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 5, SlowPeriod = 20, Name = "SMA(5,20)" },
//                new TestRsiStrategy { Name = "RSI Strategy" }
//            };
            
//            // Act
//            var result = await service.RunComparisonAsync("TEST", strategies);
            
//            // Assert
//            Assert.NotNull(result);
//            Assert.Equal(strategies.Count, result.StrategyResults.Count);
//            Assert.Equal("TEST", result.Symbol);
            
//            // Check that all strategy names are present in results
//            foreach (var strategy in strategies)
//            {
//                Assert.Contains(result.StrategyResults, s => s.StrategyName == strategy.Name);
//            }
//        }
        
//        /// <summary>
//        /// Test that the correlation matrix is calculated correctly
//        /// </summary>
//        [Fact]
//        public async Task RunComparison_ShouldCalculateCorrelationMatrix()
//        {
//            // Arrange
//            var service = new MultiStrategyBacktestService();
//            var strategies = new List<Quantra.Services.StrategyProfile>
//            {
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 10, SlowPeriod = 30, Name = "SMA(10,30)" },
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 5, SlowPeriod = 20, Name = "SMA(5,20)" }
//            };
            
//            // Act
//            var result = await service.RunComparisonAsync("TEST", strategies);
            
//            // Assert
//            Assert.NotNull(result.CorrelationMatrix);
//            Assert.Equal(strategies.Count, result.CorrelationMatrix.GetLength(0));
//            Assert.Equal(strategies.Count, result.CorrelationMatrix.GetLength(1));
            
//            // Diagonal should be 1.0 (perfect correlation with self)
//            for (int i = 0; i < strategies.Count; i++)
//            {
//                Assert.Equal(1.0, result.CorrelationMatrix[i, i], 3);
//            }
            
//            // Matrix should be symmetric
//            for (int i = 0; i < strategies.Count; i++)
//            {
//                for (int j = 0; j < strategies.Count; j++)
//                {
//                    Assert.Equal(result.CorrelationMatrix[i, j], result.CorrelationMatrix[j, i], 8);
//                }
//            }
//        }
        
//        /// <summary>
//        /// Test that the strategy rankings are calculated correctly
//        /// </summary>
//        [Fact]
//        public async Task RunComparison_ShouldCalculateStrategyRankings()
//        {
//            // Arrange
//            var service = new MultiStrategyBacktestService();
//            var strategies = new List<Quantra.Services.StrategyProfile>
//            {
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 10, SlowPeriod = 30, Name = "SMA(10,30)" },
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 5, SlowPeriod = 20, Name = "SMA(5,20)" },
//                new TestRsiStrategy { Name = "RSI Strategy" }
//            };
            
//            // Act
//            var result = await service.RunComparisonAsync("TEST", strategies);
            
//            // Assert
//            Assert.NotEmpty(result.TotalReturnRanking);
//            Assert.NotEmpty(result.SharpeRatioRanking);
//            Assert.NotEmpty(result.MaxDrawdownRanking);
//            Assert.NotEmpty(result.ProfitFactorRanking);
//            Assert.NotEmpty(result.WinRateRanking);
            
//            // All rankings should contain all strategies
//            foreach (var strategy in strategies)
//            {
//                Assert.Contains(strategy.Name, result.TotalReturnRanking);
//                Assert.Contains(strategy.Name, result.SharpeRatioRanking);
//                Assert.Contains(strategy.Name, result.MaxDrawdownRanking);
//                Assert.Contains(strategy.Name, result.ProfitFactorRanking);
//                Assert.Contains(strategy.Name, result.WinRateRanking);
//            }
//        }
        
//        /// <summary>
//        /// Test that the optimal portfolio weights calculation works correctly
//        /// </summary>
//        [Fact]
//        public async Task CalculateOptimalPortfolioWeights_ShouldReturnValidWeights()
//        {
//            // Arrange
//            var service = new MultiStrategyBacktestService();
//            var strategies = new List<Quantra.Services.StrategyProfile>
//            {
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 10, SlowPeriod = 30, Name = "SMA(10,30)" },
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 5, SlowPeriod = 20, Name = "SMA(5,20)" }
//            };
            
//            // Act
//            var result = await service.RunComparisonAsync("TEST", strategies);
//            var weights = result.CalculateOptimalPortfolioWeights();
            
//            // Assert
//            Assert.NotNull(weights);
//            Assert.Equal(strategies.Count, weights.Count);
            
//            // Weights should sum to approximately 1.0
//            double totalWeight = weights.Values.Sum();
//            Assert.InRange(totalWeight, 0.99, 1.01);
            
//            // All weights should be positive
//            foreach (var weight in weights.Values)
//            {
//                Assert.True(weight >= 0);
//            }
//        }
        
//        /// <summary>
//        /// Test that the combined portfolio simulation works correctly
//        /// </summary>
//        [Fact]
//        public async Task SimulateCombinedPortfolio_ShouldReturnValidResult()
//        {
//            // Arrange
//            var service = new MultiStrategyBacktestService();
//            var strategies = new List<Quantra.Services.StrategyProfile>
//            {
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 10, SlowPeriod = 30, Name = "SMA(10,30)" },
//                new Quantra.Services.SmaCrossoverStrategy { FastPeriod = 5, SlowPeriod = 20, Name = "SMA(5,20)" }
//            };
            
//            // Act
//            var result = await service.RunComparisonAsync("TEST", strategies);
//            var weights = new Dictionary<string, double>
//            {
//                { "SMA(10,30)", 0.5 },
//                { "SMA(5,20)", 0.5 }
//            };
//            var combinedResult = result.SimulateCombinedPortfolio(weights);
            
//            // Assert
//            Assert.NotNull(combinedResult);
//            Assert.NotEmpty(combinedResult.EquityCurve);
//            Assert.NotEmpty(combinedResult.DrawdownCurve);
//            Assert.Equal(result.Symbol + " (Combined Portfolio)", combinedResult.Symbol);
            
//            // Combined result should have valid performance metrics
//            Assert.InRange(combinedResult.SharpeRatio, -10, 10);
//            Assert.InRange(combinedResult.SortinoRatio, -10, 10);
//            Assert.InRange(combinedResult.MaxDrawdown, 0, 1);
//        }
//    }
    
//    /// <summary>
//    /// Simple RSI strategy for testing
//    /// </summary>
//    public class TestRsiStrategy : Quantra.Models.StrategyProfile
//    {
//        public int RsiPeriod { get; set; } = 14;
//        public int OverboughtLevel { get; set; } = 70;
//        public int OversoldLevel { get; set; } = 30;
        
//        public TestRsiStrategy()
//        {
//            this.Name = "RSI Strategy";
//        }
        
//        public override string GenerateSignal(List<HistoricalPrice> historical, int? index = null)
//        {
//            int idx = index ?? (historical?.Count ?? 0) - 1;
//            if (idx < RsiPeriod + 1)
//                return null;
                
//            // Calculate RSI
//            var prices = historical.Select(h => h.Close).ToList();
//            double rsi = CalculateRSI(prices.Take(idx + 1).ToList(), RsiPeriod);
            
//            // Get previous RSI for comparison
//            double prevRsi = CalculateRSI(prices.Take(idx).ToList(), RsiPeriod);
            
//            // Generate signals based on RSI levels
//            if (prevRsi > OverboughtLevel && rsi <= OverboughtLevel)
//                return "SELL"; // RSI falling from overbought
                
//            if (prevRsi < OversoldLevel && rsi >= OversoldLevel)
//                return "BUY"; // RSI rising from oversold
                
//            return null;
//        }
        
//        public override bool ValidateConditions(Dictionary<string, double> indicators)
//        {
//            // Simple validation for test: require RSI indicator
//            return indicators != null && indicators.ContainsKey("RSI");
//        }
        
//        private double CalculateRSI(List<double> prices, int period)
//        {
//            if (prices.Count <= period)
//                return 50; // Default to neutral
                
//            List<double> gains = new List<double>();
//            List<double> losses = new List<double>();
            
//            // Calculate price changes
//            for (int i = 1; i < prices.Count; i++)
//            {
//                double change = prices[i] - prices[i - 1];
//                gains.Add(change > 0 ? change : 0);
//                losses.Add(change < 0 ? -change : 0);
//            }
            
//            // Get last 'period' values
//            gains = gains.Skip(Math.Max(0, gains.Count - period)).Take(period).ToList();
//            losses = losses.Skip(Math.Max(0, losses.Count - period)).Take(period).ToList();
            
//            double avgGain = gains.Any() ? gains.Average() : 0;
//            double avgLoss = losses.Any() ? losses.Average() : 0.001; // Avoid division by zero
            
//            double rs = avgLoss > 0 ? avgGain / avgLoss : 100;
//            double rsi = 100 - (100 / (1 + rs));
            
//            return rsi;
//        }
//    }
//}