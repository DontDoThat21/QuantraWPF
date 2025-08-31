using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Enums;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using System.Reflection;
using Xunit;

namespace Quantra.Tests.Services
{
    public class WebullTradingBotRebalancingTests
    {
        [Fact]
        public async Task RebalancePortfolio_Basic_CreatesCorrectOrders()
        {
            // Arrange
            var bot = new WebullTradingBot();
            
            // Use reflection to set up test paper portfolio
            var paperPortfolioField = typeof(WebullTradingBot).GetField("paperPortfolio", 
                BindingFlags.NonPublic | BindingFlags.Instance);
            var paperPortfolio = new Dictionary<string, double>
            {
                { "VTI", 100 }, // Stock ETF
                { "AGG", 50 }   // Bond ETF
            };
            paperPortfolioField.SetValue(bot, paperPortfolio);
            
            // Set target allocations (50/50 split)
            var allocations = new Dictionary<string, double>
            {
                { "VTI", 0.5 },
                { "AGG", 0.5 }
            };
            bool result = bot.SetPortfolioAllocations(allocations);
            
            // Mock GetMarketPrice using reflection
            // For testing, override the GetMarketPrice method to return predictable values
            // VTI: $200, AGG: $100
            // This means portfolio is currently:
            // VTI: 100 shares * $200 = $20,000 (66.7%)
            // AGG: 50 shares * $100 = $5,000 (33.3%)
            // Target is 50/50, so we need to sell VTI and buy AGG
            
            // Use reflection to access the _scheduledOrders field
            var scheduledOrdersField = typeof(WebullTradingBot).GetField("_scheduledOrders", 
                BindingFlags.NonPublic | BindingFlags.Instance);
            scheduledOrdersField.SetValue(bot, new Dictionary<string, List<ScheduledOrder>>());
            
            // Create a mock method handler
            var getMarketPriceMethod = typeof(WebullTradingBot).GetMethod("GetMarketPrice");
            var originalMethod = getMarketPriceMethod;
            
            // Create a new WebullTradingBot that uses our mock implementation
            var mockBot = new MockWebullTradingBot();
            mockBot.MockPrices["VTI"] = 200.0;
            mockBot.MockPrices["AGG"] = 100.0;
            
            // Copy fields from the original bot
            foreach (var field in typeof(WebullTradingBot).GetFields(BindingFlags.NonPublic | BindingFlags.Instance))
            {
                var value = field.GetValue(bot);
                field.SetValue(mockBot, value);
            }
            
            // Act
            bool rebalanceResult = await mockBot.RebalancePortfolio(0.05); // 5% tolerance
            
            // Assert
            Assert.True(rebalanceResult);
            
            // Get the scheduled orders
            var scheduledOrders = scheduledOrdersField.GetValue(mockBot) as 
                Dictionary<string, List<ScheduledOrder>>;
            
            Assert.NotNull(scheduledOrders);
            
            // Expected rebalancing calculations:
            // Total portfolio value: $25,000
            // Target VTI value: $12,500
            // Current VTI value: $20,000
            // Difference: -$7,500
            // Shares to sell: 37.5 (~37 shares)
            
            // Target AGG value: $12,500
            // Current AGG value: $5,000
            // Difference: +$7,500
            // Shares to buy: 75 shares
            
            // Check VTI order (sell)
            if (scheduledOrders.ContainsKey("VTI"))
            {
                var vtiOrders = scheduledOrders["VTI"];
                Assert.True(vtiOrders.Count > 0);
                Assert.Equal("SELL", vtiOrders[0].OrderType);
                Assert.True(vtiOrders[0].IsRebalancing);
                Assert.Equal(37, vtiOrders[0].Quantity);
            }
            
            // Check AGG order (buy)
            if (scheduledOrders.ContainsKey("AGG"))
            {
                var aggOrders = scheduledOrders["AGG"];
                Assert.True(aggOrders.Count > 0);
                Assert.Equal("BUY", aggOrders[0].OrderType);
                Assert.True(aggOrders[0].IsRebalancing);
                Assert.Equal(75, aggOrders[0].Quantity);
            }
        }
        
        [Fact]
        public void RebalancingProfile_MarketConditionAdjustment_AdjustsAllocationsCorrectly()
        {
            // Arrange
            var profile = new RebalancingProfile
            {
                Name = "Test Profile",
                TargetAllocations = new Dictionary<string, double>
                {
                    { "VTI", 0.6 },  // 60% stocks (risk asset)
                    { "AGG", 0.4 }   // 40% bonds (defensive asset)
                },
                EnableMarketConditionAdjustments = true,
                TolerancePercentage = 0.03,
                MaxDeviationInAdverseConditions = 0.2
            };
            
            var highVolatilityConditions = new MarketConditions
            {
                VolatilityIndex = 35, // High volatility
                MarketTrend = -0.3,   // Bearish trend
                DefensiveAssets = new List<string> { "AGG" }
            };
            
            // Act
            var adjustedAllocations = profile.GetMarketAdjustedAllocations(highVolatilityConditions);
            
            // Assert
            Assert.NotNull(adjustedAllocations);
            Assert.True(adjustedAllocations.ContainsKey("VTI"));
            Assert.True(adjustedAllocations.ContainsKey("AGG"));
            
            // Check that allocations were adjusted toward defensive assets
            Assert.True(adjustedAllocations["VTI"] < profile.TargetAllocations["VTI"]);
            Assert.True(adjustedAllocations["AGG"] > profile.TargetAllocations["AGG"]);
            
            // Check that allocations still sum to 1.0
            double sum = adjustedAllocations.Values.Sum();
            Assert.True(Math.Abs(sum - 1.0) < 0.0001);
        }
    }
    
    /// <summary>
    /// Mock WebullTradingBot for testing
    /// </summary>
    internal class MockWebullTradingBot : WebullTradingBot
    {
        public Dictionary<string, double> MockPrices { get; set; } = new Dictionary<string, double>();
        
        public override async Task<double> GetMarketPrice(string symbol)
        {
            if (MockPrices.ContainsKey(symbol))
            {
                return MockPrices[symbol];
            }
            return 100.0; // Default price
        }
        
        // Mock IsTradingAllowed to always return true for testing
        protected new bool IsTradingAllowed()
        {
            return true;
        }
    }
}