using System;
using System.Linq;
using Quantra.Enums;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using System.Reflection;
using Xunit;
using Quantra.DAL.Services;

namespace Quantra.Tests.Services
{
    public class WebullTradingBotTests
    {
        [Fact]
        public void SplitLargeOrder_Basic_ShouldCreateCorrectNumberOfChunks()
        {
            // Arrange
            var bot = new WebullTradingBot();
            string symbol = "AAPL";
            int quantity = 1000;
            string orderType = "BUY";
            double price = 175.0;
            int chunks = 5;
            int intervalMinutes = 10;

            // Act
            bool result = bot.SplitLargeOrder(symbol, quantity, orderType, price, chunks, intervalMinutes);

            // Assert
            Assert.True(result);

            // Use reflection to access private field _scheduledOrders
            var scheduledOrdersField = typeof(WebullTradingBot).GetField("_scheduledOrders",
                BindingFlags.NonPublic | BindingFlags.Instance);
            var scheduledOrders = scheduledOrdersField.GetValue(bot) as
                System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<ScheduledOrder>>;

            Assert.NotNull(scheduledOrders);
            Assert.True(scheduledOrders.ContainsKey(symbol));
            Assert.Equal(chunks, scheduledOrders[symbol].Count);
            Assert.Equal(quantity, scheduledOrders[symbol].Sum(o => o.Quantity));
        }

        [Fact]
        public void SplitLargeOrder_Enhanced_WithPriceVariance_ShouldVaryPrices()
        {
            // Arrange
            var bot = new WebullTradingBot();
            string symbol = "MSFT";
            int quantity = 1000;
            string orderType = "BUY";
            double price = 350.0;
            int chunks = 4;
            int intervalMinutes = 15;
            double priceVariancePercent = 2.0; // 2%
            bool randomizeIntervals = false;
            var distribution = OrderDistributionType.Equal;

            // Act
            bool result = bot.SplitLargeOrder(symbol, quantity, orderType, price, chunks, intervalMinutes,
                priceVariancePercent, randomizeIntervals, distribution);

            // Assert
            Assert.True(result);

            // Use reflection to access private field _scheduledOrders
            var scheduledOrdersField = typeof(WebullTradingBot).GetField("_scheduledOrders",
                BindingFlags.NonPublic | BindingFlags.Instance);
            var scheduledOrders = scheduledOrdersField.GetValue(bot) as
                System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<ScheduledOrder>>;

            Assert.NotNull(scheduledOrders);
            Assert.True(scheduledOrders.ContainsKey(symbol));
            Assert.Equal(chunks, scheduledOrders[symbol].Count);

            // Check if at least one order has a different price (variance applied)
            bool hasPriceVariance = scheduledOrders[symbol].Any(o => Math.Abs(o.Price - price) > 0.01);
            Assert.True(hasPriceVariance);

            // Verify all prices are within the variance range
            double minPrice = price * (1 - priceVariancePercent / 100);
            double maxPrice = price * (1 + priceVariancePercent / 100);
            bool allPricesInRange = scheduledOrders[symbol].All(o => o.Price >= minPrice && o.Price <= maxPrice);
            Assert.True(allPricesInRange);
        }

        [Theory]
        [InlineData(OrderDistributionType.FrontLoaded)]
        [InlineData(OrderDistributionType.BackLoaded)]
        [InlineData(OrderDistributionType.Normal)]
        [InlineData(OrderDistributionType.Equal)]
        public void SplitLargeOrder_WithDifferentDistributions_ShouldCreateValidChunks(OrderDistributionType distribution)
        {
            // Arrange
            var bot = new WebullTradingBot();
            string symbol = "TSLA";
            int quantity = 2000;
            string orderType = "SELL";
            double price = 220.0;
            int chunks = 5;
            int intervalMinutes = 10;

            // Act
            bool result = bot.SplitLargeOrder(symbol, quantity, orderType, price, chunks, intervalMinutes,
                0, false, distribution);

            // Assert
            Assert.True(result);

            // Use reflection to access private field _scheduledOrders
            var scheduledOrdersField = typeof(WebullTradingBot).GetField("_scheduledOrders",
                BindingFlags.NonPublic | BindingFlags.Instance);
            var scheduledOrders = scheduledOrdersField.GetValue(bot) as
                System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<ScheduledOrder>>;

            Assert.NotNull(scheduledOrders);
            Assert.True(scheduledOrders.ContainsKey(symbol));
            Assert.Equal(chunks, scheduledOrders[symbol].Count);

            // Verify total quantity is correct
            int totalShares = scheduledOrders[symbol].Sum(o => o.Quantity);
            Assert.Equal(quantity, totalShares);

            // Verify all chunks have quantities greater than zero
            bool allChunksHaveShares = scheduledOrders[symbol].All(o => o.Quantity > 0);
            Assert.True(allChunksHaveShares);

            // Verify split order tracking properties are set correctly
            bool allAreSplitOrders = scheduledOrders[symbol].All(o => o.IsSplitOrder);
            Assert.True(allAreSplitOrders);

            // All should have the same group ID
            string groupId = scheduledOrders[symbol].First().SplitOrderGroupId;
            bool allSameGroupId = scheduledOrders[symbol].All(o => o.SplitOrderGroupId == groupId);
            Assert.True(allSameGroupId);

            // Sequence numbers should be 1 to chunks
            for (int i = 0; i < chunks; i++)
            {
                Assert.True(scheduledOrders[symbol].Any(o => o.SplitOrderSequence == i + 1));
            }
        }

        [Fact]
        public void CancelSplitOrderGroup_ShouldRemoveAllRemainingChunks()
        {
            // Arrange
            var bot = new WebullTradingBot();
            string symbol = "AMZN";
            int quantity = 500;
            string orderType = "BUY";
            double price = 180.0;
            int chunks = 3;
            int intervalMinutes = 5;

            // Create a split order
            bool createResult = bot.SplitLargeOrder(symbol, quantity, orderType, price, chunks, intervalMinutes);
            Assert.True(createResult);

            // Use reflection to access private field _scheduledOrders and get the group ID
            var scheduledOrdersField = typeof(WebullTradingBot).GetField("_scheduledOrders",
                BindingFlags.NonPublic | BindingFlags.Instance);
            var scheduledOrders = scheduledOrdersField.GetValue(bot) as
                System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<ScheduledOrder>>;

            string groupId = scheduledOrders[symbol].First().SplitOrderGroupId;

            // Act
            int cancelCount = bot.CancelSplitOrderGroup(groupId);

            // Assert
            Assert.Equal(chunks, cancelCount);

            // Refresh scheduledOrders after cancellation
            scheduledOrders = scheduledOrdersField.GetValue(bot) as
                System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<ScheduledOrder>>;

            // Verify orders were removed
            Assert.False(scheduledOrders.ContainsKey(symbol) &&
                scheduledOrders[symbol].Any(o => o.SplitOrderGroupId == groupId));
        }
    }
}