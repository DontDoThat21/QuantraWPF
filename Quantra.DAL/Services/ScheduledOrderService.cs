using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Enums;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    public class ScheduledOrderService : IScheduledOrderService
    {
        private Dictionary<string, List<ScheduledOrder>> _scheduledOrders = new Dictionary<string, List<ScheduledOrder>>();
        private readonly IOrderService _orderService;

        public ScheduledOrderService(IOrderService orderService)
        {
            _orderService = orderService;
        }

        /// <summary>
        /// Splits a large order into smaller chunks
        /// </summary>
        public bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes)
        {
            // Call the enhanced version with default parameters
            return SplitLargeOrder(symbol, quantity, orderType, price, chunks, intervalMinutes,
                priceVariancePercent: 0,
                randomizeIntervals: false,
                distribution: OrderDistributionType.Equal);
        }

        /// <summary>
        /// Splits a large order into smaller chunks with advanced options for minimizing market impact
        /// </summary>
        public bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes,
            double priceVariancePercent, bool randomizeIntervals, OrderDistributionType distribution)
        {
            try
            {
                // Validate parameters
                if (chunks <= 0 || quantity <= 0 || string.IsNullOrEmpty(symbol))
                {
                    //DatabaseMonolith.Log("Error", $"Invalid parameters for split order: Symbol={symbol}, Quantity={quantity}, Chunks={chunks}");
                    return false;
                }

                // Ensure price variance is within a reasonable range (0-5%)
                priceVariancePercent = Math.Max(0, Math.Min(5, priceVariancePercent));

                // Generate a unique ID for this group of split orders
                string splitOrderGroupId = $"{symbol}-{Guid.NewGuid():N}";

                // Calculate order quantities based on distribution type
                List<int> chunkSizes = CalculateChunkSizes(quantity, chunks, distribution);

                // Calculate base intervals based on randomization setting
                List<int> intervals = CalculateIntervals(chunks, intervalMinutes, randomizeIntervals);

                // Keep track of cumulative time for scheduling
                int cumulativeMinutes = 0;

                // Schedule each chunk
                for (int i = 0; i < chunks; i++)
                {
                    int chunkShares = chunkSizes[i];

                    // Apply price variance if specified
                    double chunkPrice = GenerateRandomPriceVariance(price, priceVariancePercent);

                    // Add interval for this chunk to cumulative time
                    cumulativeMinutes += intervals[i];

                    // Create the scheduled order
                    var order = new ScheduledOrder
                    {
                        Symbol = symbol,
                        Quantity = chunkShares,
                        OrderType = orderType,
                        Price = chunkPrice,
                        ExecutionTime = DateTime.Now.AddMinutes(cumulativeMinutes),
                        IsSplitOrder = true,
                        SplitOrderGroupId = splitOrderGroupId,
                        SplitOrderSequence = i + 1,
                        SplitOrderTotalChunks = chunks
                    };

                    // Add to scheduled orders
                    if (!_scheduledOrders.ContainsKey(symbol))
                    {
                        _scheduledOrders[symbol] = new List<ScheduledOrder>();
                    }
                    _scheduledOrders[symbol].Add(order);
                }

                // Log details of the split order
                string distributionName = distribution.ToString();
                string intervalType = randomizeIntervals ? "randomized" : "fixed";
                string priceVariance = priceVariancePercent > 0 ? $" with price variance of ±{priceVariancePercent:F1}%" : "";

                //DatabaseMonolith.Log("Info", $"Enhanced order split for {symbol}: {quantity} {orderType} shares into {chunks} chunks " +
                //$"using {distributionName} distribution, {intervalType} intervals{priceVariance}. " +
                //$"Group ID: {splitOrderGroupId}");

                return true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to split large order for {symbol}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Cancels all remaining chunks of a split order group
        /// </summary>
        public int CancelSplitOrderGroup(string splitOrderGroupId)
        {
            try
            {
                int cancelledCount = 0;
                var symbolsToCheck = _scheduledOrders.Keys.ToList();

                foreach (var symbol in symbolsToCheck)
                {
                    var ordersToRemove = _scheduledOrders[symbol]
                        .Where(o => o.IsSplitOrder && o.SplitOrderGroupId == splitOrderGroupId)
                        .ToList();

                    foreach (var order in ordersToRemove)
                    {
                        _scheduledOrders[symbol].Remove(order);
                        cancelledCount++;
                    }

                    // Clean up empty symbol entries
                    if (_scheduledOrders[symbol].Count == 0)
                    {
                        _scheduledOrders.Remove(symbol);
                    }
                }

                //DatabaseMonolith.Log("Info", $"Cancelled {cancelledCount} remaining chunks from split order group {splitOrderGroupId}");
                return cancelledCount;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to cancel split order group {splitOrderGroupId}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Gets scheduled orders for a symbol
        /// </summary>
        public List<ScheduledOrder> GetScheduledOrders(string symbol)
        {
            return _scheduledOrders.ContainsKey(symbol) ? _scheduledOrders[symbol].ToList() : new List<ScheduledOrder>();
        }

        /// <summary>
        /// Gets all scheduled orders
        /// </summary>
        public Dictionary<string, List<ScheduledOrder>> GetAllScheduledOrders()
        {
            return new Dictionary<string, List<ScheduledOrder>>(_scheduledOrders);
        }

        /// <summary>
        /// Monitors scheduled orders and executes them when due
        /// </summary>
        public async Task MonitorScheduledOrders()
        {
            try
            {
                var now = DateTime.Now;
                var symbolsToCheck = _scheduledOrders.Keys.ToList();

                foreach (var symbol in symbolsToCheck)
                {
                    var ordersToExecute = _scheduledOrders[symbol]
                        .Where(o => o.ExecutionTime <= now)
                        .ToList();

                    foreach (var order in ordersToExecute)
                    {
                        try
                        {
                            // Execute the order through the order service
                            bool success = await _orderService.PlaceLimitOrder(order.Symbol, order.Quantity, order.OrderType, order.Price);

                            if (success)
                            {
                                //DatabaseMonolith.Log("Info", $"Executed scheduled order: {order.OrderType} {order.Quantity} {order.Symbol} at {order.Price:C2}");

                                // Remove the executed order
                                _scheduledOrders[symbol].Remove(order);
                            }
                            else
                            {
                                //DatabaseMonolith.Log("Warning", $"Failed to execute scheduled order: {order.OrderType} {order.Quantity} {order.Symbol} at {order.Price:C2}");

                                // Reschedule for 1 minute later
                                order.ExecutionTime = now.AddMinutes(1);
                            }
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Error", $"Error executing scheduled order for {order.Symbol}", ex.ToString());

                            // Reschedule for 5 minutes later on error
                            order.ExecutionTime = now.AddMinutes(5);
                        }
                    }

                    // Clean up empty symbol entries
                    if (_scheduledOrders[symbol].Count == 0)
                    {
                        _scheduledOrders.Remove(symbol);
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error in scheduled order monitoring", ex.ToString());
            }
        }

        /// <summary>
        /// Adds a scheduled order
        /// </summary>
        public bool AddScheduledOrder(ScheduledOrder order)
        {
            try
            {
                if (order == null || string.IsNullOrEmpty(order.Symbol))
                {
                    return false;
                }

                if (!_scheduledOrders.ContainsKey(order.Symbol))
                {
                    _scheduledOrders[order.Symbol] = new List<ScheduledOrder>();
                }

                _scheduledOrders[order.Symbol].Add(order);
                return true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to add scheduled order for {order?.Symbol}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Removes a scheduled order
        /// </summary>
        public bool RemoveScheduledOrder(string symbol, string orderId)
        {
            try
            {
                if (!_scheduledOrders.ContainsKey(symbol))
                {
                    return false;
                }

                var orderToRemove = _scheduledOrders[symbol].FirstOrDefault(o => o.SplitOrderGroupId == orderId);
                if (orderToRemove != null)
                {
                    _scheduledOrders[symbol].Remove(orderToRemove);

                    // Clean up empty symbol entries
                    if (_scheduledOrders[symbol].Count == 0)
                    {
                        _scheduledOrders.Remove(symbol);
                    }

                    return true;
                }

                return false;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to remove scheduled order {orderId} for {symbol}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Calculates the size of each chunk based on the distribution type
        /// </summary>
        public List<int> CalculateChunkSizes(int quantity, int chunks, OrderDistributionType distribution)
        {
            List<int> chunkSizes = new List<int>();

            switch (distribution)
            {
                case OrderDistributionType.FrontLoaded:
                    // Front-loaded: Larger chunks at the beginning, tapering off
                    double totalWeight = chunks * (chunks + 1) / 2.0; // Sum of 1 to chunks

                    for (int i = chunks; i >= 1; i--)
                    {
                        int chunkSize = (int)Math.Round(i / totalWeight * quantity);
                        chunkSizes.Add(chunkSize);
                    }
                    break;

                case OrderDistributionType.BackLoaded:
                    // Back-loaded: Smaller chunks at the beginning, larger at the end
                    totalWeight = chunks * (chunks + 1) / 2.0; // Sum of 1 to chunks

                    for (int i = 1; i <= chunks; i++)
                    {
                        int chunkSize = (int)Math.Round(i / totalWeight * quantity);
                        chunkSizes.Add(chunkSize);
                    }
                    break;

                case OrderDistributionType.Normal:
                    // Normal (bell curve): Middle chunks are larger
                    double mean = (chunks - 1) / 2.0;
                    double stdDev = chunks / 6.0; // ~99% within the range
                    double[] weights = new double[chunks];
                    double weightSum = 0;

                    for (int i = 0; i < chunks; i++)
                    {
                        weights[i] = Math.Exp(-0.5 * Math.Pow((i - mean) / stdDev, 2));
                        weightSum += weights[i];
                    }

                    for (int i = 0; i < chunks; i++)
                    {
                        int chunkSize = (int)Math.Round(weights[i] / weightSum * quantity);
                        chunkSizes.Add(chunkSize);
                    }
                    break;

                case OrderDistributionType.Equal:
                default:
                    // Equal distribution (with remainder added to first chunk)
                    int sharesPerChunk = quantity / chunks;
                    int remainder = quantity % chunks;

                    for (int i = 0; i < chunks; i++)
                    {
                        int chunkSize = sharesPerChunk;
                        if (i == 0)
                        {
                            chunkSize += remainder;
                        }
                        chunkSizes.Add(chunkSize);
                    }
                    break;
            }

            // Ensure we distribute exactly the requested quantity
            int totalAllocated = chunkSizes.Sum();
            if (totalAllocated != quantity)
            {
                int diff = quantity - totalAllocated;
                chunkSizes[0] += diff;
            }

            return chunkSizes;
        }

        /// <summary>
        /// Generates random price variance within percentage bounds
        /// </summary>
        public double GenerateRandomPriceVariance(double basePrice, double variancePercent)
        {
            if (variancePercent <= 0)
            {
                return basePrice;
            }

            Random random = new Random();
            // Calculate variance within the specified percentage range
            double varianceFactor = 1.0 + (random.NextDouble() * 2 - 1) * variancePercent / 100.0;
            return Math.Round(basePrice * varianceFactor, 2);
        }

        /// <summary>
        /// Calculates randomized interval if randomization is enabled
        /// </summary>
        public int CalculateRandomizedInterval(int baseInterval, bool randomizeIntervals)
        {
            if (!randomizeIntervals)
            {
                return baseInterval;
            }

            Random random = new Random();
            // Randomize interval between 50% and 150% of base interval
            int minInterval = Math.Max(1, (int)(baseInterval * 0.5));
            int maxInterval = (int)(baseInterval * 1.5);
            return random.Next(minInterval, maxInterval + 1);
        }

        /// <summary>
        /// Calculates the time intervals between chunks
        /// </summary>
        private List<int> CalculateIntervals(int chunks, int baseIntervalMinutes, bool randomize)
        {
            List<int> intervals = new List<int>();

            for (int i = 0; i < chunks; i++)
            {
                if (i == 0)
                {
                    // First chunk is executed immediately
                    intervals.Add(0);
                }
                else
                {
                    intervals.Add(CalculateRandomizedInterval(baseIntervalMinutes, randomize));
                }
            }

            return intervals;
        }

        /// <summary>
        /// Places multiple orders as part of a strategy
        /// </summary>
        public async Task<bool> PlaceMultiLegOrder(List<ScheduledOrder> orders)
        {
            try
            {
                if (orders == null || orders.Count == 0)
                {
                    return false;
                }

                bool allSuccessful = true;
                foreach (var order in orders)
                {
                    bool success = await _orderService.PlaceLimitOrder(order.Symbol, order.Quantity, order.OrderType, order.Price);
                    if (!success)
                    {
                        allSuccessful = false;
                        //DatabaseMonolith.Log("Warning", $"Failed to place multi-leg order: {order.OrderType} {order.Quantity} {order.Symbol} at {order.Price:C2}");
                    }
                }

                return allSuccessful;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to place multi-leg order", ex.ToString());
                return false;
            }
        }
    }
}