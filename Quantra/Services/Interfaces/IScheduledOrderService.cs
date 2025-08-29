using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Enums;
using Quantra.Models;

namespace Quantra.Services.Interfaces
{
    public interface IScheduledOrderService
    {
        /// <summary>
        /// Splits a large order into smaller chunks
        /// </summary>
        bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes);
        
        /// <summary>
        /// Splits a large order into smaller chunks with advanced options for minimizing market impact
        /// </summary>
        bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes,
            double priceVariancePercent, bool randomizeIntervals, OrderDistributionType distribution);
            
        /// <summary>
        /// Cancels all remaining chunks of a split order group
        /// </summary>
        int CancelSplitOrderGroup(string splitOrderGroupId);
        
        /// <summary>
        /// Gets scheduled orders for a symbol
        /// </summary>
        List<ScheduledOrder> GetScheduledOrders(string symbol);
        
        /// <summary>
        /// Gets all scheduled orders
        /// </summary>
        Dictionary<string, List<ScheduledOrder>> GetAllScheduledOrders();
        
        /// <summary>
        /// Monitors scheduled orders and executes them when due
        /// </summary>
        Task MonitorScheduledOrders();
        
        /// <summary>
        /// Adds a scheduled order
        /// </summary>
        bool AddScheduledOrder(ScheduledOrder order);
        
        /// <summary>
        /// Removes a scheduled order
        /// </summary>
        bool RemoveScheduledOrder(string symbol, string orderId);
        
        /// <summary>
        /// Calculates chunk sizes based on distribution type
        /// </summary>
        List<int> CalculateChunkSizes(int quantity, int chunks, OrderDistributionType distribution);
        
        /// <summary>
        /// Generates random price variance within percentage bounds
        /// </summary>
        double GenerateRandomPriceVariance(double basePrice, double variancePercent);
        
        /// <summary>
        /// Calculates randomized interval if randomization is enabled
        /// </summary>
        int CalculateRandomizedInterval(int baseInterval, bool randomizeIntervals);
        
        /// <summary>
        /// Places multiple orders as part of a strategy
        /// </summary>
        Task<bool> PlaceMultiLegOrder(List<ScheduledOrder> orders);
    }
}