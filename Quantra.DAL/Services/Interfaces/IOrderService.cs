using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using Quantra.Enums;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    public interface IOrderService
    {
        /// <summary>
        /// Places a limit order for the specified symbol
        /// </summary>
        Task<bool> PlaceLimitOrder(string symbol, int quantity, string orderType, double price);
        
        /// <summary>
        /// Gets the current market price for a symbol
        /// </summary>
        Task<double> GetMarketPrice(string symbol);
        
        /// <summary>
        /// Sets the trading mode (paper or real)
        /// </summary>
        void SetTradingMode(TradingMode mode);

        /// <summary>
        /// Gets the user trading settings
        /// </summary>
        bool GetApiModalCheckSetting();

        /// <summary>
        /// Saves the user trading settings
        /// </summary>
        void SaveApiModalCheckSetting(bool enableChecks);

        /// <summary>
        /// Loads order history from database
        /// </summary>
        ObservableCollection<OrderModel> LoadOrderHistory();
        
        /// <summary>
        /// Creates a new order model with default values
        /// </summary>
        OrderModel CreateDefaultOrder();
        
        /// <summary>
        /// Places a bracket order with stop loss and take profit
        /// </summary>
        Task<bool> PlaceBracketOrder(string symbol, int quantity, string orderType, double price, double stopLossPrice, double takeProfitPrice);
        
        /// <summary>
        /// Sets a trailing stop for a position
        /// </summary>
        bool SetTrailingStop(string symbol, double initialPrice, double trailingDistance);
        
        /// <summary>
        /// Sets a time-based exit for a position
        /// </summary>
        bool SetTimeBasedExit(string symbol, DateTime exitTime);
        
        /// <summary>
        /// Calculates position size based on risk parameters
        /// </summary>
        int CalculatePositionSizeByRisk(string symbol, double price, double stopLossPrice, double riskPercentage, double accountSize);
        
        /// <summary>
        /// Sets up a dollar-cost averaging strategy
        /// </summary>
        bool SetupDollarCostAveraging(string symbol, int totalShares, int numberOfOrders, int intervalDays);
        
        /// <summary>
        /// Sets portfolio target allocations
        /// </summary>
        bool SetPortfolioAllocations(Dictionary<string, double> allocations);
        
        /// <summary>
        /// Rebalances portfolio to match target allocations
        /// </summary>
        Task<bool> RebalancePortfolio(double tolerancePercentage = 0.02);
        
        /// <summary>
        /// Places multiple orders as part of a strategy
        /// </summary>
        Task<bool> PlaceMultiLegOrder(List<ScheduledOrder> orders);
        
        /// <summary>
        /// Splits a large order into smaller chunks
        /// </summary>
        bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes);
        
        /// <summary>
        /// Splits a large order into smaller chunks with advanced options for minimizing market impact
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="quantity">Total quantity to trade</param>
        /// <param name="orderType">BUY or SELL</param>
        /// <param name="price">Base limit price</param>
        /// <param name="chunks">Number of chunks to split into</param>
        /// <param name="intervalMinutes">Base minutes between each chunk</param>
        /// <param name="priceVariancePercent">Percentage to vary price between chunks (0-5%)</param>
        /// <param name="randomizeIntervals">Whether to randomize time intervals between chunks</param>
        /// <param name="distribution">How to distribute quantity across chunks</param>
        bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes,
            double priceVariancePercent, bool randomizeIntervals, OrderDistributionType distribution);
            
        /// <summary>
        /// Cancels all remaining chunks of a split order group
        /// </summary>
        int CancelSplitOrderGroup(string splitOrderGroupId);
        
        /// <summary>
        /// Activates emergency stop to halt all trading
        /// </summary>
        bool ActivateEmergencyStop();
        
        /// <summary>
        /// Deactivates emergency stop to resume trading
        /// </summary>
        bool DeactivateEmergencyStop();
        
        /// <summary>
        /// Checks if emergency stop is currently active
        /// </summary>
        bool IsEmergencyStopActive();
        
        /// <summary>
        /// Sets trading hour restrictions
        /// </summary>
        bool SetTradingHourRestrictions(TimeOnly marketOpen, TimeOnly marketClose);
        
        /// <summary>
        /// Sets which market sessions are enabled for trading
        /// </summary>
        bool SetEnabledMarketSessions(MarketSession sessions);
        
        /// <summary>
        /// Gets which market sessions are currently enabled for trading
        /// </summary>
        MarketSession GetEnabledMarketSessions();
        
        /// <summary>
        /// Sets the time boundaries for different market sessions
        /// </summary>
        /// <param name="preMarketOpenTime">Pre-market opening time (default: 4:00 AM)</param>
        /// <param name="regularMarketOpenTime">Regular market opening time (default: 9:30 AM)</param>
        /// <param name="regularMarketCloseTime">Regular market closing time (default: 4:00 PM)</param>
        /// <param name="afterHoursCloseTime">After-hours closing time (default: 8:00 PM)</param>
        /// <returns>True if session times were set successfully</returns>
        bool SetMarketSessionTimes(TimeOnly preMarketOpenTime, TimeOnly regularMarketOpenTime, 
            TimeOnly regularMarketCloseTime, TimeOnly afterHoursCloseTime);
            
        /// <summary>
        /// Gets the current market session time boundaries
        /// </summary>
        /// <returns>A tuple containing the session time boundaries (preMarketOpen, regularMarketOpen, regularMarketClose, afterHoursClose)</returns>
        (TimeOnly preMarketOpen, TimeOnly regularMarketOpen, TimeOnly regularMarketClose, TimeOnly afterHoursClose) GetMarketSessionTimes();
        
        /// <summary>
        /// Checks if trading is currently allowed
        /// </summary>
        bool IsTradingAllowed();
    }
}
