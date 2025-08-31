using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    public interface IRiskManagementService
    {
        /// <summary>
        /// Sets a trailing stop for a position
        /// </summary>
        bool SetTrailingStop(string symbol, double initialPrice, double trailingDistance, string orderType = "SELL");
        
        /// <summary>
        /// Sets a time-based exit for a position
        /// </summary>
        bool SetTimeBasedExit(string symbol, DateTime exitTime);
        
        /// <summary>
        /// Sets a time-based exit with strategy type
        /// </summary>
        bool SetTimeBasedExit(string symbol, DateTime exitTime, TimeBasedExit exitStrategy);
        
        /// <summary>
        /// Places a bracket order with stop loss and take profit
        /// </summary>
        Task<bool> PlaceBracketOrder(string symbol, int quantity, string orderType, double price, double stopLossPrice, double takeProfitPrice);
        
        /// <summary>
        /// Monitors trailing stops and adjusts them based on price movements
        /// </summary>
        Task MonitorTrailingStops();
        
        /// <summary>
        /// Monitors bracket orders (stop loss and take profit)
        /// </summary>
        Task MonitorBracketOrders();
        
        /// <summary>
        /// Monitors time-based exits
        /// </summary>
        Task MonitorTimeBasedExits();
        
        /// <summary>
        /// Gets all active trailing stops
        /// </summary>
        Dictionary<string, TrailingStopInfo> GetActiveTrailingStops();
        
        /// <summary>
        /// Gets all active time-based exits
        /// </summary>
        Dictionary<string, TimeBasedExit> GetActiveTimeBasedExits();
        
        /// <summary>
        /// Removes a trailing stop for a symbol
        /// </summary>
        bool RemoveTrailingStop(string symbol);
        
        /// <summary>
        /// Removes a time-based exit for a symbol
        /// </summary>
        bool RemoveTimeBasedExit(string symbol);
        
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
    }
}