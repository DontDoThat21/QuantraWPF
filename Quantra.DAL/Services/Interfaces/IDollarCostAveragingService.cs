using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    public interface IDollarCostAveragingService
    {
        /// <summary>
        /// Sets up a dollar-cost averaging strategy based on shares
        /// </summary>
        string SetupDollarCostAveraging(string symbol, int totalShares, int numberOfOrders, int intervalDays, 
            string orderType = "BUY", double priceOffset = 0.0);
            
        /// <summary>
        /// Sets up a dollar-cost averaging strategy based on total amount
        /// </summary>
        string SetupDollarCostAveraging(string symbol, double totalAmount, int numberOfOrders, int intervalDays, 
            string orderType = "BUY", double priceOffset = 0.0);
            
        /// <summary>
        /// Pauses a dollar-cost averaging strategy
        /// </summary>
        bool PauseDollarCostAveragingStrategy(string strategyId);
        
        /// <summary>
        /// Resumes a paused dollar-cost averaging strategy
        /// </summary>
        bool ResumeDollarCostAveragingStrategy(string strategyId);
        
        /// <summary>
        /// Cancels a dollar-cost averaging strategy
        /// </summary>
        bool CancelDollarCostAveragingStrategy(string strategyId);
        
        /// <summary>
        /// Gets all active DCA strategies
        /// </summary>
        List<DCAStrategy> GetActiveDCAStrategies();
        
        /// <summary>
        /// Gets DCA strategy information by ID
        /// </summary>
        DCAStrategy GetDCAStrategyInfo(string strategyId);
        
        /// <summary>
        /// Gets DCA strategies for a specific symbol
        /// </summary>
        List<DCAStrategy> GetDCAStrategiesForSymbol(string symbol);
        
        /// <summary>
        /// Schedules the next DCA order for a strategy
        /// </summary>
        bool ScheduleDollarCostAveragingOrder(string strategyId);
        
        /// <summary>
        /// Processes all pending DCA orders
        /// </summary>
        Task ProcessDollarCostAveragingOrders();
    }
}