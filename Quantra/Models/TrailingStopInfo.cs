using System;

namespace Quantra.Models
{
    /// <summary>
    /// Represents information about a trailing stop order
    /// </summary>
    public class TrailingStopInfo
    {
        /// <summary>
        /// The symbol for the security
        /// </summary>
        public string Symbol { get; set; }
        
        /// <summary>
        /// Initial price when the trailing stop was set
        /// </summary>
        public double InitialPrice { get; set; }
        
        /// <summary>
        /// Current trigger price for the stop loss
        /// </summary>
        public double CurrentStopPrice { get; set; }
        
        /// <summary>
        /// Distance for the trailing stop, as a percentage (e.g., 0.05 for 5%)
        /// </summary>
        public double TrailingDistance { get; set; }
        
        /// <summary>
        /// Highest price reached since the trailing stop was set
        /// </summary>
        public double HighestPrice { get; set; }
        
        /// <summary>
        /// Date and time when the trailing stop was created
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.Now;
        
        /// <summary>
        /// Date and time when the trailing stop was last updated
        /// </summary>
        public DateTime LastUpdatedAt { get; set; } = DateTime.Now;
        
        /// <summary>
        /// Creates a new trailing stop info object
        /// </summary>
        public TrailingStopInfo() { }
        
        /// <summary>
        /// Creates a new trailing stop info object with initial values
        /// </summary>
        /// <param name="symbol">The security symbol</param>
        /// <param name="initialPrice">The initial price</param>
        /// <param name="trailingDistance">The trailing distance as a percentage (e.g., 0.05 for 5%)</param>
        public TrailingStopInfo(string symbol, double initialPrice, double trailingDistance)
        {
            Symbol = symbol;
            InitialPrice = initialPrice;
            HighestPrice = initialPrice;
            TrailingDistance = trailingDistance;
            
            // Initial stop price is calculated as initialPrice * (1 - trailingDistance) for long positions
            CurrentStopPrice = initialPrice * (1 - trailingDistance);
        }
        
        /// <summary>
        /// Updates the trailing stop based on a new price
        /// </summary>
        /// <param name="currentPrice">The current price of the security</param>
        /// <returns>True if the stop has been triggered (current price <= stop price), otherwise false</returns>
        public bool UpdateStopPrice(double currentPrice)
        {
            LastUpdatedAt = DateTime.Now;
            
            // Check if stop is triggered
            if (currentPrice <= CurrentStopPrice)
            {
                return true;
            }
            
            // Update highest price if current price is higher
            if (currentPrice > HighestPrice)
            {
                HighestPrice = currentPrice;
                
                // Update stop price based on new highest price
                CurrentStopPrice = HighestPrice * (1 - TrailingDistance);
            }
            
            return false;
        }
    }
}