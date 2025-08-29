using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Enums;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a multi-leg trading strategy combining multiple related orders
    /// </summary>
    public class MultiLegStrategy
    {
        /// <summary>
        /// Unique identifier for the strategy
        /// </summary>
        public string StrategyId { get; set; } = Guid.NewGuid().ToString("N");
        
        /// <summary>
        /// Name or description of the strategy
        /// </summary>
        public string Name { get; set; }
        
        /// <summary>
        /// Type of multi-leg strategy
        /// </summary>
        public MultiLegStrategyType StrategyType { get; set; }
        
        /// <summary>
        /// Collection of orders that make up this strategy
        /// </summary>
        public List<ScheduledOrder> Legs { get; set; } = new List<ScheduledOrder>();
        
        /// <summary>
        /// Whether all legs of the strategy should be executed simultaneously
        /// </summary>
        public bool ExecuteSimultaneously { get; set; } = true;
        
        /// <summary>
        /// Whether the strategy requires all legs to be executed or none
        /// </summary>
        public bool AllOrNone { get; set; } = true;
        
        /// <summary>
        /// Maximum allowed price difference for the entire strategy (used for limit orders)
        /// </summary>
        public double? MaxPriceDifference { get; set; }
        
        /// <summary>
        /// Creation timestamp
        /// </summary>
        public DateTime CreatedTime { get; set; } = DateTime.Now;
        
        /// <summary>
        /// Notes or comments about the strategy
        /// </summary>
        public string Notes { get; set; }
        
        /// <summary>
        /// Risk level associated with this strategy (1-10, with 10 being highest risk)
        /// </summary>
        public int RiskLevel { get; set; }
        
        /// <summary>
        /// Maximum loss if the strategy goes completely against you
        /// </summary>
        public double? MaximumLoss { get; set; }
        
        /// <summary>
        /// Maximum potential profit from the strategy
        /// </summary>
        public double? MaximumProfit { get; set; }
        
        /// <summary>
        /// Validates that the strategy configuration is valid for the specified strategy type
        /// </summary>
        /// <returns>True if the strategy is valid, otherwise false</returns>
        public bool Validate()
        {
            // Basic validation
            if (Legs == null || Legs.Count == 0)
            {
                return false;
            }
            
            // Strategy-specific validation
            switch (StrategyType)
            {
                case MultiLegStrategyType.VerticalSpread:
                    return ValidateVerticalSpread();
                    
                case MultiLegStrategyType.Straddle:
                    return ValidateStraddle();
                    
                case MultiLegStrategyType.Strangle:
                    return ValidateStrangle();
                    
                case MultiLegStrategyType.IronCondor:
                    return ValidateIronCondor();
                    
                case MultiLegStrategyType.ButterflySpread:
                    return ValidateButterflySpread();
                    
                case MultiLegStrategyType.PairsTrade:
                    return ValidatePairsTrade();
                    
                case MultiLegStrategyType.CoveredCall:
                    return ValidateCoveredCall();
                    
                case MultiLegStrategyType.BasketOrder:
                    // Basket orders are flexible in composition
                    return Legs.Count > 0;
                    
                case MultiLegStrategyType.Custom:
                default:
                    // Custom strategies don't have specific validation requirements
                    return Legs.Count > 0;
            }
        }
        
        /// <summary>
        /// Calculate the total cost/credit of the strategy
        /// </summary>
        /// <returns>Net cost/credit of the strategy (negative values indicate a credit)</returns>
        public double CalculateNetCost()
        {
            if (Legs == null || Legs.Count == 0)
            {
                return 0;
            }
            
            double netCost = 0;
            
            foreach (var leg in Legs)
            {
                if (leg.OrderType.Equals("BUY", StringComparison.OrdinalIgnoreCase))
                {
                    netCost += leg.Price * leg.Quantity;
                }
                else if (leg.OrderType.Equals("SELL", StringComparison.OrdinalIgnoreCase))
                {
                    netCost -= leg.Price * leg.Quantity;
                }
            }
            
            return netCost;
        }
        
        #region Strategy Validation Methods
        
        private bool ValidateVerticalSpread()
        {
            // Vertical spread needs exactly 2 legs
            if (Legs.Count != 2)
            {
                return false;
            }
            
            // Both legs must be for the same symbol
            var symbol = Legs[0].Symbol;
            if (Legs.Any(leg => leg.Symbol != symbol))
            {
                return false;
            }
            
            // One leg must be BUY and one must be SELL
            bool hasBuy = Legs.Any(leg => leg.OrderType.Equals("BUY", StringComparison.OrdinalIgnoreCase));
            bool hasSell = Legs.Any(leg => leg.OrderType.Equals("SELL", StringComparison.OrdinalIgnoreCase));
            
            return hasBuy && hasSell;
        }
        
        private bool ValidateStraddle()
        {
            // Straddle needs exactly 2 legs
            if (Legs.Count != 2)
            {
                return false;
            }
            
            // Both legs must be for the same symbol
            var symbol = Legs[0].Symbol;
            if (Legs.Any(leg => leg.Symbol != symbol))
            {
                return false;
            }
            
            return true;
        }
        
        private bool ValidateStrangle()
        {
            // Strangle needs exactly 2 legs
            if (Legs.Count != 2)
            {
                return false;
            }
            
            // Both legs must be for the same symbol
            var symbol = Legs[0].Symbol;
            if (Legs.Any(leg => leg.Symbol != symbol))
            {
                return false;
            }
            
            return true;
        }
        
        private bool ValidateIronCondor()
        {
            // Iron Condor needs exactly 4 legs
            if (Legs.Count != 4)
            {
                return false;
            }
            
            // All legs must be for the same symbol
            var symbol = Legs[0].Symbol;
            if (Legs.Any(leg => leg.Symbol != symbol))
            {
                return false;
            }
            
            // Must have 2 BUY and 2 SELL orders
            int buyCount = Legs.Count(leg => leg.OrderType.Equals("BUY", StringComparison.OrdinalIgnoreCase));
            int sellCount = Legs.Count(leg => leg.OrderType.Equals("SELL", StringComparison.OrdinalIgnoreCase));
            
            return buyCount == 2 && sellCount == 2;
        }
        
        private bool ValidateButterflySpread()
        {
            // Butterfly needs exactly 3 legs
            if (Legs.Count != 3)
            {
                return false;
            }
            
            // All legs must be for the same symbol
            var symbol = Legs[0].Symbol;
            if (Legs.Any(leg => leg.Symbol != symbol))
            {
                return false;
            }
            
            return true;
        }
        
        private bool ValidatePairsTrade()
        {
            // Pairs trade needs exactly 2 legs
            if (Legs.Count != 2)
            {
                return false;
            }
            
            // Symbols should be different
            if (Legs[0].Symbol == Legs[1].Symbol)
            {
                return false;
            }
            
            // One leg must be BUY and one must be SELL
            bool hasBuy = Legs.Any(leg => leg.OrderType.Equals("BUY", StringComparison.OrdinalIgnoreCase));
            bool hasSell = Legs.Any(leg => leg.OrderType.Equals("SELL", StringComparison.OrdinalIgnoreCase));
            
            return hasBuy && hasSell;
        }
        
        private bool ValidateCoveredCall()
        {
            // Covered Call needs exactly 2 legs
            if (Legs.Count != 2)
            {
                return false;
            }
            
            // Both legs must be for the same symbol
            var symbol = Legs[0].Symbol;
            if (Legs.Any(leg => leg.Symbol != symbol))
            {
                return false;
            }
            
            // One leg must be BUY (stock) and one must be SELL (call)
            bool hasBuy = Legs.Any(leg => leg.OrderType.Equals("BUY", StringComparison.OrdinalIgnoreCase));
            bool hasSell = Legs.Any(leg => leg.OrderType.Equals("SELL", StringComparison.OrdinalIgnoreCase));
            
            return hasBuy && hasSell;
        }
        
        #endregion
    }
}