using System;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.Models
{
    /// <summary>
    /// Represents current market conditions used for rebalancing decisions
    /// </summary>
    public class MarketConditions
    {
        /// <summary>
        /// Current volatility index (e.g., VIX)
        /// </summary>
        public double VolatilityIndex { get; set; } = 15.0; // Default to moderate volatility
        
        /// <summary>
        /// Market trend indicator (-1 to 1 scale, negative = bearish, positive = bullish)
        /// </summary>
        public double MarketTrend { get; set; } = 0.0; // Default to neutral
        
        /// <summary>
        /// Current interest rate environment
        /// </summary>
        public double InterestRate { get; set; } = 0.03; // Default to 3%
        
        /// <summary>
        /// Economic growth indicator (-1 to 1 scale, negative = contracting, positive = expanding)
        /// </summary>
        public double EconomicGrowth { get; set; } = 0.0; // Default to neutral
        
        /// <summary>
        /// List of defensive assets (bonds, utilities, consumer staples, etc.)
        /// </summary>
        public List<string> DefensiveAssets { get; set; } = new List<string>();
        
        /// <summary>
        /// Date these conditions were recorded
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.Now;
        
        /// <summary>
        /// Creates a new market conditions object with default values
        /// </summary>
        public MarketConditions() 
        {
            // Default defensive assets (common ETFs and stock categories)
            DefensiveAssets = new List<string>
            {
                // Bond ETFs
                "AGG", "BND", "GOVT", "IEF", "TLT", "SHY",
                // Utility ETFs/stocks
                "XLU", "VPU", "IDU", "NEE", "DUK", "SO", "D",
                // Consumer Staples
                "XLP", "VDC", "PG", "KO", "WMT", "PEP", "COST",
                // Gold and precious metals
                "GLD", "IAU", "SLV", "PPLT", "GDX"
            };
        }
        
        /// <summary>
        /// Creates market conditions with specified values
        /// </summary>
        public MarketConditions(double volatility, double trend, double rate, double growth) : this()
        {
            VolatilityIndex = volatility;
            MarketTrend = Math.Max(-1.0, Math.Min(1.0, trend));  // Clamp to [-1, 1]
            InterestRate = Math.Max(0.0, rate);                 // Non-negative
            EconomicGrowth = Math.Max(-1.0, Math.Min(1.0, growth)); // Clamp to [-1, 1]
        }
        
        /// <summary>
        /// Determines if a symbol represents a defensive asset
        /// </summary>
        public bool IsDefensiveAsset(string symbol)
        {
            if (string.IsNullOrEmpty(symbol)) return false;
            
            // Check against list of known defensive assets
            if (DefensiveAssets.Contains(symbol.ToUpper())) return true;
            
            // Simple heuristic for sectors based on ticker prefixes (very simplified)
            if (symbol.StartsWith("TLT") || symbol.StartsWith("IEF") || 
                symbol.StartsWith("AGG") || symbol.StartsWith("BND") ||
                symbol.StartsWith("GLD") || symbol.StartsWith("SLV"))
                return true;
            
            return false;
        }
        
        /// <summary>
        /// Market risk assessment on a scale of 0 (low risk) to 1 (high risk)
        /// </summary>
        public double OverallRiskLevel
        {
            get
            {
                // Combine various factors into overall risk assessment
                double volatilityFactor = Math.Min(1.0, VolatilityIndex / 40.0); // VIX above 40 is max risk
                double trendFactor = (1 - MarketTrend) / 2; // Convert [-1, 1] to [1, 0]
                double growthFactor = (1 - EconomicGrowth) / 2; // Convert [-1, 1] to [1, 0]
                
                // Weighted average
                return 0.5 * volatilityFactor + 0.3 * trendFactor + 0.2 * growthFactor;
            }
        }
    }
}