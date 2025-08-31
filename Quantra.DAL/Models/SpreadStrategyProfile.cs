using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a spread trading strategy profile for backtesting option spread strategies
    /// </summary>
    public class SpreadStrategyProfile : StrategyProfile
    {
        /// <summary>
        /// The spread configuration defining the option legs
        /// </summary>
        public SpreadConfiguration SpreadConfig { get; set; }

        /// <summary>
        /// Minimum time to expiration in days before closing position
        /// </summary>
        public int MinDaysToExpiration { get; set; } = 7;

        /// <summary>
        /// Target profit percentage for early exit (e.g., 0.5 for 50% of max profit)
        /// </summary>
        public double TargetProfitPercentage { get; set; } = 0.5;

        /// <summary>
        /// Stop loss percentage (e.g., -2.0 for 200% of premium paid)
        /// </summary>
        public double StopLossPercentage { get; set; } = -2.0;

        /// <summary>
        /// Risk-free rate for option pricing
        /// </summary>
        public double RiskFreeRate { get; set; } = 0.02; // 2% default

        /// <summary>
        /// Constructor for spread strategy profile
        /// </summary>
        /// <param name="spreadConfig">The spread configuration</param>
        /// <param name="name">Strategy name</param>
        public SpreadStrategyProfile(SpreadConfiguration spreadConfig, string name = null)
        {
            SpreadConfig = spreadConfig ?? throw new ArgumentNullException(nameof(spreadConfig));
            Name = name ?? $"{spreadConfig.SpreadType} Strategy";
        }

        /// <summary>
        /// Returns technical indicators required by this spread strategy
        /// </summary>
        public override IEnumerable<string> RequiredIndicators => new[] { "PRICE", "VOLATILITY", "VOLUME" };

        /// <summary>
        /// Generates trading signals for spread strategies
        /// For spread strategies, signals are based on entry/exit conditions rather than direction
        /// </summary>
        /// <param name="historical">Historical price data</param>
        /// <param name="index">Current index in historical data</param>
        /// <returns>Trading signal: ENTER, EXIT, or null</returns>
        public override string GenerateSignal(List<HistoricalPrice> historical, int? index = null)
        {
            int idx = index ?? (historical.Count - 1);
            if (idx < 20) return null; // Need sufficient history

            var currentPrice = historical[idx];
            
            // Simple entry conditions based on spread type
            // In a real implementation, this would be more sophisticated
            switch (SpreadConfig.SpreadType)
            {
                case Enums.MultiLegStrategyType.VerticalSpread:
                    return GenerateVerticalSpreadSignal(historical, idx);
                    
                case Enums.MultiLegStrategyType.Straddle:
                    return GenerateStraddleSignal(historical, idx);
                    
                case Enums.MultiLegStrategyType.Strangle:
                    return GenerateStrangleSignal(historical, idx);
                    
                case Enums.MultiLegStrategyType.IronCondor:
                    return GenerateIronCondorSignal(historical, idx);
                    
                default:
                    return GenerateGenericSpreadSignal(historical, idx);
            }
        }

        /// <summary>
        /// Generate signal for vertical spreads (bull call/put, bear call/put)
        /// </summary>
        private string GenerateVerticalSpreadSignal(List<HistoricalPrice> historical, int index)
        {
            // Look for trending conditions - vertical spreads work well in trending markets
            var recentPrices = historical.Skip(Math.Max(0, index - 9)).Take(10).Select(h => h.Close).ToList();
            if (recentPrices.Count < 10) return null;

            var trend = CalculateTrend(recentPrices);
            var volatility = CalculateVolatility(recentPrices);

            // Enter when we have a clear trend and moderate volatility
            if (Math.Abs(trend) > 0.02 && volatility < 0.3) // 2% trend, less than 30% volatility
            {
                return "ENTER";
            }

            return null;
        }

        /// <summary>
        /// Generate signal for straddles (long/short straddle)
        /// </summary>
        private string GenerateStraddleSignal(List<HistoricalPrice> historical, int index)
        {
            // Straddles work well when expecting high volatility
            var recentPrices = historical.Skip(Math.Max(0, index - 19)).Take(20).Select(h => h.Close).ToList();
            if (recentPrices.Count < 20) return null;

            var currentVolatility = CalculateVolatility(recentPrices.Skip(10).Take(10).ToList());
            var pastVolatility = CalculateVolatility(recentPrices.Take(10).ToList());

            // Enter long straddle when volatility is low but expected to increase
            if (currentVolatility < 0.15 && pastVolatility > currentVolatility * 1.5)
            {
                return "ENTER";
            }

            return null;
        }

        /// <summary>
        /// Generate signal for strangles (long/short strangle)
        /// </summary>
        private string GenerateStrangleSignal(List<HistoricalPrice> historical, int index)
        {
            // Strangles involve different strike prices and work well in moderately volatile markets
            var recentPrices = historical.Skip(Math.Max(0, index - 19)).Take(20).Select(h => h.Close).ToList();
            if (recentPrices.Count < 20) return null;

            var currentVolatility = CalculateVolatility(recentPrices.Skip(10).Take(10).ToList());
            var pastVolatility = CalculateVolatility(recentPrices.Take(10).ToList());

            // Enter long strangle when volatility is moderate and expected to increase
            if (currentVolatility > 0.15 && currentVolatility < 0.3 && pastVolatility > currentVolatility * 1.2)
            {
                return "ENTER";
            }

            return null;
        }

        /// <summary>
        /// Generate signal for iron condors
        /// </summary>
        private string GenerateIronCondorSignal(List<HistoricalPrice> historical, int index)
        {
            // Iron condors work well in range-bound, low volatility markets
            var recentPrices = historical.Skip(Math.Max(0, index - 19)).Take(20).Select(h => h.Close).ToList();
            if (recentPrices.Count < 20) return null;

            var volatility = CalculateVolatility(recentPrices);
            var range = (recentPrices.Max() - recentPrices.Min()) / recentPrices.Average();

            // Enter when volatility is low and price is range-bound
            if (volatility < 0.15 && range < 0.1) // Low vol and tight range
            {
                return "ENTER";
            }

            return null;
        }

        /// <summary>
        /// Generic spread signal generation
        /// </summary>
        private string GenerateGenericSpreadSignal(List<HistoricalPrice> historical, int index)
        {
            // Default to moderate volatility entry conditions
            var recentPrices = historical.Skip(Math.Max(0, index - 9)).Take(10).Select(h => h.Close).ToList();
            if (recentPrices.Count < 10) return null;

            var volatility = CalculateVolatility(recentPrices);
            
            // Enter when volatility is in moderate range
            if (volatility > 0.1 && volatility < 0.4)
            {
                return "ENTER";
            }

            return null;
        }

        /// <summary>
        /// Calculate trend from price series (positive = uptrend, negative = downtrend)
        /// </summary>
        private double CalculateTrend(List<double> prices)
        {
            if (prices.Count < 2) return 0;
            return (prices.Last() - prices.First()) / prices.First();
        }

        /// <summary>
        /// Calculate volatility from price series using standard deviation of returns
        /// </summary>
        private double CalculateVolatility(List<double> prices)
        {
            if (prices.Count < 2) return 0;

            var returns = new List<double>();
            for (int i = 1; i < prices.Count; i++)
            {
                returns.Add((prices[i] - prices[i - 1]) / prices[i - 1]);
            }

            if (returns.Count == 0) return 0;

            var mean = returns.Average();
            var variance = returns.Sum(r => Math.Pow(r - mean, 2)) / returns.Count;
            return Math.Sqrt(variance) * Math.Sqrt(252); // Annualized volatility
        }

        /// <summary>
        /// Validate if the current market conditions meet the spread strategy criteria
        /// </summary>
        /// <param name="indicators">Dictionary of indicator values</param>
        /// <returns>True if conditions are met, false otherwise</returns>
        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            if (indicators == null || !indicators.Any())
                return false;

            // Check for required indicators for spread strategies
            if (!indicators.TryGetValue("PRICE", out double price) || price <= 0)
                return false;

            // Optional but recommended indicators for spread strategies
            bool hasVolatility = indicators.TryGetValue("VOLATILITY", out double volatility);
            bool hasVolume = indicators.TryGetValue("VOLUME", out double volume);

            // Basic validation - we need at least price data
            // Additional validation based on spread type
            switch (SpreadConfig?.SpreadType)
            {
                case Enums.MultiLegStrategyType.IronCondor:
                    // Iron condors work best in low volatility environments
                    return !hasVolatility || volatility < 0.25;
                    
                case Enums.MultiLegStrategyType.Straddle:
                case Enums.MultiLegStrategyType.Strangle:
                    // Straddles and strangles benefit from volatility information
                    return !hasVolatility || volatility > 0.1;
                    
                default:
                    // For other spread types, basic price validation is sufficient
                    return true;
            }
        }
    }
}