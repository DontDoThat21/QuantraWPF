using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Filter criteria for options chain queries
    /// </summary>
    public class OptionsChainFilter
    {
        /// <summary>
        /// Underlying symbol to filter
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Specific expiration date to filter (null for all expirations)
        /// </summary>
        public DateTime? SelectedExpiration { get; set; }

        /// <summary>
        /// Option types to include: "CALL", "PUT", or both
        /// </summary>
        public List<string> OptionTypes { get; set; } = new List<string> { "CALL", "PUT" };

        /// <summary>
        /// Minimum strike price
        /// </summary>
        public double? MinStrike { get; set; }

        /// <summary>
        /// Maximum strike price
        /// </summary>
        public double? MaxStrike { get; set; }

        /// <summary>
        /// Filter to only in-the-money options
        /// </summary>
        public bool OnlyITM { get; set; }

        /// <summary>
        /// Filter to only liquid options (with minimum volume/open interest)
        /// </summary>
        public bool OnlyLiquid { get; set; }

        /// <summary>
        /// Minimum volume threshold for liquid options
        /// </summary>
        public long MinVolume { get; set; } = 100;

        /// <summary>
        /// Minimum open interest threshold for liquid options
        /// </summary>
        public long MinOpenInterest { get; set; } = 50;

        /// <summary>
        /// Minimum days to expiration
        /// </summary>
        public int? MinDTE { get; set; }

        /// <summary>
        /// Maximum days to expiration
        /// </summary>
        public int? MaxDTE { get; set; }

        /// <summary>
        /// Minimum implied volatility
        /// </summary>
        public double? MinIV { get; set; }

        /// <summary>
        /// Maximum implied volatility
        /// </summary>
        public double? MaxIV { get; set; }

        /// <summary>
        /// Creates a default filter for a symbol
        /// </summary>
        public OptionsChainFilter()
        {
        }

        /// <summary>
        /// Creates a filter with basic symbol and expiration
        /// </summary>
        public OptionsChainFilter(string symbol, DateTime? expiration = null)
        {
            Symbol = symbol;
            SelectedExpiration = expiration;
        }

        /// <summary>
        /// Applies the filter to an option contract
        /// </summary>
        /// <param name="option">Option to check</param>
        /// <param name="currentPrice">Current underlying price for ITM calculation</param>
        /// <returns>True if option passes all filters</returns>
        public bool PassesFilter(OptionData option, double currentPrice)
        {
            if (option == null)
                return false;

            // Symbol filter
            if (!string.IsNullOrEmpty(Symbol) && 
                !option.UnderlyingSymbol.Equals(Symbol, StringComparison.OrdinalIgnoreCase))
                return false;

            // Expiration filter
            if (SelectedExpiration.HasValue && option.ExpirationDate.Date != SelectedExpiration.Value.Date)
                return false;

            // Option type filter
            if (OptionTypes != null && OptionTypes.Count > 0 && 
                !OptionTypes.Contains(option.OptionType?.ToUpper()))
                return false;

            // Strike range filter
            if (MinStrike.HasValue && option.StrikePrice < MinStrike.Value)
                return false;

            if (MaxStrike.HasValue && option.StrikePrice > MaxStrike.Value)
                return false;

            // ITM filter
            if (OnlyITM && !option.IsITM)
                return false;

            // Liquidity filter
            if (OnlyLiquid && (option.Volume < MinVolume || option.OpenInterest < MinOpenInterest))
                return false;

            // Days to expiration filter
            var dte = (int)option.DaysToExpiration;
            if (MinDTE.HasValue && dte < MinDTE.Value)
                return false;

            if (MaxDTE.HasValue && dte > MaxDTE.Value)
                return false;

            // IV filter
            if (MinIV.HasValue && option.ImpliedVolatility < MinIV.Value)
                return false;

            if (MaxIV.HasValue && option.ImpliedVolatility > MaxIV.Value)
                return false;

            return true;
        }

        /// <summary>
        /// Creates a filter for near-term options (30 days or less)
        /// </summary>
        public static OptionsChainFilter CreateNearTermFilter(string symbol)
        {
            return new OptionsChainFilter(symbol)
            {
                MaxDTE = 30,
                OnlyLiquid = true
            };
        }

        /// <summary>
        /// Creates a filter for LEAPS (long-term options, 9+ months)
        /// </summary>
        public static OptionsChainFilter CreateLEAPSFilter(string symbol)
        {
            return new OptionsChainFilter(symbol)
            {
                MinDTE = 270 // 9 months
            };
        }

        /// <summary>
        /// Creates a filter for at-the-money options only
        /// </summary>
        public static OptionsChainFilter CreateATMFilter(string symbol, double currentPrice, double atmRange = 0.02)
        {
            var minStrike = currentPrice * (1 - atmRange);
            var maxStrike = currentPrice * (1 + atmRange);

            return new OptionsChainFilter(symbol)
            {
                MinStrike = minStrike,
                MaxStrike = maxStrike,
                OnlyLiquid = true
            };
        }
    }
}
