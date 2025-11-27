using System;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a symbol search result from Alpha Vantage API
    /// </summary>
    public class SymbolSearchResult
    {
        /// <summary>
        /// Stock symbol (e.g., "BA", "AAPL")
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Company name (e.g., "Boeing Company", "Apple Inc.")
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Security type (e.g., "Equity", "ETF")
        /// </summary>
        public string Type { get; set; }

        /// <summary>
        /// Region (e.g., "United States")
        /// </summary>
        public string Region { get; set; }

        /// <summary>
        /// Market opening time
        /// </summary>
        public string MarketOpen { get; set; }

        /// <summary>
        /// Market closing time
        /// </summary>
        public string MarketClose { get; set; }

        /// <summary>
        /// Timezone (e.g., "UTC-04")
        /// </summary>
        public string Timezone { get; set; }

        /// <summary>
        /// Currency (e.g., "USD")
        /// </summary>
        public string Currency { get; set; }

        /// <summary>
        /// Match score (0-1, where 1 is exact match)
        /// </summary>
        public double MatchScore { get; set; }

        /// <summary>
        /// Display text for ComboBox (Symbol - Name)
        /// </summary>
        public string DisplayText => $"{Symbol} - {Name}";

        /// <summary>
        /// Override ToString for debugging
        /// </summary>
        public override string ToString()
        {
            return DisplayText;
        }
    }
}
