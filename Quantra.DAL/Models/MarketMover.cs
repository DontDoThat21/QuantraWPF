using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a market mover (gainer, loser, or most actively traded)
    /// from the Alpha Vantage TOP_GAINERS_LOSERS API
    /// </summary>
    public class MarketMover
    {
        /// <summary>
        /// Stock ticker symbol
        /// </summary>
        public string Ticker { get; set; }

        /// <summary>
        /// Current price
        /// </summary>
        public double Price { get; set; }

        /// <summary>
        /// Price change amount
        /// </summary>
        public double ChangeAmount { get; set; }

        /// <summary>
        /// Percentage change
        /// </summary>
        public double ChangePercentage { get; set; }

        /// <summary>
        /// Trading volume
        /// </summary>
        public long Volume { get; set; }

        /// <summary>
        /// Category: Gainer, Loser, or MostActive
        /// </summary>
        public MarketMoverCategory Category { get; set; }
    }

    /// <summary>
    /// Category of market mover
    /// </summary>
    public enum MarketMoverCategory
    {
        Gainer,
        Loser,
        MostActive
    }

    /// <summary>
    /// Response container for top gainers/losers API
    /// </summary>
    public class TopMoversResponse
    {
        /// <summary>
        /// Metadata string
        /// </summary>
        public string Metadata { get; set; }

        /// <summary>
        /// Last updated timestamp
        /// </summary>
        public DateTime LastUpdated { get; set; }

        /// <summary>
        /// List of top gainers
        /// </summary>
        public List<MarketMover> TopGainers { get; set; } = new List<MarketMover>();

        /// <summary>
        /// List of top losers
        /// </summary>
        public List<MarketMover> TopLosers { get; set; } = new List<MarketMover>();

        /// <summary>
        /// List of most actively traded stocks
        /// </summary>
        public List<MarketMover> MostActivelyTraded { get; set; } = new List<MarketMover>();
    }
}
