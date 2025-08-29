using System;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a single option contract data
    /// </summary>
    public class OptionData
    {
        /// <summary>
        /// The underlying symbol (e.g., AAPL)
        /// </summary>
        public string UnderlyingSymbol { get; set; }

        /// <summary>
        /// Strike price of the option
        /// </summary>
        public double StrikePrice { get; set; }

        /// <summary>
        /// Expiration date of the option
        /// </summary>
        public DateTime ExpirationDate { get; set; }

        /// <summary>
        /// Option type (CALL or PUT)
        /// </summary>
        public string OptionType { get; set; }

        /// <summary>
        /// Current bid price
        /// </summary>
        public double Bid { get; set; }

        /// <summary>
        /// Current ask price
        /// </summary>
        public double Ask { get; set; }

        /// <summary>
        /// Last traded price
        /// </summary>
        public double LastPrice { get; set; }

        /// <summary>
        /// Implied volatility
        /// </summary>
        public double ImpliedVolatility { get; set; }

        /// <summary>
        /// Option delta
        /// </summary>
        public double Delta { get; set; }

        /// <summary>
        /// Option gamma
        /// </summary>
        public double Gamma { get; set; }

        /// <summary>
        /// Option theta
        /// </summary>
        public double Theta { get; set; }

        /// <summary>
        /// Option vega
        /// </summary>
        public double Vega { get; set; }

        /// <summary>
        /// Option rho
        /// </summary>
        public double Rho { get; set; }

        /// <summary>
        /// Volume traded
        /// </summary>
        public long Volume { get; set; }

        /// <summary>
        /// Open interest
        /// </summary>
        public long OpenInterest { get; set; }

        /// <summary>
        /// Timestamp when this option data was fetched
        /// </summary>
        public DateTime? FetchTimestamp { get; set; }

        /// <summary>
        /// Gets the mid price (average of bid and ask)
        /// </summary>
        public double MidPrice => (Bid + Ask) / 2.0;

        /// <summary>
        /// Gets the option symbol representation
        /// </summary>
        public string OptionSymbol => $"{UnderlyingSymbol}{ExpirationDate:yyMMdd}{OptionType[0]}{StrikePrice:00000000}";
    }
}