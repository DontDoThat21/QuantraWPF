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
        /// Alpha Vantage contract identifier
        /// </summary>
        public string ContractId { get; set; }

        /// <summary>
        /// Black-Scholes calculated theoretical price
        /// </summary>
        public double TheoreticalPrice { get; set; }

        /// <summary>
        /// IV rank vs. 52-week range (0-100 percentile)
        /// </summary>
        public double IVPercentile { get; set; }

        /// <summary>
        /// Gets the mid price (average of bid and ask)
        /// </summary>
        public double MidPrice => (Bid + Ask) / 2.0;

        /// <summary>
        /// Gets the option symbol representation
        /// </summary>
        public string OptionSymbol => $"{UnderlyingSymbol}{ExpirationDate:yyMMdd}{OptionType[0]}{StrikePrice:00000000}";

        /// <summary>
        /// Days until expiration
        /// </summary>
        public double DaysToExpiration => (ExpirationDate - DateTime.Now).TotalDays;

        /// <summary>
        /// Years until expiration (for calculations)
        /// </summary>
        public double TimeToExpiration => DaysToExpiration / 365.0;

        /// <summary>
        /// Whether this option is in-the-money
        /// </summary>
        public bool IsITM
        {
            get
            {
                if (OptionType?.ToUpper() == "CALL")
                    return LastPrice > StrikePrice;
                else
                    return LastPrice < StrikePrice;
            }
        }

        /// <summary>
        /// Whether this option is at-the-money (within 2% of strike)
        /// </summary>
        public bool IsATM
        {
            get
            {
                if (LastPrice == 0 || StrikePrice == 0)
                    return false;
                    
                double percentDiff = Math.Abs((LastPrice - StrikePrice) / StrikePrice);
                return percentDiff <= 0.02;
            }
        }

        /// <summary>
        /// Whether this option is out-of-the-money
        /// </summary>
        public bool IsOTM => !IsITM && !IsATM;

        /// <summary>
        /// Intrinsic value of the option
        /// </summary>
        public double IntrinsicValue
        {
            get
            {
                if (OptionType?.ToUpper() == "CALL")
                    return Math.Max(0, LastPrice - StrikePrice);
                else
                    return Math.Max(0, StrikePrice - LastPrice);
            }
        }

        /// <summary>
        /// Intrinsic value property alias for compatibility
        /// </summary>
        public double Intrinsic => IntrinsicValue;

        /// <summary>
        /// Time value (extrinsic value) of the option
        /// </summary>
        public double TimeValue => Math.Max(0, MidPrice - IntrinsicValue);

        /// <summary>
        /// Extrinsic value property alias for compatibility
        /// </summary>
        public double Extrinsic => TimeValue;

        /// <summary>
        /// In-the-money property alias for compatibility
        /// </summary>
        public bool InTheMoney => IsITM;

        /// <summary>
        /// Bid-ask spread
        /// </summary>
        public double Spread => Ask - Bid;

        /// <summary>
        /// Bid-ask spread as percentage of mid price
        /// </summary>
        public double SpreadPercent => MidPrice > 0 ? (Spread / MidPrice) * 100 : 0;
    }
}