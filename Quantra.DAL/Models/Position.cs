using System;

namespace Quantra.Models
{
    /// <summary>
    /// Represents an options position for Greek calculations
    /// </summary>
    public class Position
    {
        /// <summary>
        /// The underlying asset symbol
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Current price of the underlying asset
        /// </summary>
        public double UnderlyingPrice { get; set; }

        /// <summary>
        /// Option strike price
        /// </summary>
        public double StrikePrice { get; set; }

        /// <summary>
        /// Time to expiration in years
        /// </summary>
        public double TimeToExpiration { get; set; }

        /// <summary>
        /// Risk-free interest rate (e.g., 0.03 for 3%)
        /// </summary>
        public double RiskFreeRate { get; set; }

        /// <summary>
        /// Implied volatility (e.g., 0.20 for 20%)
        /// </summary>
        public double Volatility { get; set; }

        /// <summary>
        /// Option type: true for Call, false for Put
        /// </summary>
        public bool IsCall { get; set; }

        /// <summary>
        /// Number of contracts (positive for long, negative for short)
        /// </summary>
        public int Quantity { get; set; }

        /// <summary>
        /// Current market price of the option
        /// </summary>
        public double OptionPrice { get; set; }

        /// <summary>
        /// Creates a position with specified parameters
        /// </summary>
        public Position()
        {
            RiskFreeRate = 0.03; // Default 3%
            Volatility = 0.20;   // Default 20%
            Quantity = 1;        // Default 1 contract
        }
    }
}