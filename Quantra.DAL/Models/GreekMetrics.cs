using System;

namespace Quantra.Models
{
    /// <summary>
    /// Represents the Greek letter metrics for options trading
    /// </summary>
    public class GreekMetrics
    {
        /// <summary>
        /// Alpha: Excess return generation
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// Beta: Market exposure sensitivity
        /// </summary>
        public double Beta { get; set; }

        /// <summary>
        /// Sigma: Volatility measure
        /// </summary>
        public double Sigma { get; set; }

        /// <summary>
        /// Omega: Advanced risk-return optimization metric
        /// </summary>
        public double Omega { get; set; }

        /// <summary>
        /// Delta: Price sensitivity to underlying asset changes
        /// </summary>
        public double Delta { get; set; }

        /// <summary>
        /// Gamma: Rate of change of Delta with respect to underlying price
        /// </summary>
        public double Gamma { get; set; }

        /// <summary>
        /// Theta: Rate of change of option value with respect to time (time decay)
        /// </summary>
        public double Theta { get; set; }

        /// <summary>
        /// Vega: Sensitivity to changes in implied volatility
        /// </summary>
        public double Vega { get; set; }

        /// <summary>
        /// Rho: Sensitivity to changes in interest rates
        /// </summary>
        public double Rho { get; set; }

        /// <summary>
        /// Timestamp when these metrics were calculated
        /// </summary>
        public DateTime CalculatedAt { get; set; } = DateTime.Now;
    }
}