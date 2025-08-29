using System;
using System.Collections.Generic;
using Quantra.Enums;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a spread configuration with multiple option legs
    /// </summary>
    public class SpreadConfiguration
    {
        /// <summary>
        /// Unique identifier for the spread
        /// </summary>
        public string SpreadId { get; set; } = Guid.NewGuid().ToString("N");

        /// <summary>
        /// Type of spread strategy
        /// </summary>
        public MultiLegStrategyType SpreadType { get; set; }

        /// <summary>
        /// Display name for the spread
        /// </summary>
        public string DisplayName { get; set; }

        /// <summary>
        /// Underlying symbol
        /// </summary>
        public string UnderlyingSymbol { get; set; }

        /// <summary>
        /// Current price of underlying
        /// </summary>
        public double UnderlyingPrice { get; set; }

        /// <summary>
        /// List of option legs in the spread
        /// </summary>
        public List<OptionLeg> Legs { get; set; } = new List<OptionLeg>();

        /// <summary>
        /// Total cost/credit of the spread (negative for credit)
        /// </summary>
        public double NetCost { get; set; }

        /// <summary>
        /// Maximum potential profit
        /// </summary>
        public double MaxProfit { get; set; }

        /// <summary>
        /// Maximum potential loss
        /// </summary>
        public double MaxLoss { get; set; }

        /// <summary>
        /// Breakeven price(s)
        /// </summary>
        public List<double> BreakevenPrices { get; set; } = new List<double>();

        /// <summary>
        /// Probability of profit (if available)
        /// </summary>
        public double ProbabilityOfProfit { get; set; }

        /// <summary>
        /// Days to expiration
        /// </summary>
        public int DaysToExpiration { get; set; }

        /// <summary>
        /// Created timestamp
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.Now;
    }

    /// <summary>
    /// Represents a single leg in an options spread
    /// </summary>
    public class OptionLeg
    {
        /// <summary>
        /// Option data for this leg
        /// </summary>
        public OptionData Option { get; set; }

        /// <summary>
        /// Action: BUY or SELL
        /// </summary>
        public string Action { get; set; }

        /// <summary>
        /// Number of contracts
        /// </summary>
        public int Quantity { get; set; } = 1;

        /// <summary>
        /// Price paid/received for this leg
        /// </summary>
        public double Price { get; set; }

        /// <summary>
        /// Total cost/credit for this leg
        /// </summary>
        public double TotalCost => Price * Quantity * 100; // Options are per 100 shares
    }
}