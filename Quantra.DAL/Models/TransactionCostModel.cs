using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Models transaction costs for realistic backtesting including commissions, spreads, slippage, etc.
    /// </summary>
    public class TransactionCostModel
    {
        // Commission models
        public enum CommissionType
        {
            Fixed,          // Fixed fee per trade
            PerShare,       // Fee per share traded
            Percentage      // Percentage of trade value
        }

        // Slippage models
        public enum SlippageType
        {
            None,           // No slippage
            Fixed,          // Fixed amount of slippage
            Percentage,     // Percentage of price
            VolumeBasedPercentage // Price slippage based on trade size relative to volume
        }

        #region Properties

        // Commission parameters
        public CommissionType CommissionModel { get; set; } = CommissionType.Fixed;
        public double CommissionValue { get; set; } = 0; // Value depends on CommissionModel (e.g., $5 flat, $0.01/share, 0.1%)
        public double MinCommission { get; set; } = 0;   // Minimum commission per trade
        public double MaxCommission { get; set; } = double.MaxValue; // Maximum commission per trade

        // Spread parameters
        public double SpreadPercentage { get; set; } = 0; // Spread as a percentage of price (e.g., 0.1% = 0.001)
        public double MinSpreadAbsolute { get; set; } = 0; // Minimum spread in absolute terms

        // Slippage parameters
        public SlippageType SlippageModel { get; set; } = SlippageType.None;
        public double SlippageValue { get; set; } = 0; // Interpretation depends on SlippageModel

        // Exchange and regulatory fees
        public double ExchangeFeePerShare { get; set; } = 0; // Per share exchange fees
        public double ExchangeFeeMinimum { get; set; } = 0; // Minimum exchange fee per trade
        public double RegulatoryFeePercentage { get; set; } = 0; // SEC fees, etc.

        // Tax parameters
        public double SalesTax { get; set; } = 0; // For certain jurisdictions
        public double CapitalGainsTaxRate { get; set; } = 0; // Can be used for after-tax performance analysis

        #endregion

        #region Factory Methods

        /// <summary>
        /// Creates a basic cost model with fixed commission per trade
        /// </summary>
        /// <param name="commissionPerTrade">Fixed commission amount per trade</param>
        public static TransactionCostModel CreateFixedCommissionModel(double commissionPerTrade)
        {
            return new TransactionCostModel
            {
                CommissionModel = CommissionType.Fixed,
                CommissionValue = commissionPerTrade
            };
        }

        /// <summary>
        /// Creates a cost model with percentage-based commissions
        /// </summary>
        /// <param name="commissionPercentage">Commission percentage (e.g., 0.1% = 0.001)</param>
        /// <param name="minimumCommission">Minimum commission per trade</param>
        public static TransactionCostModel CreatePercentageCommissionModel(
            double commissionPercentage,
            double minimumCommission = 0)
        {
            return new TransactionCostModel
            {
                CommissionModel = CommissionType.Percentage,
                CommissionValue = commissionPercentage,
                MinCommission = minimumCommission
            };
        }

        /// <summary>
        /// Creates a cost model with per-share commissions
        /// </summary>
        /// <param name="commissionPerShare">Commission per share</param>
        /// <param name="minimumCommission">Minimum commission per trade</param>
        public static TransactionCostModel CreatePerShareCommissionModel(
            double commissionPerShare,
            double minimumCommission = 0)
        {
            return new TransactionCostModel
            {
                CommissionModel = CommissionType.PerShare,
                CommissionValue = commissionPerShare,
                MinCommission = minimumCommission
            };
        }

        /// <summary>
        /// Creates a retail brokerage cost model with typical fees for US equities
        /// </summary>
        public static TransactionCostModel CreateRetailBrokerageModel()
        {
            return new TransactionCostModel
            {
                CommissionModel = CommissionType.Fixed,
                CommissionValue = 5.95, // Typical retail commission
                SpreadPercentage = 0.0001, // 1 basis point (0.01%)
                SlippageModel = SlippageType.Percentage,
                SlippageValue = 0.0005, // 5 basis points (0.05%) slippage
                RegulatoryFeePercentage = 0.000024 // SEC fees, etc.
            };
        }

        /// <summary>
        /// Creates a cost model for active traders with better commission rates but still accounting for spread and slippage
        /// </summary>
        public static TransactionCostModel CreateActiveTraderModel()
        {
            return new TransactionCostModel
            {
                CommissionModel = CommissionType.PerShare,
                CommissionValue = 0.005, // Half a cent per share
                MinCommission = 1.00,
                SpreadPercentage = 0.0001, // 1 basis point spread
                SlippageModel = SlippageType.Percentage,
                SlippageValue = 0.0003, // 3 basis points slippage
                RegulatoryFeePercentage = 0.000024 // SEC fees
            };
        }

        /// <summary>
        /// Creates a cost model for institutional traders with minimal commissions but still accounting for market impact
        /// </summary>
        public static TransactionCostModel CreateInstitutionalModel()
        {
            return new TransactionCostModel
            {
                CommissionModel = CommissionType.PerShare,
                CommissionValue = 0.002, // 0.2 cents per share
                SpreadPercentage = 0.0001, // 1 basis point spread
                SlippageModel = SlippageType.VolumeBasedPercentage,
                SlippageValue = 0.1, // Slippage factor for volume-based calculation
                RegulatoryFeePercentage = 0.000024 // SEC fees
            };
        }

        /// <summary>
        /// Creates a zero-cost model (not realistic but useful for comparing the effect of costs)
        /// </summary>
        public static TransactionCostModel CreateZeroCostModel()
        {
            return new TransactionCostModel();
        }

        #endregion

        #region Cost Calculation Methods

        /// <summary>
        /// Calculate commission for a trade
        /// </summary>
        /// <param name="quantity">Number of shares/contracts</param>
        /// <param name="price">Execution price</param>
        /// <returns>Commission amount</returns>
        public double CalculateCommission(int quantity, double price)
        {
            double commission = 0;
            double tradeValue = Math.Abs(quantity) * price;

            switch (CommissionModel)
            {
                case CommissionType.Fixed:
                    commission = CommissionValue;
                    break;

                case CommissionType.PerShare:
                    commission = Math.Abs(quantity) * CommissionValue;
                    break;

                case CommissionType.Percentage:
                    commission = tradeValue * CommissionValue;
                    break;
            }

            // Apply minimum/maximum commission constraints
            commission = Math.Max(MinCommission, Math.Min(MaxCommission, commission));

            return commission;
        }

        /// <summary>
        /// Calculate slippage for a trade
        /// </summary>
        /// <param name="quantity">Number of shares/contracts traded</param>
        /// <param name="price">Expected execution price</param>
        /// <param name="isBuy">Whether this is a buy order</param>
        /// <param name="volumeForDay">Optional: Volume for the day (for volume-based slippage)</param>
        /// <returns>Price slippage amount (always positive)</returns>
        public double CalculateSlippage(int quantity, double price, bool isBuy, double volumeForDay = 0)
        {
            double slippage = 0;

            switch (SlippageModel)
            {
                case SlippageType.None:
                    slippage = 0;
                    break;

                case SlippageType.Fixed:
                    slippage = SlippageValue;
                    break;

                case SlippageType.Percentage:
                    slippage = price * SlippageValue;
                    break;

                case SlippageType.VolumeBasedPercentage:
                    if (volumeForDay > 0)
                    {
                        // Higher slippage when trade is larger percentage of daily volume
                        double volumeRatio = Math.Abs(quantity) / volumeForDay;
                        slippage = price * SlippageValue * Math.Pow(volumeRatio, 0.5); // Square root model
                    }
                    else
                    {
                        slippage = price * SlippageValue * 0.01; // Default to 1% of slippage value if no volume data
                    }
                    break;
            }

            return slippage;
        }

        /// <summary>
        /// Calculate spread cost for a trade
        /// </summary>
        /// <param name="price">Midpoint price</param>
        /// <returns>Half the spread (amount added/subtracted from midpoint price)</returns>
        public double CalculateHalfSpread(double price)
        {
            double halfSpread = price * SpreadPercentage / 2;
            return Math.Max(MinSpreadAbsolute / 2, halfSpread);
        }

        /// <summary>
        /// Calculate regulatory fees for a trade
        /// </summary>
        /// <param name="quantity">Number of shares/contracts</param>
        /// <param name="price">Execution price</param>
        /// <param name="isSell">Whether this is a sell order (some fees only apply to sells)</param>
        /// <returns>Total regulatory fees</returns>
        public double CalculateRegulatoryFees(int quantity, double price, bool isSell)
        {
            double tradeValue = Math.Abs(quantity) * price;
            double fees = 0;

            // SEC fees typically only apply to sells
            if (isSell)
            {
                fees += tradeValue * RegulatoryFeePercentage;
            }

            // Exchange fees
            fees += Math.Max(ExchangeFeeMinimum, Math.Abs(quantity) * ExchangeFeePerShare);

            return fees;
        }

        /// <summary>
        /// Calculate all transaction costs for a trade
        /// </summary>
        /// <param name="quantity">Number of shares/contracts</param>
        /// <param name="price">Expected execution price before slippage/spread</param>
        /// <param name="isBuy">Whether this is a buy order</param>
        /// <param name="volumeForDay">Optional: Volume for the day</param>
        /// <returns>Total transaction costs and effective execution price</returns>
        public (double TotalCost, double EffectivePrice) CalculateAllCosts(
            int quantity, double price, bool isBuy, double volumeForDay = 0)
        {
            // Calculate spread component
            double halfSpread = CalculateHalfSpread(price);

            // Calculate slippage
            double slippage = CalculateSlippage(quantity, price, isBuy, volumeForDay);

            // Effective execution price after spread and slippage
            double effectivePrice = isBuy ?
                price + halfSpread + slippage : // Buys execute at higher price
                price - halfSpread - slippage;  // Sells execute at lower price

            // Calculate commission
            double commission = CalculateCommission(Math.Abs(quantity), effectivePrice);

            // Calculate regulatory fees
            double regFees = CalculateRegulatoryFees(Math.Abs(quantity), effectivePrice, !isBuy);

            // Total transaction cost
            double totalCost = commission + regFees + (Math.Abs(quantity) * (halfSpread + slippage));

            return (totalCost, effectivePrice);
        }

        #endregion
    }
}