using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing an order in trading history
    /// </summary>
    [Table("OrderHistory")]
    public class OrderHistoryEntity
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        [MaxLength(20)]
        public string OrderType { get; set; }

        [Required]
        public int Quantity { get; set; }

        [Required]
        public float Price { get; set; }

        public float? StopLoss { get; set; }

        public float? TakeProfit { get; set; }

        public bool IsPaperTrade { get; set; }

        [MaxLength(50)]
        public string Status { get; set; }

        [MaxLength(200)]
        public string PredictionSource { get; set; }

        [Required]
        public DateTime Timestamp { get; set; }
    }

    /// <summary>
    /// Entity representing a completed trade record
    /// </summary>
    [Table("TradeRecords")]
    public class TradeRecordEntity
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        [MaxLength(20)]
        public string Action { get; set; }

        [Required]
        public double Price { get; set; }

        [Required]
        public double TargetPrice { get; set; }

        public double Confidence { get; set; }

        [Required]
        public DateTime ExecutionTime { get; set; }

        [MaxLength(50)]
        public string Status { get; set; }

        public string Notes { get; set; }
    }

    /// <summary>
    /// Entity representing a trading rule
    /// </summary>
    [Table("TradingRules")]
    public class TradingRuleEntity
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(200)]
        public string Name { get; set; }

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        [MaxLength(20)]
        public string OrderType { get; set; }

        public bool IsActive { get; set; }

        [Required]
        public string Conditions { get; set; }  // JSON serialized list of conditions

        [Required]
        public DateTime CreatedDate { get; set; }

        [Required]
        public DateTime LastModified { get; set; }

        // Additional properties for advanced trading rules
        [Required]
        public double MinConfidence { get; set; }

        [Required]
        public double EntryPrice { get; set; }

        [Required]
        public double ExitPrice { get; set; }

        [Required]
        public double StopLoss { get; set; }

        [Required]
        public int Quantity { get; set; }
    }

    /// <summary>
    /// Entity representing a saved backtest result for comparison and analysis
    /// </summary>
    [Table("BacktestResults")]
    public class BacktestResultEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Stock symbol that was backtested
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        /// <summary>
        /// Name of the strategy used
        /// </summary>
        [Required]
        [MaxLength(200)]
        public string StrategyName { get; set; }

        /// <summary>
        /// Timeframe used for the backtest (e.g., "daily", "1h")
        /// </summary>
        [MaxLength(50)]
        public string TimeFrame { get; set; }

        /// <summary>
        /// Start date of the backtest period
        /// </summary>
        [Required]
        public DateTime StartDate { get; set; }

        /// <summary>
        /// End date of the backtest period
        /// </summary>
        [Required]
        public DateTime EndDate { get; set; }

        /// <summary>
        /// Initial capital used for the backtest
        /// </summary>
        [Required]
        public double InitialCapital { get; set; }

        /// <summary>
        /// Final equity at the end of the backtest
        /// </summary>
        [Required]
        public double FinalEquity { get; set; }

        /// <summary>
        /// Total return as a decimal (e.g., 0.15 for 15%)
        /// </summary>
        [Required]
        public double TotalReturn { get; set; }

        /// <summary>
        /// Maximum drawdown as a decimal
        /// </summary>
        [Required]
        public double MaxDrawdown { get; set; }

        /// <summary>
        /// Win rate as a decimal
        /// </summary>
        public double WinRate { get; set; }

        /// <summary>
        /// Total number of trades executed
        /// </summary>
        public int TotalTrades { get; set; }

        /// <summary>
        /// Number of winning trades
        /// </summary>
        public int WinningTrades { get; set; }

        /// <summary>
        /// Sharpe ratio (risk-adjusted return)
        /// </summary>
        public double SharpeRatio { get; set; }

        /// <summary>
        /// Sortino ratio (downside risk-adjusted return)
        /// </summary>
        public double SortinoRatio { get; set; }

        /// <summary>
        /// Compound Annual Growth Rate
        /// </summary>
        public double CAGR { get; set; }

        /// <summary>
        /// Calmar ratio (CAGR / Max Drawdown)
        /// </summary>
        public double CalmarRatio { get; set; }

        /// <summary>
        /// Profit factor (gross profit / gross loss)
        /// </summary>
        public double ProfitFactor { get; set; }

        /// <summary>
        /// Information ratio
        /// </summary>
        public double InformationRatio { get; set; }

        /// <summary>
        /// Total transaction costs incurred
        /// </summary>
        public double TotalTransactionCosts { get; set; }

        /// <summary>
        /// Gross return before costs
        /// </summary>
        public double GrossReturn { get; set; }

        /// <summary>
        /// JSON serialized equity curve data
        /// </summary>
        public string EquityCurveJson { get; set; }

        /// <summary>
        /// JSON serialized trades data
        /// </summary>
        public string TradesJson { get; set; }

        /// <summary>
        /// JSON serialized strategy parameters
        /// </summary>
        public string StrategyParametersJson { get; set; }

        /// <summary>
        /// Notes or comments about the backtest
        /// </summary>
        [MaxLength(1000)]
        public string Notes { get; set; }

        /// <summary>
        /// When the backtest was run
        /// </summary>
        [Required]
        public DateTime CreatedAt { get; set; }

        /// <summary>
        /// Optional user-defined name for the backtest run
        /// </summary>
        [MaxLength(200)]
        public string RunName { get; set; }
    }
}
