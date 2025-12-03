using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing a paper trading session
    /// </summary>
    [Table("PaperTradingSessions")]
    public class PaperTradingSessionEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Unique session identifier
        /// </summary>
        [Required]
        public Guid SessionId { get; set; }

        /// <summary>
        /// User ID if authentication is implemented
        /// </summary>
        public int? UserId { get; set; }

        /// <summary>
        /// When the session started
        /// </summary>
        [Required]
        public DateTime StartTime { get; set; }

        /// <summary>
        /// When the session ended (null if still running)
        /// </summary>
        public DateTime? EndTime { get; set; }

        /// <summary>
        /// Initial cash balance at session start
        /// </summary>
        [Required]
        public decimal InitialBalance { get; set; }

        /// <summary>
        /// Final cash balance at session end
        /// </summary>
        public decimal? FinalBalance { get; set; }

        /// <summary>
        /// Final portfolio value (cash + positions)
        /// </summary>
        public decimal? FinalPortfolioValue { get; set; }

        /// <summary>
        /// Total profit/loss for the session
        /// </summary>
        public decimal? TotalPnL { get; set; }

        /// <summary>
        /// Realized profit/loss from closed positions
        /// </summary>
        public decimal? RealizedPnL { get; set; }

        /// <summary>
        /// Unrealized profit/loss from open positions
        /// </summary>
        public decimal? UnrealizedPnL { get; set; }

        /// <summary>
        /// Number of trades executed during the session
        /// </summary>
        public int TradeCount { get; set; }

        /// <summary>
        /// Number of winning trades
        /// </summary>
        public int WinningTrades { get; set; }

        /// <summary>
        /// Number of losing trades
        /// </summary>
        public int LosingTrades { get; set; }

        /// <summary>
        /// Win rate percentage
        /// </summary>
        public decimal? WinRate { get; set; }

        /// <summary>
        /// Session status (Active, Completed, Reset)
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string Status { get; set; }

        /// <summary>
        /// Optional notes about the session
        /// </summary>
        [MaxLength(1000)]
        public string Notes { get; set; }

        /// <summary>
        /// When the record was created
        /// </summary>
        [Required]
        public DateTime CreatedAt { get; set; }

        /// <summary>
        /// When the record was last updated
        /// </summary>
        [Required]
        public DateTime UpdatedAt { get; set; }
    }
}
