using System;
using System.Collections.Generic;
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
        /// Unique session identifier (Guid)
        /// </summary>
        [Required]
        [MaxLength(36)]
        public string SessionId { get; set; }

        /// <summary>
        /// Session name/description
        /// </summary>
        [MaxLength(200)]
        public string Name { get; set; }

        /// <summary>
        /// Initial cash balance for the session
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal InitialCash { get; set; }

        /// <summary>
        /// Current cash balance
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal CashBalance { get; set; }

        /// <summary>
        /// Total realized P&L for the session
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal RealizedPnL { get; set; }

        /// <summary>
        /// Whether the session is currently active
        /// </summary>
        [Required]
        public bool IsActive { get; set; }

        /// <summary>
        /// When the session was started
        /// </summary>
        [Required]
        public DateTime StartedAt { get; set; }

        /// <summary>
        /// When the session was last updated
        /// </summary>
        [Required]
        public DateTime LastUpdatedAt { get; set; }

        /// <summary>
        /// When the session was ended (null if still active)
        /// </summary>
        public DateTime? EndedAt { get; set; }

        /// <summary>
        /// Navigation property for positions
        /// </summary>
        public virtual ICollection<PaperTradingPositionEntity> Positions { get; set; }

        /// <summary>
        /// Navigation property for orders
        /// </summary>
        public virtual ICollection<PaperTradingOrderEntity> Orders { get; set; }
    }

    /// <summary>
    /// Entity representing an open or closed paper trading position
    /// </summary>
    [Table("PaperTradingPositions")]
    public class PaperTradingPositionEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Foreign key to the trading session
        /// </summary>
        [Required]
        public int SessionId { get; set; }

        /// <summary>
        /// Stock symbol
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        /// <summary>
        /// Current quantity (positive = long, negative = short)
        /// </summary>
        [Required]
        public int Quantity { get; set; }

        /// <summary>
        /// Average cost per share
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal AverageCost { get; set; }

        /// <summary>
        /// Current market price
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal CurrentPrice { get; set; }

        /// <summary>
        /// Unrealized profit/loss
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal UnrealizedPnL { get; set; }

        /// <summary>
        /// Realized profit/loss from partial closes
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal RealizedPnL { get; set; }

        /// <summary>
        /// Asset type (Stock, ETF, Option)
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string AssetType { get; set; }

        /// <summary>
        /// Whether the position is closed (soft delete)
        /// </summary>
        [Required]
        public bool IsClosed { get; set; }

        /// <summary>
        /// When the position was opened
        /// </summary>
        [Required]
        public DateTime OpenedAt { get; set; }

        /// <summary>
        /// When the position was closed (null if still open)
        /// </summary>
        public DateTime? ClosedAt { get; set; }

        /// <summary>
        /// When the position was last updated
        /// </summary>
        [Required]
        public DateTime LastUpdatedAt { get; set; }

        /// <summary>
        /// Navigation property for the session
        /// </summary>
        [ForeignKey("SessionId")]
        public virtual PaperTradingSessionEntity Session { get; set; }

        /// <summary>
        /// Navigation property for fills
        /// </summary>
        public virtual ICollection<PaperTradingFillEntity> Fills { get; set; }
    }

    /// <summary>
    /// Entity representing a paper trading order
    /// </summary>
    [Table("PaperTradingOrders")]
    public class PaperTradingOrderEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Unique order identifier (matches in-memory Order.Id)
        /// </summary>
        [Required]
        [MaxLength(36)]
        public string OrderId { get; set; }

        /// <summary>
        /// Foreign key to the trading session
        /// </summary>
        [Required]
        public int SessionId { get; set; }

        /// <summary>
        /// Stock symbol
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        /// <summary>
        /// Order type (Market, Limit, Stop, StopLimit, TrailingStop)
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string OrderType { get; set; }

        /// <summary>
        /// Order side (Buy, Sell)
        /// </summary>
        [Required]
        [MaxLength(10)]
        public string Side { get; set; }

        /// <summary>
        /// Order state (Pending, Submitted, PartiallyFilled, Filled, Cancelled, Expired, Rejected)
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string State { get; set; }

        /// <summary>
        /// Total quantity ordered
        /// </summary>
        [Required]
        public int Quantity { get; set; }

        /// <summary>
        /// Quantity that has been filled
        /// </summary>
        [Required]
        public int FilledQuantity { get; set; }

        /// <summary>
        /// Limit price (for limit and stop-limit orders)
        /// </summary>
        [Column(TypeName = "decimal(18,4)")]
        public decimal LimitPrice { get; set; }

        /// <summary>
        /// Stop price (for stop and stop-limit orders)
        /// </summary>
        [Column(TypeName = "decimal(18,4)")]
        public decimal StopPrice { get; set; }

        /// <summary>
        /// Average fill price
        /// </summary>
        [Column(TypeName = "decimal(18,4)")]
        public decimal AvgFillPrice { get; set; }

        /// <summary>
        /// Time-in-force (Day, GTC, IOC, FOK)
        /// </summary>
        [Required]
        [MaxLength(10)]
        public string TimeInForce { get; set; }

        /// <summary>
        /// Asset type (Stock, ETF, Option)
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string AssetType { get; set; }

        /// <summary>
        /// Reject reason (if order was rejected)
        /// </summary>
        [MaxLength(500)]
        public string RejectReason { get; set; }

        /// <summary>
        /// Optional notes
        /// </summary>
        [MaxLength(1000)]
        public string Notes { get; set; }

        /// <summary>
        /// When the order was created
        /// </summary>
        [Required]
        public DateTime CreatedAt { get; set; }

        /// <summary>
        /// When the order was submitted
        /// </summary>
        public DateTime? SubmittedAt { get; set; }

        /// <summary>
        /// When the order was completely filled
        /// </summary>
        public DateTime? FilledAt { get; set; }

        /// <summary>
        /// Order expiration time
        /// </summary>
        public DateTime? ExpirationTime { get; set; }

        /// <summary>
        /// When the order was last updated
        /// </summary>
        [Required]
        public DateTime LastUpdatedAt { get; set; }

        /// <summary>
        /// Navigation property for the session
        /// </summary>
        [ForeignKey("SessionId")]
        public virtual PaperTradingSessionEntity Session { get; set; }

        /// <summary>
        /// Navigation property for fills
        /// </summary>
        public virtual ICollection<PaperTradingFillEntity> Fills { get; set; }
    }

    /// <summary>
    /// Entity representing a fill (execution) of a paper trading order
    /// </summary>
    [Table("PaperTradingFills")]
    public class PaperTradingFillEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Unique fill identifier (matches in-memory OrderFill.FillId)
        /// </summary>
        [Required]
        [MaxLength(36)]
        public string FillId { get; set; }

        /// <summary>
        /// Foreign key to the order
        /// </summary>
        [Required]
        public int OrderEntityId { get; set; }

        /// <summary>
        /// Foreign key to the position (if applicable)
        /// </summary>
        public int? PositionEntityId { get; set; }

        /// <summary>
        /// Stock symbol
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        /// <summary>
        /// Quantity filled
        /// </summary>
        [Required]
        public int Quantity { get; set; }

        /// <summary>
        /// Execution price
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal Price { get; set; }

        /// <summary>
        /// Side (Buy, Sell)
        /// </summary>
        [Required]
        [MaxLength(10)]
        public string Side { get; set; }

        /// <summary>
        /// Commission charged
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal Commission { get; set; }

        /// <summary>
        /// Slippage incurred
        /// </summary>
        [Required]
        [Column(TypeName = "decimal(18,4)")]
        public decimal Slippage { get; set; }

        /// <summary>
        /// Exchange/venue where fill occurred
        /// </summary>
        [MaxLength(50)]
        public string Exchange { get; set; }

        /// <summary>
        /// When the fill occurred
        /// </summary>
        [Required]
        public DateTime FillTime { get; set; }

        /// <summary>
        /// Navigation property for the order
        /// </summary>
        [ForeignKey("OrderEntityId")]
        public virtual PaperTradingOrderEntity Order { get; set; }

        /// <summary>
        /// Navigation property for the position
        /// </summary>
        [ForeignKey("PositionEntityId")]
        public virtual PaperTradingPositionEntity Position { get; set; }
    }
}
