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
 public double Price { get; set; }

 public double? StopLoss { get; set; }

    public double? TakeProfit { get; set; }

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
    }
}
