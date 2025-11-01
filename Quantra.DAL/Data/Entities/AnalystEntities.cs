using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing an analyst rating
    /// </summary>
    [Table("AnalystRatings")]
    public class AnalystRatingEntity
    {
     [Key]
  public int Id { get; set; }

   [Required]
     [MaxLength(20)]
  public string Symbol { get; set; }

     [Required]
    [MaxLength(200)]
   public string AnalystName { get; set; }

        [Required]
     [MaxLength(50)]
     public string Rating { get; set; }

     [MaxLength(50)]
public string PreviousRating { get; set; }

   [Required]
   public double PriceTarget { get; set; }

 public double PreviousPriceTarget { get; set; }

        [Required]
   public DateTime RatingDate { get; set; }

        [Required]
        [MaxLength(50)]
  public string ChangeType { get; set; }
    }

    /// <summary>
    /// Entity representing analyst consensus history
  /// </summary>
    [Table("ConsensusHistory")]
 public class ConsensusHistoryEntity
    {
     [Key]
   public int Id { get; set; }

    [Required]
     [MaxLength(20)]
  public string Symbol { get; set; }

        [Required]
        [MaxLength(50)]
        public string ConsensusRating { get; set; }

        [Required]
 public double ConsensusScore { get; set; }

  public int BuyCount { get; set; }

   public int HoldCount { get; set; }

     public int SellCount { get; set; }

        public int UpgradeCount { get; set; }

public int DowngradeCount { get; set; }

        public double AveragePriceTarget { get; set; }

        [Required]
[MaxLength(50)]
   public string ConsensusTrend { get; set; }

        public double RatingsStrengthIndex { get; set; }

        [Required]
   public DateTime SnapshotDate { get; set; }
  }

    /// <summary>
    /// Entity representing Alpha Vantage API usage for rate limiting
    /// </summary>
    [Table("AlphaVantageApiUsage")]
    public class AlphaVantageApiUsage
    {
        [Key]
        public int Id { get; set; }

   [Required]
        public DateTime TimestampUtc { get; set; }

     [MaxLength(200)]
     public string Endpoint { get; set; }

      public string Parameters { get; set; }
    }
}
