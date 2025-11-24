using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing a stock prediction
    /// </summary>
    [Table("StockPredictions")]
    public class StockPredictionEntity
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        [MaxLength(20)]
        public string PredictedAction { get; set; }

        [Required]
        public double Confidence { get; set; }

        [Required]
        public double CurrentPrice { get; set; }

        [Required]
        public double TargetPrice { get; set; }

        [Required]
        public double PotentialReturn { get; set; }

        [Required]
        public DateTime CreatedDate { get; set; }

        // Navigation property
        public virtual ICollection<PredictionIndicatorEntity> Indicators { get; set; }
    }

    /// <summary>
    /// Entity representing an indicator value for a prediction
    /// </summary>
    [Table("PredictionIndicators")]
    public class PredictionIndicatorEntity
    {
        [Required]
        public int PredictionId { get; set; }

        [Required]
        [MaxLength(100)]
        public string IndicatorName { get; set; }

        [Required]
        public double IndicatorValue { get; set; }

        // Navigation property
        [ForeignKey("PredictionId")]
        public virtual StockPredictionEntity Prediction { get; set; }
    }

    /// <summary>
    /// Entity representing cached ML model prediction results
    /// </summary>
    [Table("PredictionCache")]
    public class PredictionCacheEntity
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        [MaxLength(50)]
        public string ModelVersion { get; set; }

        [Required]
        [MaxLength(100)]
        public string InputDataHash { get; set; }

        public double? PredictedPrice { get; set; }

        [MaxLength(20)]
        public string PredictedAction { get; set; }

        public double? Confidence { get; set; }

        public DateTime? PredictionTimestamp { get; set; }

        [Required]
        public DateTime CreatedAt { get; set; }
    }
}
