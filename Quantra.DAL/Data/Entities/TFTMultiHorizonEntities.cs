using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing a multi-horizon prediction for TFT (Temporal Fusion Transformer) model.
    /// Stores price predictions with confidence intervals for specific time horizons (1, 3, 5, 10 days).
    /// </summary>
    [Table("StockPredictionHorizons")]
    public class StockPredictionHorizonEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Foreign key reference to the parent StockPrediction.
        /// </summary>
        [Required]
        public int PredictionId { get; set; }

        /// <summary>
        /// Prediction horizon in days (e.g., 1, 3, 5, 10). Must be a positive value.
        /// </summary>
        [Required]
        [Range(1, 365, ErrorMessage = "Horizon must be between 1 and 365 days")]
        public int Horizon { get; set; }

        /// <summary>
        /// Median predicted price at this horizon (50th percentile).
        /// </summary>
        public double TargetPrice { get; set; }

        /// <summary>
        /// Lower bound of prediction confidence interval (10th percentile).
        /// </summary>
        public double LowerBound { get; set; }

        /// <summary>
        /// Upper bound of prediction confidence interval (90th percentile).
        /// </summary>
        public double UpperBound { get; set; }

        /// <summary>
        /// Horizon-specific confidence score (0.0 to 1.0).
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Date when the prediction is expected to come to fruition (PredictionDate + Horizon days).
        /// </summary>
        public DateTime ExpectedFruitionDate { get; set; }

        /// <summary>
        /// Actual price observed after the horizon period passed. Null until the horizon date arrives.
        /// </summary>
        public double? ActualPrice { get; set; }

        /// <summary>
        /// Calculated actual return: (ActualPrice - CurrentPrice) / CurrentPrice.
        /// Populated after horizon date passes.
        /// </summary>
        public double? ActualReturn { get; set; }

        /// <summary>
        /// Error percentage: (ActualPrice - TargetPrice) / TargetPrice.
        /// Populated after horizon date passes for model evaluation.
        /// </summary>
        public double? ErrorPct { get; set; }

        // Navigation property
        [ForeignKey("PredictionId")]
        public virtual StockPredictionEntity Prediction { get; set; }
    }

    /// <summary>
    /// Entity representing feature importance/weights from TFT variable selection networks.
    /// Shows which features contributed most to each prediction.
    /// </summary>
    [Table("PredictionFeatureImportance")]
    public class PredictionFeatureImportanceEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Foreign key reference to the parent StockPrediction.
        /// </summary>
        [Required]
        public int PredictionId { get; set; }

        /// <summary>
        /// Name of the feature (e.g., RSI, MACD, Volume, etc.).
        /// </summary>
        [Required]
        [MaxLength(100)]
        public string FeatureName { get; set; }

        /// <summary>
        /// Importance score from TFT attention mechanism.
        /// </summary>
        public double ImportanceScore { get; set; }

        /// <summary>
        /// Type of feature in TFT architecture: 'static', 'known', or 'observed'.
        /// - static: Time-invariant features (e.g., sector, market cap category)
        /// - known: Known future inputs (e.g., day of week, holidays)
        /// - observed: Time-varying observed inputs (e.g., price, volume, indicators)
        /// </summary>
        [MaxLength(20)]
        public string FeatureType { get; set; }

        // Navigation property
        [ForeignKey("PredictionId")]
        public virtual StockPredictionEntity Prediction { get; set; }
    }

    /// <summary>
    /// Entity representing temporal attention weights from TFT interpretable attention mechanism.
    /// Shows which past time steps were most influential in the prediction.
    /// </summary>
    [Table("PredictionTemporalAttention")]
    public class PredictionTemporalAttentionEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Foreign key reference to the parent StockPrediction.
        /// </summary>
        [Required]
        public int PredictionId { get; set; }

        /// <summary>
        /// Relative time offset (negative values, e.g., -1 = yesterday, -5 = 5 days ago).
        /// </summary>
        [Required]
        public int TimeStep { get; set; }

        /// <summary>
        /// Attention weight for this time step (higher = more influential).
        /// </summary>
        public double AttentionWeight { get; set; }

        // Navigation property
        [ForeignKey("PredictionId")]
        public virtual StockPredictionEntity Prediction { get; set; }
    }
}
