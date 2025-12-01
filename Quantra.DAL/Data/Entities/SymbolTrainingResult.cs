using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Tracks individual symbol performance during model training
    /// </summary>
    [Table("SymbolTrainingResults")]
    public class SymbolTrainingResult
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; }

        /// <summary>
        /// Reference to the training session
        /// </summary>
        [Required]
        public int TrainingHistoryId { get; set; }

        [ForeignKey(nameof(TrainingHistoryId))]
        public virtual ModelTrainingHistory TrainingHistory { get; set; }

        /// <summary>
        /// Stock symbol
        /// </summary>
        [Required]
        [MaxLength(10)]
        [Column(TypeName = "varchar(10)")]
        public string Symbol { get; set; }

        /// <summary>
        /// Number of data points for this symbol
        /// </summary>
        public int DataPointsCount { get; set; }

        /// <summary>
        /// Number of training samples from this symbol
        /// </summary>
        public int TrainingSamplesCount { get; set; }

        /// <summary>
        /// Number of test samples from this symbol
        /// </summary>
        public int TestSamplesCount { get; set; }

        /// <summary>
        /// Symbol-specific MAE (if calculated)
        /// </summary>
        public double? SymbolMAE { get; set; }

        /// <summary>
        /// Symbol-specific RMSE (if calculated)
        /// </summary>
        public double? SymbolRMSE { get; set; }

        /// <summary>
        /// Whether this symbol's data was included in training
        /// </summary>
        public bool IncludedInTraining { get; set; }

        /// <summary>
        /// Reason if excluded from training
        /// </summary>
        [MaxLength(500)]
        public string ExclusionReason { get; set; }

        /// <summary>
        /// Date range of data used
        /// </summary>
        public DateTime? DataStartDate { get; set; }

        public DateTime? DataEndDate { get; set; }
    }
}
