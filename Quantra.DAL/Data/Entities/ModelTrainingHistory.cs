using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Tracks ML model training sessions and performance metrics
    /// </summary>
    [Table("ModelTrainingHistory")]
    public class ModelTrainingHistory
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; }

        /// <summary>
        /// Training session timestamp
        /// </summary>
        [Required]
        public DateTime TrainingDate { get; set; }

        /// <summary>
        /// Model type used (pytorch, tensorflow, random_forest)
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string ModelType { get; set; }

        /// <summary>
        /// Architecture type (lstm, gru, transformer)
        /// </summary>
        [MaxLength(50)]
        public string ArchitectureType { get; set; }

        /// <summary>
        /// Number of symbols used in training
        /// </summary>
        public int SymbolsCount { get; set; }

        /// <summary>
        /// Number of training samples
        /// </summary>
        public int TrainingSamples { get; set; }

        /// <summary>
        /// Number of test samples
        /// </summary>
        public int TestSamples { get; set; }

        /// <summary>
        /// Training time in seconds
        /// </summary>
        public double TrainingTimeSeconds { get; set; }

        /// <summary>
        /// Mean Absolute Error
        /// </summary>
        public double MAE { get; set; }

        /// <summary>
        /// Root Mean Squared Error
        /// </summary>
        public double RMSE { get; set; }

        /// <summary>
        /// R² Score (coefficient of determination)
        /// </summary>
        public double R2Score { get; set; }

        /// <summary>
        /// Optional notes about this training session
        /// </summary>
        public string Notes { get; set; }

        /// <summary>
        /// Whether this model is currently active for predictions
        /// </summary>
        public bool IsActive { get; set; }
    }
}
