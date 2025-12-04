using System;
using System.Collections.Generic;
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

        /// <summary>
        /// The date when the prediction is expected to come to fruition (target date for the price prediction)
        /// </summary>
        public DateTime? ExpectedFruitionDate { get; set; }

        /// <summary>
        /// The model type used for this prediction (pytorch, tensorflow, random_forest)
        /// </summary>
        [MaxLength(50)]
        public string? ModelType { get; set; }

        /// <summary>
        /// The architecture type used for this prediction (lstm, gru, transformer, tft)
        /// </summary>
        [MaxLength(50)]
        public string? ArchitectureType { get; set; }

        /// <summary>
        /// Reference to the training history ID that was used to generate this prediction
        /// </summary>
        public int? TrainingHistoryId { get; set; }

        /// <summary>
        /// Trading rule or strategy name that generated this prediction (optional)
        /// </summary>
        [MaxLength(200)]
        public string? TradingRule { get; set; }

        /// <summary>
        /// Original user query that triggered this prediction (optional)
        /// </summary>
        [MaxLength(1000)]
        public string? UserQuery { get; set; }

        /// <summary>
        /// Links prediction to a chat history record (optional foreign key to ChatHistory.Id)
        /// </summary>
        public int? ChatHistoryId { get; set; }

        // Navigation properties
        public virtual ICollection<PredictionIndicatorEntity> Indicators { get; set; }

        /// <summary>
        /// Navigation property to the chat history record that triggered this prediction
        /// </summary>
        [ForeignKey("ChatHistoryId")]
        public virtual ChatHistoryEntity ChatHistory { get; set; }
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
        public string? PredictedAction { get; set; }

        public double? Confidence { get; set; }

        public DateTime? PredictionTimestamp { get; set; }

        [Required]
        public DateTime CreatedAt { get; set; }

        /// <summary>
        /// Tracks how many times this cache entry has been accessed
        /// </summary>
        public int AccessCount { get; set; }

        /// <summary>
        /// Timestamp of the last access to help identify stale entries
        /// </summary>
        public DateTime? LastAccessedAt { get; set; }
    }

    /// <summary>
    /// Entity representing chat history for Market Chat conversations
    /// </summary>
    [Table("ChatHistory")]
    public class ChatHistoryEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Unique session identifier for grouping related chat messages
        /// </summary>
        [Required]
        [MaxLength(100)]
        public string SessionId { get; set; }

        /// <summary>
        /// The content of the chat message
        /// </summary>
        [Required]
        public string Content { get; set; }

        /// <summary>
        /// Whether this message is from the user (true) or AI assistant (false)
        /// </summary>
        [Required]
        public bool IsFromUser { get; set; }

        /// <summary>
        /// Timestamp when the message was created
        /// </summary>
        [Required]
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Type of message (UserQuestion, AssistantResponse, SystemMessage, etc.)
        /// </summary>
        [MaxLength(50)]
        public string? MessageType { get; set; }

        /// <summary>
        /// Optional stock symbol associated with this chat message
        /// </summary>
        [MaxLength(20)]
        public string? Symbol { get; set; }

        /// <summary>
        /// User identifier for multi-user support (optional)
        /// </summary>
        [MaxLength(100)]
        public string? UserId { get; set; }

        /// <summary>
        /// Navigation property for predictions that reference this chat history record
        /// </summary>
        public virtual ICollection<StockPredictionEntity> Predictions { get; set; }
    }
}
