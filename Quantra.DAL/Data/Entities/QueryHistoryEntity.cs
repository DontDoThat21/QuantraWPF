using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing a query history entry for audit trail (MarketChat story 5).
    /// Logs all natural language queries executed through Market Chat.
    /// </summary>
    [Table("QueryHistory")]
    public class QueryHistoryEntity
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// The original natural language query from the user
        /// </summary>
        [Required]
        [MaxLength(2000)]
        public string OriginalQuery { get; set; }

        /// <summary>
        /// The translated SQL query that was executed
        /// </summary>
        [MaxLength]
        public string TranslatedSql { get; set; }

        /// <summary>
        /// The table(s) that were queried (comma-separated)
        /// </summary>
        [MaxLength(500)]
        public string QueriedTables { get; set; }

        /// <summary>
        /// Whether the query was executed successfully
        /// </summary>
        [Required]
        public bool Success { get; set; }

        /// <summary>
        /// Number of rows returned
        /// </summary>
        public int RowCount { get; set; }

        /// <summary>
        /// Error message if the query failed
        /// </summary>
        [MaxLength(2000)]
        public string ErrorMessage { get; set; }

        /// <summary>
        /// Whether the query was blocked for safety reasons
        /// </summary>
        public bool WasBlocked { get; set; }

        /// <summary>
        /// Reason the query was blocked, if applicable
        /// </summary>
        [MaxLength(500)]
        public string BlockedReason { get; set; }

        /// <summary>
        /// Time taken to execute the query in milliseconds
        /// </summary>
        public long ExecutionTimeMs { get; set; }

        /// <summary>
        /// Timestamp when the query was executed
        /// </summary>
        [Required]
        public DateTime ExecutedAt { get; set; }

        /// <summary>
        /// Optional user identifier for multi-user support
        /// </summary>
        [MaxLength(100)]
        public string UserId { get; set; }

        /// <summary>
        /// Optional session identifier for grouping related queries
        /// </summary>
        [MaxLength(100)]
        public string SessionId { get; set; }
    }
}
