using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing a saved filter configuration for StockExplorer
    /// </summary>
    [Table("SavedFilters")]
    public class SavedFilter
    {
        [Key]
        public int Id { get; set; }

        /// <summary>
        /// Name of the saved filter
        /// </summary>
        [Required]
        [MaxLength(100)]
        public string Name { get; set; }

        /// <summary>
        /// Optional description of the filter
        /// </summary>
        [MaxLength(500)]
        public string? Description { get; set; }

        /// <summary>
        /// Foreign key to UserCredentials table. Null for system/default filters.
        /// </summary>
        public int? UserId { get; set; }

        /// <summary>
        /// Navigation property to the user who owns this filter
        /// </summary>
        [ForeignKey("UserId")]
        public virtual UserCredential? User { get; set; }

        /// <summary>
        /// Whether this is a system-provided filter (cannot be deleted by users)
        /// </summary>
        public bool IsSystemFilter { get; set; }

        // Filter values
        [MaxLength(100)]
        public string? SymbolFilter { get; set; }

        [MaxLength(100)]
        public string? PriceFilter { get; set; }

        [MaxLength(100)]
        public string? PeRatioFilter { get; set; }

        [MaxLength(100)]
        public string? VwapFilter { get; set; }

        [MaxLength(100)]
        public string? RsiFilter { get; set; }

        [MaxLength(100)]
        public string? ChangePercentFilter { get; set; }

        [MaxLength(100)]
        public string? MarketCapFilter { get; set; }

        // Metadata
        [Column("CreatedDate")]
        public DateTime CreatedAt { get; set; }

        [Column("ModifiedDate")]
        public DateTime LastModified { get; set; }

        public override string ToString()
        {
            return Name;
        }
    }
}
