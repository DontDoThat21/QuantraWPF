using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing insider transactions in the database
    /// </summary>
    [Table("InsiderTransactions")]
    [Microsoft.EntityFrameworkCore.Index(nameof(Symbol), Name = "idx_insider_transactions_symbol")]
    [Microsoft.EntityFrameworkCore.Index(nameof(TransactionDate), Name = "idx_insider_transactions_date")]
    [Microsoft.EntityFrameworkCore.Index(nameof(LastUpdated), Name = "idx_insider_transactions_lastupdated")]
    public class InsiderTransactionEntity
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; }

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        [Column(TypeName = "datetime2")]
        public DateTime FilingDate { get; set; }

        [Required]
        [Column(TypeName = "datetime2")]
        public DateTime TransactionDate { get; set; }

        [MaxLength(255)]
        public string OwnerName { get; set; }

        [MaxLength(50)]
        public string OwnerCik { get; set; }

        [MaxLength(255)]
        public string OwnerTitle { get; set; }

        [MaxLength(100)]
        public string SecurityType { get; set; }

        [MaxLength(10)]
        public string TransactionCode { get; set; }

        public int? SharesTraded { get; set; }

        public double? PricePerShare { get; set; }

        public int? SharesOwnedFollowing { get; set; }

        [MaxLength(10)]
        public string AcquisitionOrDisposal { get; set; }

        [Required]
        [Column(TypeName = "datetime2")]
        public DateTime LastUpdated { get; set; } = DateTime.Now;
    }
}
