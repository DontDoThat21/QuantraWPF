using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing earnings calendar data for a stock symbol.
    /// Used for TFT model known future inputs - earnings dates we know ahead of time.
    /// </summary>
    [Table("EarningsCalendar")]
    public class EarningsCalendarEntity
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; }

        [Required]
        [MaxLength(10)]
        public string Symbol { get; set; }

        [Required]
        public DateTime EarningsDate { get; set; }

        [MaxLength(10)]
        public string FiscalQuarter { get; set; }

        [Column(TypeName = "decimal(10, 4)")]
        public decimal? EPSEstimate { get; set; }

        [Required]
        public DateTime LastUpdated { get; set; }
    }
}
