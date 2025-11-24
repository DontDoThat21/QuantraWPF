using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Quantra.DAL.Data.Entities
{
    /// <summary>
    /// Entity representing a stock symbol in the database
    /// </summary>
    [Table("StockSymbols")]
    public class StockSymbolEntity
    {
        [Key]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [MaxLength(500)]
        public string Name { get; set; }

        [MaxLength(200)]
        public string Sector { get; set; }

        [MaxLength(200)]
        public string Industry { get; set; }

        public DateTime? LastUpdated { get; set; }
    }

    /// <summary>
    /// Entity representing cached stock data
    /// </summary>
    [Table("StockDataCache")]
    public class StockDataCache
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        [MaxLength(50)]
        public string TimeRange { get; set; }

        [Required]
        public string Data { get; set; }  // JSON serialized StockData

        [Required]
        public DateTime CachedAt { get; set; }

        public DateTime? ExpiresAt { get; set; }
    }

    /// <summary>
    /// Entity representing cached fundamental data
    /// </summary>
    [Table("FundamentalDataCache")]
    public class FundamentalDataCache
    {
        [Required]
        [MaxLength(20)]
        public string Symbol { get; set; }

        [Required]
        [MaxLength(100)]
        public string DataType { get; set; }

        public double? Value { get; set; }

        [Required]
        public DateTime CacheTime { get; set; }

        // Composite key defined in configuration
    }
}
