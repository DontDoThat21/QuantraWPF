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

        public double? EPS { get; set; }

        public DateTime? LastUpdated { get; set; }
    }

    /// <summary>
    /// Entity representing cached stock data
    /// </summary>
        [Table("StockDataCache")]
        public class StockDataCache
        {
            [Required]
            [MaxLength(20)]
            public string Symbol { get; set; }

            [Required]
            [MaxLength(50)]
            public string TimeRange { get; set; }

            [MaxLength(50)]
            public string Interval { get; set; }

            [Required]
            public string Data { get; set; }  // JSON serialized StockData

            [Required]
            [Column("CacheTime")]
            public DateTime CachedAt { get; set; }
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

        /// <summary>
        /// Entity representing stock data displayed in the StockExplorer view.
        /// This table stores all loaded stock data from AlphaVantage for display purposes.
        /// NOT to be confused with StockDataCache which is used for ML predictions.
        /// </summary>
        [Table("StockExplorerData")]
        public class StockExplorerDataEntity
        {
            [Key]
            public int Id { get; set; }

            /// <summary>
            /// Stock symbol (e.g., AAPL, MSFT)
            /// </summary>
            [Required]
            [MaxLength(20)]
            public string Symbol { get; set; }

            /// <summary>
            /// Company name
            /// </summary>
            [MaxLength(500)]
            public string Name { get; set; }

            /// <summary>
            /// Current stock price
            /// </summary>
            public double Price { get; set; }

            /// <summary>
            /// Price change in dollars
            /// </summary>
            public double Change { get; set; }

            /// <summary>
            /// Price change as percentage
            /// </summary>
            public double ChangePercent { get; set; }

            /// <summary>
            /// Day's high price
            /// </summary>
            public double DayHigh { get; set; }

            /// <summary>
            /// Day's low price
            /// </summary>
            public double DayLow { get; set; }

            /// <summary>
            /// Market capitalization
            /// </summary>
            public double MarketCap { get; set; }

            /// <summary>
            /// Trading volume
            /// </summary>
            public double Volume { get; set; }

            /// <summary>
            /// Stock sector (e.g., Technology, Healthcare)
            /// </summary>
            [MaxLength(200)]
            public string Sector { get; set; }

            /// <summary>
            /// Relative Strength Index (0-100)
            /// </summary>
            public double RSI { get; set; }

            /// <summary>
            /// Price-to-Earnings ratio
            /// </summary>
            public double PERatio { get; set; }

            /// <summary>
            /// Volume-Weighted Average Price
            /// </summary>
            public double VWAP { get; set; }

            /// <summary>
            /// Date of the stock data
            /// </summary>
            public DateTime Date { get; set; }

            /// <summary>
            /// When this data was last updated from AlphaVantage API
            /// </summary>
            public DateTime LastUpdated { get; set; }

            /// <summary>
            /// When this record was last accessed/viewed
            /// </summary>
            public DateTime LastAccessed { get; set; }

            /// <summary>
            /// Timestamp of the data point
            /// </summary>
            public DateTime Timestamp { get; set; }

            /// <summary>
            /// When this data was cached in the database
            /// </summary>
            public DateTime? CacheTime { get; set; }
        }
    }
