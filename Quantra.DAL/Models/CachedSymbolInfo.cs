namespace Quantra.Models
{
    /// <summary>
    /// Represents information about a cached stock symbol for display in dropdowns
    /// </summary>
    public class CachedSymbolInfo
    {
        /// <summary>
        /// The stock symbol (e.g., AAPL, MSFT)
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Additional cache information for display (e.g., time range, last updated)
        /// </summary>
        public string CacheInfo { get; set; }

        public override string ToString()
        {
            return $"{Symbol} - {CacheInfo}";
        }
    }
}
