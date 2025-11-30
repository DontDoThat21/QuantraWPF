namespace Quantra.DAL.Enums
{
    /// <summary>
    /// Defines different modes for loading historical data in the StockExplorer
    /// </summary>
    public enum HistoricalsLoadMode
    {
        /// <summary>
        /// Load historical data for all symbols from the AlphaVantage API (12459 total)
        /// </summary>
        AllSymbols,
        
        /// <summary>
        /// Load historical data only for symbols that haven't been loaded yet
        /// </summary>
        NonLoadedOnly,
        
        /// <summary>
        /// Load historical data for a predefined stock configuration
        /// </summary>
        StockConfiguration
    }
}
