namespace Quantra.Enums
{
    /// <summary>
    /// Defines different modes for symbol selection in the StockExplorer
    /// </summary>
    public enum SymbolSelectionMode
    {
        /// <summary>
        /// Individual asset selection mode - user can manually select any symbol
        /// </summary>
        IndividualAsset,
        
        /// <summary>
        /// Top volume stocks with RSI discrepancies - shows stocks with high volume and RSI divergence patterns
        /// </summary>
        TopVolumeRsiDiscrepancies,
        
        /// <summary>
        /// Top P/E ratio stocks - shows stocks sorted by price-to-earnings ratio
        /// </summary>
        TopPE,
        
        /// <summary>
        /// High volume stocks - shows stocks with highest trading volume
        /// </summary>
        HighVolume,
        
        /// <summary>
        /// Low P/E value stocks - shows undervalued stocks based on P/E ratio
        /// </summary>
        LowPE,
        
        /// <summary>
        /// RSI oversold stocks - shows stocks with RSI below 30 (potentially oversold)
        /// </summary>
        RsiOversold,
        
        /// <summary>
        /// RSI overbought stocks - shows stocks with RSI above 70 (potentially overbought)
        /// </summary>
        RsiOverbought,
        
        /// <summary>
        /// All database stocks - shows all cached stock data from the database
        /// </summary>
        AllDatabase,
        
        /// <summary>
        /// High Theta stocks - shows stocks with high time decay suitable for theta harvesting strategies
        /// </summary>
        HighTheta,
        
        /// <summary>
        /// High Beta stocks - shows stocks with high market correlation (beta > 1.2) for momentum trading
        /// </summary>
        HighBeta,
        
        /// <summary>
        /// High Alpha stocks - shows stocks generating consistent excess returns vs market
        /// </summary>
        HighAlpha,
        
        /// <summary>
        /// Bullish Cup and Handle pattern - shows stocks with bullish cup and handle formation
        /// </summary>
        BullishCupAndHandle,
        
        /// <summary>
        /// Bearish Cup and Handle pattern - shows stocks with bearish cup and handle formation (inverted pattern)
        /// </summary>
        BearishCupAndHandle,
        
        /// <summary>
        /// OHLCV Candles mode - shows intraday OHLCV candlestick data from TIME_SERIES_INTRADAY API
        /// </summary>
        OhlcvCandles
    }
}