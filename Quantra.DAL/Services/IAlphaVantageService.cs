using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra;
using Quantra.Models;

public interface IAlphaVantageService
{
    void LogApiUsage();
    void LogApiUsage(string operation, string details = null);
    int GetCurrentDbApiCallCount();

    /// <summary>
    /// Fetches all technical indicator data for the given stock symbol in a single API call if possible.
    /// </summary>
    /// <param name="symbol">The stock symbol.</param>
    /// <returns>A dictionary of indicator name to value.</returns>
    Task<Dictionary<string, double>> GetAllTechnicalIndicatorsAsync(string symbol);

    /// <summary>
    /// Fetches indicator data for the given stock symbol.
    /// </summary>
    /// <param name="symbol">The stock symbol.</param>
    /// <returns>A list of indicator data.</returns>
    Task<List<StockIndicator>> GetIndicatorsAsync(string symbol);

    /// <summary>
    /// Searches for symbols using the Alpha Vantage SYMBOL_SEARCH endpoint
    /// </summary>
    /// <param name="keywords">Keywords to search for</param>
    /// <returns>List of matching symbols</returns>
    Task<List<SymbolSearchResult>> SearchSymbolsAsync(string keywords);

    /// <summary>
    /// Gets quote data for a specific symbol
    /// </summary>
    /// <param name="symbol">Stock symbol</param>
    /// <returns>Quote data</returns>
    Task<QuoteData> GetQuoteDataAsync(string symbol);

    // Analytics API Methods
    /// <summary>
    /// Get fixed window analytics for performance metrics calculation
    /// Uses ANALYTICS_FIXED_WINDOW endpoint from Alpha Vantage
    /// </summary>
    /// <param name="symbols">Comma-separated list of symbols (e.g., "AAPL,SPY,QQQ")</param>
    /// <param name="startDate">Start date for analysis window</param>
    /// <param name="interval">Time interval (DAILY, WEEKLY, MONTHLY)</param>
    /// <param name="calculations">Comma-separated calculations (e.g., "MEAN_VALUE,STDDEV,CORRELATION")</param>
    /// <param name="ohlcType">Price type to use (close, open, high, low)</param>
    /// <returns>Analytics result with calculated metrics</returns>
    Task<AnalyticsFixedWindowResult> GetAnalyticsFixedWindowAsync(
        string symbols,
        System.DateTime startDate,
        string interval = "DAILY",
        string calculations = "MEAN_VALUE,STDDEV,CORRELATION",
        string ohlcType = "close");
}
