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

    // Company Overview and Static Metadata Methods for TFT Model
    /// <summary>
    /// Gets company overview data from Alpha Vantage OVERVIEW endpoint with 7-day caching
    /// Used for TFT static metadata features (Sector, MarketCap, Exchange)
    /// </summary>
    /// <param name="symbol">Stock ticker symbol</param>
    /// <returns>CompanyOverview object with fundamental data</returns>
    Task<CompanyOverview> GetCompanyOverviewAsync(string symbol);

    /// <summary>
    /// Gets sector code as numerical value for ML/TFT model
    /// Maps sector names to numeric codes for static metadata features
    /// </summary>
    /// <param name="sector">Sector name from company overview</param>
    /// <returns>Numeric code for sector</returns>
    int GetSectorCode(string sector);

    /// <summary>
    /// Gets market cap category as numerical value for ML/TFT model
    /// Categories: Small-cap=0, Mid-cap=1, Large-cap=2, Mega-cap=3
    /// </summary>
    /// <param name="marketCap">Market capitalization in dollars</param>
    /// <returns>Numeric code for market cap category</returns>
    int GetMarketCapCategory(decimal? marketCap);

    /// <summary>
    /// Gets market cap category as numerical value for ML/TFT model (long overload)
    /// </summary>
    /// <param name="marketCap">Market capitalization in dollars as long</param>
    /// <returns>Numeric code for market cap category</returns>
    int GetMarketCapCategory(long marketCap);

    /// <summary>
    /// Gets exchange code as numerical value for ML/TFT model
    /// Maps exchange names to numeric codes: NYSE=0, NASDAQ=1, AMEX=2, Other=3
    /// </summary>
    /// <param name="exchange">Exchange name from company overview</param>
    /// <returns>Numeric code for exchange</returns>
    int GetExchangeCode(string exchange);

    /// <summary>
    /// Gets earnings calendar data from Alpha Vantage EARNINGS endpoint.
    /// Returns historical and future earnings dates for a symbol.
    /// Used for TFT model known future inputs.
    /// </summary>
    /// <param name="symbol">Stock ticker symbol</param>
    /// <returns>List of earnings calendar data including report dates and EPS estimates</returns>
    Task<List<EarningsCalendarData>> GetEarningsCalendarAsync(string symbol);

    // Market Context Methods for TFT Model
    /// <summary>
    /// Gets Treasury Yield data from Alpha Vantage TREASURY_YIELD endpoint.
    /// Used for market context features in TFT model (interest rate environment).
    /// </summary>
    /// <param name="maturity">Maturity period: 3month, 2year, 5year, 7year, 10year, 30year</param>
    /// <param name="interval">Data interval: daily, weekly, monthly</param>
    /// <returns>Treasury yield percentage value</returns>
    Task<double> GetTreasuryYieldAsync(string maturity = "10year", string interval = "daily");

    /// <summary>
    /// Gets Federal Funds Rate from Alpha Vantage FEDERAL_FUNDS_RATE endpoint.
    /// Used for market context features in TFT model (monetary policy environment).
    /// </summary>
    /// <param name="interval">Data interval: daily, weekly, monthly</param>
    /// <returns>Federal funds rate percentage value</returns>
    Task<double> GetFederalFundsRateAsync(string interval = "daily");

    /// <summary>
    /// Gets top gainers, losers, and most actively traded stocks.
    /// Used for market breadth calculation in MarketContextService.
    /// </summary>
    /// <returns>TopMoversResponse containing lists of top gainers, losers, and most active</returns>
    Task<TopMoversResponse> GetTopMoversAsync();

    /// <summary>
    /// Gets daily historical price data for a symbol.
    /// Used for S&P 500 and sector ETF return calculations.
    /// </summary>
    /// <param name="symbol">Stock ticker symbol</param>
    /// <param name="outputSize">Output size: compact (last 100 data points) or full (full history)</param>
    /// <param name="dataType">Data type: json or csv</param>
    /// <returns>List of historical price data</returns>
    Task<List<HistoricalPrice>> GetDailyData(string symbol, string outputSize = "full", string dataType = "json");
}
