using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    public interface IAlphaVantageService
    {
        Task<double> GetQuoteData(string symbol, string? interval = null);
        Task<QuoteData> GetQuoteDataAsync(string symbol);
        Task<List<string>> GetAllStockSymbols();
        Task<List<SymbolSearchResult>> SearchSymbolsAsync(string keywords);
        Task<double> GetRSI(string symbol, string? interval = null);
        Task<double> GetLatestADX(string symbol, string? interval = null);
        Task<double> GetATR(string symbol, string? interval = null);
        Task<double> GetMomentumScore(string symbol, string? interval = null);
        Task<(double StochK, double StochD)> GetSTOCH(string symbol, string? interval = null);
        Task<double> GetCCI(string symbol, string? interval = null);
        Task<double> GetUltimateOscillator(string symbol, string? interval = null);
        Task<double> GetMFI(string symbol, string? interval = null);
        Task<double> GetVWAP(string symbol, string? interval = null);
        Task<(double Macd, double MacdSignal, double MacdHist)> GetMACD(string symbol, string? interval = null, string? seriesType = null);
        Task<double> GetOBV(string symbol, string? interval = null);
        Task<T> SendWithSlidingWindowAsync<T>(string functionName, Dictionary<string, string> parameters);
        Task<List<string>> GetMostVolatileStocksAsync();
        void LogApiUsage();
        void LogApiUsage(string endpoint, string? parameters);
        int GetCurrentDbApiCallCount();
        int ApiCallLimit { get; }
        Task<Dictionary<string, double>> GetAllTechnicalIndicatorsAsync(string symbol);
        Task<List<StockIndicator>> GetIndicatorsAsync(string symbol);
        Task<List<double>> GetHistoricalClosingPricesAsync(string symbol, int count);
        
        // Time Series Data Methods based on AlphaVantage API
        // Intraday: TIME_SERIES_INTRADAY (1min, 5min, 15min, 30min, 60min)
        Task<List<HistoricalPrice>> GetIntradayData(string symbol, string interval = "5min", string outputSize = "compact", string dataType = "json");
        
        // Daily/Weekly/Monthly non-adjusted endpoints
        Task<List<HistoricalPrice>> GetDailyData(string symbol, string outputSize = "full", string dataType = "json");
        Task<List<HistoricalPrice>> GetWeeklyData(string symbol, string dataType = "json");
        Task<List<HistoricalPrice>> GetMonthlyData(string symbol, string dataType = "json");
        
        // Extended historical data with optional adjusted prices
        Task<List<HistoricalPrice>> GetExtendedHistoricalData(string symbol, string interval = "daily", string outputSize = "full", string dataType = "json", bool useAdjusted = true);
        
        // Forex and Crypto data methods
        Task<List<HistoricalPrice>> GetForexHistoricalData(string fromSymbol, string toSymbol, string interval = "daily");
        Task<List<HistoricalPrice>> GetCryptoHistoricalData(string symbol, string market = "USD", string interval = "daily");
        
        // Property to check if using premium API
        bool IsPremiumKey { get; }

        // Intelligence and News API Methods
        Task<NewsSentimentResponse> GetNewsSentimentAsync(
            string tickers = null,
            string topics = null,
            string timeFrom = null,
            string timeTo = null,
            string sort = "LATEST",
            int limit = 50);
        Task<TopMoversResponse> GetTopMoversAsync();
        Task<InsiderTransactionsResponse> GetInsiderTransactionsAsync(string symbol);

        // Analytics Fixed Window API Methods
        /// <summary>
        /// Gets advanced analytics metrics using the Alpha Vantage Analytics Fixed Window API
        /// </summary>
        /// <param name="symbols">Comma-separated list of symbols (e.g., "AAPL,MSFT,GOOGL")</param>
        /// <param name="range">Time range for analysis (e.g., "1month", "3month", "1year", "full")</param>
        /// <param name="interval">Data interval: DAILY, WEEKLY, or MONTHLY</param>
        /// <param name="calculations">Comma-separated list of metrics to calculate (e.g., "MEAN,STDDEV,CUMULATIVE_RETURN")</param>
        /// <returns>Analytics response with calculated metrics</returns>
        Task<AnalyticsFixedWindowResponse> GetAnalyticsFixedWindowAsync(
            string symbols,
            string range = "full",
            string interval = "DAILY",
            string calculations = "MEAN,STDDEV,CUMULATIVE_RETURN");

        /// <summary>
        /// Gets performance metrics (Sharpe, Sortino, etc.) for a symbol using the Analytics API
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="startDate">Start date for analysis</param>
        /// <param name="endDate">End date for analysis</param>
        /// <param name="benchmark">Benchmark symbol for comparison (default: SPY)</param>
        /// <returns>Performance metrics calculated from API data</returns>
        Task<PerformanceMetrics> GetPerformanceMetricsAsync(
            string symbol,
            DateTime? startDate = null,
            DateTime? endDate = null,
            string benchmark = "SPY");
    }
}