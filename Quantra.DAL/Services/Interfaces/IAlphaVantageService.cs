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
        
        // New methods for Premium API
        Task<List<HistoricalPrice>> GetForexHistoricalData(string fromSymbol, string toSymbol, string interval = "daily");
        Task<List<HistoricalPrice>> GetCryptoHistoricalData(string symbol, string market = "USD", string interval = "daily");
        Task<List<HistoricalPrice>> GetExtendedHistoricalData(string symbol, string interval = "daily", string outputSize = "full", string dataType = "json");
        
        // Order history management
        void AddOrderToHistory(OrderModel order);
        
        // Property to check if using premium API
        bool IsPremiumKey { get; }
    }
}