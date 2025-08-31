using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Enums;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    public interface IMarketDataService
    {
        /// <summary>
        /// Gets the current market price for a symbol
        /// </summary>
        Task<double> GetMarketPrice(string symbol);
        
        /// <summary>
        /// Fetches quote data for a symbol
        /// </summary>
        Task<QuoteData> FetchQuoteData(string symbol);
        
        /// <summary>
        /// Gets quote data for a symbol
        /// </summary>
        Task<QuoteData> GetQuoteData(string symbol);
        
        /// <summary>
        /// Gets stock data for a symbol and time range
        /// </summary>
        Task<StockData> GetStockData(string symbol, string timeRange);
        
        /// <summary>
        /// Fetches chart data for a symbol and time range
        /// </summary>
        Task<StockData> FetchChartData(string symbol, string timeRange);
        
        /// <summary>
        /// Gets historical closing prices for a symbol
        /// </summary>
        Task<List<double>> GetHistoricalClosingPrices(string symbol, int dataPoints);
        
        /// <summary>
        /// Gets the most volatile stocks
        /// </summary>
        Task<List<string>> GetMostVolatileStocks();
        
        /// <summary>
        /// Handles real-time market data
        /// </summary>
        void HandleMarketData(string data);
        
        /// <summary>
        /// Sets trading hour restrictions
        /// </summary>
        bool SetTradingHourRestrictions(TimeOnly marketOpen, TimeOnly marketClose);
        
        /// <summary>
        /// Sets which market sessions are enabled for trading
        /// </summary>
        bool SetEnabledMarketSessions(MarketSession sessions);
        
        /// <summary>
        /// Gets which market sessions are currently enabled for trading
        /// </summary>
        MarketSession GetEnabledMarketSessions();
        
        /// <summary>
        /// Sets the time boundaries for different market sessions
        /// </summary>
        bool SetMarketSessionTimes(TimeOnly preMarketOpenTime, TimeOnly regularMarketOpenTime, 
            TimeOnly regularMarketCloseTime, TimeOnly afterHoursCloseTime);
            
        /// <summary>
        /// Gets the current market session time boundaries
        /// </summary>
        (TimeOnly preMarketOpen, TimeOnly regularMarketOpen, TimeOnly regularMarketClose, TimeOnly afterHoursClose) GetMarketSessionTimes();
        
        /// <summary>
        /// Checks if trading is currently allowed based on market session and time restrictions
        /// </summary>
        bool IsTradingAllowed();
        
        /// <summary>
        /// Gets the current market conditions for rebalancing decisions
        /// </summary>
        Task<MarketConditions> GetMarketConditions();
    }
}