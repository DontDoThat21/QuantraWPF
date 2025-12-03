using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Quantra.DAL.TradingEngine.Data
{
    /// <summary>
    /// Represents a price quote
    /// </summary>
    public class Quote
    {
        /// <summary>
        /// Symbol of the asset
        /// </summary>
        public string Symbol { get; set; } = string.Empty;

        /// <summary>
        /// Bid price
        /// </summary>
        public decimal Bid { get; set; }

        /// <summary>
        /// Ask price
        /// </summary>
        public decimal Ask { get; set; }

        /// <summary>
        /// Last traded price
        /// </summary>
        public decimal Last { get; set; }

        /// <summary>
        /// Volume at bid
        /// </summary>
        public long BidSize { get; set; }

        /// <summary>
        /// Volume at ask
        /// </summary>
        public long AskSize { get; set; }

        /// <summary>
        /// Last trade volume
        /// </summary>
        public long LastSize { get; set; }

        /// <summary>
        /// Timestamp of the quote
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Gets the mid price (average of bid and ask)
        /// </summary>
        public decimal MidPrice => (Bid + Ask) / 2;

        /// <summary>
        /// Gets the bid-ask spread
        /// </summary>
        public decimal Spread => Ask - Bid;

        /// <summary>
        /// Gets the spread as a percentage of the mid price
        /// </summary>
        public decimal SpreadPercent => MidPrice > 0 ? (Spread / MidPrice) * 100 : 0;
    }

    /// <summary>
    /// Represents an OHLCV bar
    /// </summary>
    public class Bar
    {
        /// <summary>
        /// Symbol of the asset
        /// </summary>
        public string Symbol { get; set; } = string.Empty;

        /// <summary>
        /// Opening price
        /// </summary>
        public decimal Open { get; set; }

        /// <summary>
        /// Highest price
        /// </summary>
        public decimal High { get; set; }

        /// <summary>
        /// Lowest price
        /// </summary>
        public decimal Low { get; set; }

        /// <summary>
        /// Closing price
        /// </summary>
        public decimal Close { get; set; }

        /// <summary>
        /// Volume traded
        /// </summary>
        public long Volume { get; set; }

        /// <summary>
        /// Start time of the bar
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Duration of the bar
        /// </summary>
        public TimeSpan Interval { get; set; }

        /// <summary>
        /// Number of trades in this bar
        /// </summary>
        public int TradeCount { get; set; }

        /// <summary>
        /// Volume-weighted average price
        /// </summary>
        public decimal VWAP { get; set; }
    }

    /// <summary>
    /// Represents an options chain for a symbol
    /// </summary>
    public class OptionsChain
    {
        /// <summary>
        /// Underlying symbol
        /// </summary>
        public string UnderlyingSymbol { get; set; } = string.Empty;

        /// <summary>
        /// Current underlying price
        /// </summary>
        public decimal UnderlyingPrice { get; set; }

        /// <summary>
        /// Expiration date
        /// </summary>
        public DateTime Expiration { get; set; }

        /// <summary>
        /// Call options in the chain
        /// </summary>
        public List<OptionQuote> Calls { get; set; } = new List<OptionQuote>();

        /// <summary>
        /// Put options in the chain
        /// </summary>
        public List<OptionQuote> Puts { get; set; } = new List<OptionQuote>();

        /// <summary>
        /// Timestamp when the chain was fetched
        /// </summary>
        public DateTime Timestamp { get; set; }
    }

    /// <summary>
    /// Represents a quote for an individual option
    /// </summary>
    public class OptionQuote
    {
        /// <summary>
        /// Option symbol
        /// </summary>
        public string Symbol { get; set; } = string.Empty;

        /// <summary>
        /// Strike price
        /// </summary>
        public decimal Strike { get; set; }

        /// <summary>
        /// Whether this is a call option
        /// </summary>
        public bool IsCall { get; set; }

        /// <summary>
        /// Expiration date
        /// </summary>
        public DateTime Expiration { get; set; }

        /// <summary>
        /// Bid price
        /// </summary>
        public decimal Bid { get; set; }

        /// <summary>
        /// Ask price
        /// </summary>
        public decimal Ask { get; set; }

        /// <summary>
        /// Last traded price
        /// </summary>
        public decimal Last { get; set; }

        /// <summary>
        /// Volume traded
        /// </summary>
        public long Volume { get; set; }

        /// <summary>
        /// Open interest
        /// </summary>
        public long OpenInterest { get; set; }

        /// <summary>
        /// Implied volatility
        /// </summary>
        public decimal ImpliedVolatility { get; set; }

        /// <summary>
        /// Option delta
        /// </summary>
        public decimal Delta { get; set; }

        /// <summary>
        /// Option gamma
        /// </summary>
        public decimal Gamma { get; set; }

        /// <summary>
        /// Option theta (daily decay)
        /// </summary>
        public decimal Theta { get; set; }

        /// <summary>
        /// Option vega
        /// </summary>
        public decimal Vega { get; set; }

        /// <summary>
        /// Option rho
        /// </summary>
        public decimal Rho { get; set; }

        /// <summary>
        /// Gets the mid price
        /// </summary>
        public decimal MidPrice => (Bid + Ask) / 2;

        /// <summary>
        /// Days until expiration
        /// </summary>
        public int DaysToExpiration => (int)(Expiration.Date - DateTime.Today).TotalDays;
    }

    /// <summary>
    /// Interface for data sources (historical and live)
    /// </summary>
    public interface IDataSource
    {
        /// <summary>
        /// Gets a quote for a symbol at a specific time
        /// </summary>
        /// <param name="symbol">Symbol to quote</param>
        /// <param name="time">Time for the quote (for backtesting)</param>
        /// <returns>Quote data</returns>
        Task<Quote?> GetQuoteAsync(string symbol, DateTime time);

        /// <summary>
        /// Gets the latest quote for a symbol
        /// </summary>
        /// <param name="symbol">Symbol to quote</param>
        /// <returns>Quote data</returns>
        Task<Quote?> GetLiveQuoteAsync(string symbol);

        /// <summary>
        /// Gets historical bars for a symbol
        /// </summary>
        /// <param name="symbol">Symbol to get bars for</param>
        /// <param name="start">Start time</param>
        /// <param name="end">End time</param>
        /// <param name="interval">Bar interval</param>
        /// <returns>List of bars</returns>
        Task<IEnumerable<Bar>> GetHistoricalBarsAsync(string symbol, DateTime start, DateTime end, TimeSpan interval);

        /// <summary>
        /// Gets an options chain for a symbol
        /// </summary>
        /// <param name="symbol">Underlying symbol</param>
        /// <param name="expiration">Expiration date</param>
        /// <param name="time">Time for the chain (for backtesting)</param>
        /// <returns>Options chain</returns>
        Task<OptionsChain?> GetOptionChainAsync(string symbol, DateTime expiration, DateTime time);

        /// <summary>
        /// Gets available expiration dates for an options chain
        /// </summary>
        /// <param name="symbol">Underlying symbol</param>
        /// <returns>List of available expiration dates</returns>
        Task<IEnumerable<DateTime>> GetOptionExpirationsAsync(string symbol);

        /// <summary>
        /// Indicates if this data source provides real-time data
        /// </summary>
        bool IsRealTime { get; }

        /// <summary>
        /// Name of the data source
        /// </summary>
        string Name { get; }
    }
}
