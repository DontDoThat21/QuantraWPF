using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.DAL.TradingEngine.Data
{
    /// <summary>
    /// Real-time data source for paper trading using Alpha Vantage
    /// </summary>
    public class RealTimeDataSource : IDataSource
    {
        private readonly IAlphaVantageService _alphaVantageService;
        private readonly Dictionary<string, QuoteData> _cachedQuotes;
        private readonly Dictionary<string, DateTime> _cacheTimestamps;
        private readonly TimeSpan _cacheExpiration = TimeSpan.FromSeconds(15);

        public RealTimeDataSource(IAlphaVantageService alphaVantageService)
        {
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _cachedQuotes = new Dictionary<string, QuoteData>();
            _cacheTimestamps = new Dictionary<string, DateTime>();
        }

        /// <summary>
        /// Name of the data source
        /// </summary>
        public string Name => "RealTime";

        /// <summary>
        /// This is a real-time data source
        /// </summary>
        public bool IsRealTime => true;

        /// <summary>
        /// Gets a quote for a symbol at a specific time (for real-time, ignores time parameter)
        /// </summary>
        public async Task<Quote?> GetQuoteAsync(string symbol, DateTime time)
        {
            return await GetLiveQuoteAsync(symbol);
        }

        /// <summary>
        /// Gets the latest quote for a symbol
        /// </summary>
        public async Task<Quote?> GetLiveQuoteAsync(string symbol)
        {
            try
            {
                // Check cache first
                if (_cachedQuotes.TryGetValue(symbol, out var cachedQuote) &&
                    _cacheTimestamps.TryGetValue(symbol, out var cacheTime) &&
                    DateTime.UtcNow - cacheTime < _cacheExpiration)
                {
                    return ConvertToQuote(cachedQuote);
                }

                // Fetch from Alpha Vantage
                var quoteData = await _alphaVantageService.GetQuoteDataAsync(symbol);
                if (quoteData == null)
                {
                    return null;
                }

                // Update cache
                _cachedQuotes[symbol] = quoteData;
                _cacheTimestamps[symbol] = DateTime.UtcNow;

                return ConvertToQuote(quoteData);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error fetching quote for {symbol}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Converts Alpha Vantage QuoteData to trading engine Quote
        /// </summary>
        private Quote ConvertToQuote(QuoteData quoteData)
        {
            decimal spread = (decimal)quoteData.Price * 0.001m; // 0.1% spread estimate
            
            return new Quote
            {
                Symbol = quoteData.Symbol,
                Last = (decimal)quoteData.Price,
                Bid = (decimal)quoteData.Price - (spread / 2),
                Ask = (decimal)quoteData.Price + (spread / 2),
                BidSize = (long)Math.Max(1, quoteData.Volume / 1000), // Estimate
                AskSize = (long)Math.Max(1, quoteData.Volume / 1000),
                LastSize = 100,
                Timestamp = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Gets historical bars - not implemented for real-time source
        /// </summary>
        public Task<IEnumerable<Bar>> GetHistoricalBarsAsync(string symbol, DateTime start, DateTime end, TimeSpan interval)
        {
            // For real-time paper trading, we don't need historical bars
            return Task.FromResult<IEnumerable<Bar>>(Array.Empty<Bar>());
        }

        /// <summary>
        /// Gets options chain - not implemented for real-time source
        /// </summary>
        public Task<OptionsChain?> GetOptionChainAsync(string symbol, DateTime expiration, DateTime time)
        {
            // Not implemented for paper trading
            return Task.FromResult<OptionsChain?>(null);
        }

        /// <summary>
        /// Gets available expiration dates - not implemented for real-time source
        /// </summary>
        public Task<IEnumerable<DateTime>> GetOptionExpirationsAsync(string symbol)
        {
            // Not implemented for paper trading
            return Task.FromResult<IEnumerable<DateTime>>(Array.Empty<DateTime>());
        }

        /// <summary>
        /// Clears cached quotes
        /// </summary>
        public void ClearCache()
        {
            _cachedQuotes.Clear();
            _cacheTimestamps.Clear();
        }

        /// <summary>
        /// Clears cached quote for a specific symbol
        /// </summary>
        public void ClearCache(string symbol)
        {
            _cachedQuotes.Remove(symbol);
            _cacheTimestamps.Remove(symbol);
        }
    }
}
