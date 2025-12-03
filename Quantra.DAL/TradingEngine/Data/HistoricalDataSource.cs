using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.TradingEngine.Data
{
    /// <summary>
    /// Data source for historical data (used in backtesting)
    /// Adapts existing HistoricalDataService to the IDataSource interface
    /// </summary>
    public class HistoricalDataSource : IDataSource
    {
        private readonly Services.HistoricalDataService? _historicalDataService;
        private readonly Dictionary<string, List<Bar>> _cachedBars;
        private readonly Dictionary<string, List<HistoricalPrice>> _historicalPrices;

        /// <summary>
        /// Creates a new historical data source
        /// </summary>
        public HistoricalDataSource(Services.HistoricalDataService? historicalDataService = null)
        {
            _historicalDataService = historicalDataService;
            _cachedBars = new Dictionary<string, List<Bar>>();
            _historicalPrices = new Dictionary<string, List<HistoricalPrice>>();
        }

        /// <summary>
        /// Name of the data source
        /// </summary>
        public string Name => "Historical";

        /// <summary>
        /// Historical data source is not real-time
        /// </summary>
        public bool IsRealTime => false;

        /// <summary>
        /// Pre-loads historical data for a symbol (for efficient backtesting)
        /// </summary>
        public void LoadHistoricalData(string symbol, List<HistoricalPrice> prices)
        {
            _historicalPrices[symbol] = prices.OrderBy(p => p.Date).ToList();
            
            // Convert to bars
            var bars = prices.Select(p => new Bar
            {
                Symbol = symbol,
                Open = (decimal)p.Open,
                High = (decimal)p.High,
                Low = (decimal)p.Low,
                Close = (decimal)p.Close,
                Volume = p.Volume,
                Timestamp = p.Date,
                Interval = TimeSpan.FromDays(1)
            }).OrderBy(b => b.Timestamp).ToList();

            _cachedBars[symbol] = bars;
        }

        /// <summary>
        /// Gets a quote for a symbol at a specific time
        /// </summary>
        public Task<Quote?> GetQuoteAsync(string symbol, DateTime time)
        {
            if (!_cachedBars.TryGetValue(symbol, out var bars) || bars == null || !bars.Any())
            {
                return Task.FromResult<Quote?>(null);
            }

            // Find the bar at or before the requested time
            var bar = bars.LastOrDefault(b => b.Timestamp.Date <= time.Date);
            if (bar == null)
            {
                return Task.FromResult<Quote?>(null);
            }

            // Simulate bid/ask spread (0.1% spread)
            decimal spread = bar.Close * 0.001m;
            
            var quote = new Quote
            {
                Symbol = symbol,
                Bid = bar.Close - (spread / 2),
                Ask = bar.Close + (spread / 2),
                Last = bar.Close,
                BidSize = 100,
                AskSize = 100,
                LastSize = 100,
                Timestamp = bar.Timestamp
            };

            return Task.FromResult<Quote?>(quote);
        }

        /// <summary>
        /// Gets live quote - for historical source, returns most recent data
        /// </summary>
        public Task<Quote?> GetLiveQuoteAsync(string symbol)
        {
            return GetQuoteAsync(symbol, DateTime.UtcNow);
        }

        /// <summary>
        /// Gets historical bars for a symbol
        /// </summary>
        public Task<IEnumerable<Bar>> GetHistoricalBarsAsync(string symbol, DateTime start, DateTime end, TimeSpan interval)
        {
            if (!_cachedBars.TryGetValue(symbol, out var bars) || bars == null)
            {
                return Task.FromResult<IEnumerable<Bar>>(Array.Empty<Bar>());
            }

            var filteredBars = bars.Where(b => b.Timestamp >= start && b.Timestamp <= end).ToList();
            return Task.FromResult<IEnumerable<Bar>>(filteredBars);
        }

        /// <summary>
        /// Gets options chain - simulates based on underlying price
        /// </summary>
        public Task<OptionsChain?> GetOptionChainAsync(string symbol, DateTime expiration, DateTime time)
        {
            if (!_cachedBars.TryGetValue(symbol, out var bars) || bars == null || !bars.Any())
            {
                return Task.FromResult<OptionsChain?>(null);
            }

            // Find the bar at or before the requested time
            var bar = bars.LastOrDefault(b => b.Timestamp.Date <= time.Date);
            if (bar == null)
            {
                return Task.FromResult<OptionsChain?>(null);
            }

            decimal underlyingPrice = bar.Close;
            
            // Generate simulated options chain
            var chain = new OptionsChain
            {
                UnderlyingSymbol = symbol,
                UnderlyingPrice = underlyingPrice,
                Expiration = expiration,
                Timestamp = time
            };

            // Generate strikes around the money
            decimal atmStrike = Math.Round(underlyingPrice, 0);
            decimal strikeInterval = underlyingPrice switch
            {
                < 50 => 1m,
                < 200 => 2.5m,
                < 500 => 5m,
                _ => 10m
            };

            for (int i = -5; i <= 5; i++)
            {
                decimal strike = atmStrike + (i * strikeInterval);
                if (strike <= 0) continue;

                double daysToExp = (expiration.Date - time.Date).TotalDays;
                double yearsToExp = Math.Max(daysToExp / 365.0, 0.001);
                double volatility = 0.25; // Assumed 25% IV

                // Simplified Black-Scholes for simulation
                var (callPrice, putPrice) = SimulateOptionPrices(
                    (double)underlyingPrice, 
                    (double)strike, 
                    yearsToExp, 
                    0.03, // Risk-free rate
                    volatility);

                decimal spread = Math.Max(0.01m, (decimal)callPrice * 0.05m);

                chain.Calls.Add(new OptionQuote
                {
                    Symbol = $"{symbol}{expiration:yyMMdd}C{strike:00000}",
                    Strike = strike,
                    IsCall = true,
                    Expiration = expiration,
                    Bid = Math.Max(0, (decimal)callPrice - spread / 2),
                    Ask = (decimal)callPrice + spread / 2,
                    Last = (decimal)callPrice,
                    ImpliedVolatility = (decimal)volatility,
                    Volume = 100,
                    OpenInterest = 1000
                });

                spread = Math.Max(0.01m, (decimal)putPrice * 0.05m);

                chain.Puts.Add(new OptionQuote
                {
                    Symbol = $"{symbol}{expiration:yyMMdd}P{strike:00000}",
                    Strike = strike,
                    IsCall = false,
                    Expiration = expiration,
                    Bid = Math.Max(0, (decimal)putPrice - spread / 2),
                    Ask = (decimal)putPrice + spread / 2,
                    Last = (decimal)putPrice,
                    ImpliedVolatility = (decimal)volatility,
                    Volume = 100,
                    OpenInterest = 1000
                });
            }

            return Task.FromResult<OptionsChain?>(chain);
        }

        /// <summary>
        /// Gets available expiration dates
        /// </summary>
        public Task<IEnumerable<DateTime>> GetOptionExpirationsAsync(string symbol)
        {
            // Return standard monthly expirations for next 6 months
            var expirations = new List<DateTime>();
            DateTime current = DateTime.Today;

            for (int i = 0; i < 6; i++)
            {
                // Third Friday of each month
                DateTime firstOfMonth = new DateTime(current.Year, current.Month, 1).AddMonths(i);
                DateTime thirdFriday = GetThirdFriday(firstOfMonth);
                if (thirdFriday > DateTime.Today)
                {
                    expirations.Add(thirdFriday);
                }
            }

            return Task.FromResult<IEnumerable<DateTime>>(expirations);
        }

        private static DateTime GetThirdFriday(DateTime date)
        {
            DateTime firstOfMonth = new DateTime(date.Year, date.Month, 1);
            int daysUntilFriday = ((int)DayOfWeek.Friday - (int)firstOfMonth.DayOfWeek + 7) % 7;
            DateTime firstFriday = firstOfMonth.AddDays(daysUntilFriday);
            return firstFriday.AddDays(14); // Third Friday
        }

        private static (double call, double put) SimulateOptionPrices(
            double S, double K, double T, double r, double sigma)
        {
            if (T <= 0) T = 0.001;

            double d1 = (Math.Log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.Sqrt(T));
            double d2 = d1 - sigma * Math.Sqrt(T);

            double Nd1 = CumulativeNormal(d1);
            double Nd2 = CumulativeNormal(d2);
            double Nmd1 = CumulativeNormal(-d1);
            double Nmd2 = CumulativeNormal(-d2);

            double call = S * Nd1 - K * Math.Exp(-r * T) * Nd2;
            double put = K * Math.Exp(-r * T) * Nmd2 - S * Nmd1;

            return (Math.Max(0, call), Math.Max(0, put));
        }

        private static double CumulativeNormal(double x)
        {
            if (x < 0)
                return 1.0 - CumulativeNormal(-x);

            double k = 1.0 / (1.0 + 0.2316419 * x);
            double result = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));

            result = result * Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
            return 1.0 - result;
        }
    }
}
