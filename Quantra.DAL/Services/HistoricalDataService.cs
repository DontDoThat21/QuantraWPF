using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using Quantra.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Globalization;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    public class HistoricalDataService : IHistoricalDataService
    {
        private static readonly HttpClient client = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };

        // Add AlphaVantageService field
        private readonly AlphaVantageService _alphaVantageService;

        // Update constructor to initialize AlphaVantageService
        public HistoricalDataService(UserSettingsService userSettingsService, LoggingService loggingService, StockSymbolCacheService stockSymbolCacheService)
        {
            _alphaVantageService = new AlphaVantageService(userSettingsService, loggingService, stockSymbolCacheService);
        }

        /// <summary>
        /// Gets forex historical price data
        /// </summary>
        /// <param name="fromSymbol">From currency symbol</param>
        /// <param name="toSymbol">To currency symbol</param>
        /// <param name="interval">Data interval</param>
        /// <returns>List of historical prices</returns>
        public async Task<List<HistoricalPrice>> GetForexHistoricalData(string fromSymbol, string toSymbol, string interval = "daily")
        {
            try
            {
                if (_alphaVantageService is AlphaVantageService avService)
                {
                    return await avService.GetForexHistoricalData(fromSymbol, toSymbol, interval);
                }

                throw new InvalidOperationException("Premium API required for forex data");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching forex data for {fromSymbol}/{toSymbol}: {ex.Message}");
                return new List<HistoricalPrice>();
            }
        }

        /// <summary>
        /// Gets cryptocurrency historical price data
        /// </summary>
        /// <param name="symbol">Cryptocurrency symbol</param>
        /// <param name="market">Market (e.g., USD)</param>
        /// <param name="interval">Data interval</param>
        /// <returns>List of historical prices</returns>
        public async Task<List<HistoricalPrice>> GetCryptoHistoricalData(string symbol, string market = "USD", string interval = "daily")
        {
            try
            {
                if (_alphaVantageService is AlphaVantageService avService)
                {
                    return await avService.GetCryptoHistoricalData(symbol, market, interval);
                }

                throw new InvalidOperationException("Could not retrieve cryptocurrency data");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching crypto data for {symbol}: {ex.Message}");
                return new List<HistoricalPrice>();
            }
        }

        /// <summary>
        /// Gets complete historical price data with adjusted values for backtesting
        /// </summary>
        /// <param name="symbol">Asset symbol</param>
        /// <param name="interval">Data interval</param>
        /// <param name="assetClass">Asset class (stock, forex, crypto)</param>
        /// <returns>List of historical prices</returns>
        public async Task<List<HistoricalPrice>> GetComprehensiveHistoricalData(string symbol, string interval = "daily", string assetClass = "stock")
        {
            // Determine asset class if not specified
            if (assetClass == "auto")
            {
                if (symbol.Contains("/"))
                {
                    assetClass = "forex";
                }
                else
                {
                    string[] cryptos = { "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "XLM", "UNI" };
                    assetClass = cryptos.Contains(symbol) ? "crypto" : "stock";
                }
            }

            switch (assetClass.ToLower())
            {
                case "forex":
                    string[] parts = symbol.Split('/');
                    if (parts.Length == 2)
                    {
                        return await GetForexHistoricalData(parts[0], parts[1], interval);
                    }
                    break;

                case "crypto":
                    return await GetCryptoHistoricalData(symbol, "USD", interval);

                case "stock":
                default:
                    if (_alphaVantageService is AlphaVantageService avService)
                    {
                        return await avService.GetExtendedHistoricalData(symbol, interval);
                    }
                    else
                    {
                        return await GetHistoricalPrices(symbol, "max", interval);
                    }
            }

            return new List<HistoricalPrice>();
        }

        /// <summary>
        /// Gets historical price data for a symbol from Alpha Vantage API
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <param name="range">Time range (not all Alpha Vantage endpoints support range; see notes below)</param>
        /// <param name="interval">Data interval - valid values: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly</param>
        /// <returns>List of historical price data</returns>
        public async Task<List<HistoricalPrice>> GetHistoricalPrices(string symbol, string range = "1y", string interval = "1d")
        {
            try
            {
                // For premium API access, use the extended historical data method
                if (_alphaVantageService is AlphaVantageService avService)
                {
                    // Check if the asset is a forex pair (e.g., EUR/USD)
                    if (symbol.Contains("/"))
                    {
                        string[] parts = symbol.Split('/');
                        if (parts.Length == 2)
                        {
                            string fromSymbol = parts[0];
                            string toSymbol = parts[1];
                            return await avService.GetForexHistoricalData(fromSymbol, toSymbol, interval);
                        }
                    }

                    // Check if the asset is a cryptocurrency (e.g., BTC)
                    // This is a simple check - you might want to use a more comprehensive approach
                    string[] cryptos = { "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "XLM", "UNI" };
                    if (cryptos.Contains(symbol))
                    {
                        return await avService.GetCryptoHistoricalData(symbol, "USD", interval);
                    }

                    // For stocks and other equities, use the extended historical data
                    return await avService.GetExtendedHistoricalData(symbol, interval, "full");
                }

                // Fall back to standard implementation if not using AlphaVantageService
                // Map interval to Alpha Vantage format
                string function;
                string avInterval = "";
                if (interval.EndsWith("min"))
                {
                    function = "TIME_SERIES_INTRADAY";
                    avInterval = interval;
                }
                else if (interval == "1d" || interval == "daily")
                {
                    function = "TIME_SERIES_DAILY_ADJUSTED";
                }
                else if (interval == "1wk" || interval == "weekly")
                {
                    function = "TIME_SERIES_WEEKLY";
                }
                else if (interval == "1mo" || interval == "monthly")
                {
                    function = "TIME_SERIES_MONTHLY";
                }
                else
                {
                    function = "TIME_SERIES_DAILY_ADJUSTED";
                }

                // Build parameters dictionary for AlphaVantageService
                var parameters = new Dictionary<string, string>
                {
                    { "symbol", symbol },
                    { "outputsize", "full" },
                    { "apikey", AlphaVantageService.GetApiKey() }
                };
                if (function == "TIME_SERIES_INTRADAY")
                {
                    parameters.Add("interval", avInterval);
                }

                // Log Alpha Vantage API usage if using Alpha Vantage endpoint
                _alphaVantageService.LogApiUsage("GetHistoricalPrices", $"symbol={symbol}&range={range}&interval={interval}");

                // Use the correct SendWithSlidingWindowAsync signature
                var responseString = await _alphaVantageService.SendWithSlidingWindowAsync<string>(function, parameters);
                return ParseAlphaVantageResponse(responseString, function);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching historical data for {symbol}: {ex.Message}");
                return new List<HistoricalPrice>();
            }
        }

        /// <summary>
        /// Gets the latest stock data for a symbol using Alpha Vantage API.
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <returns>QuoteData object with the latest price and related info, or null if unavailable</returns>
        public async Task<QuoteData> GetLatestStockDataAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return null;

            // Log API usage for tracking
            _alphaVantageService.LogApiUsage("GetLatestStockDataAsync", $"symbol={symbol}");

            // Use AlphaVantageService to fetch the latest quote data
            var quote = await _alphaVantageService.GetQuoteDataAsync(symbol);

            return quote;
        }

        /// <summary>
        /// Parses the Alpha Vantage API response and converts it to a list of HistoricalPrice objects
        /// </summary>
        private List<HistoricalPrice> ParseAlphaVantageResponse(string jsonResponse, string function)
        {
            var result = new List<HistoricalPrice>();
            try
            {
                var jsonObject = JObject.Parse(jsonResponse);

                // Determine the correct time series key
                string timeSeriesKey = function switch
                {
                    "TIME_SERIES_INTRADAY" => jsonObject.Properties().FirstOrDefault(p => p.Name.StartsWith("Time Series"))?.Name,
                    "TIME_SERIES_DAILY_ADJUSTED" => "Time Series (Daily)",
                    "TIME_SERIES_WEEKLY" => "Weekly Time Series",
                    "TIME_SERIES_MONTHLY" => "Monthly Time Series",
                    _ => null
                };

                if (timeSeriesKey == null || !jsonObject.ContainsKey(timeSeriesKey))
                    return result;

                var timeSeries = jsonObject[timeSeriesKey] as JObject;
                if (timeSeries == null)
                    return result;

                foreach (var item in timeSeries)
                {
                    var dateStr = item.Key;
                    var data = item.Value;

                    if (DateTime.TryParse(dateStr, out DateTime date))
                    {
                        double ParseDouble(string key)
                        {
                            var token = data[key];
                            if (token == null) return 0;
                            return double.TryParse(token.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out double val) ? val : 0;
                        }

                        long ParseLong(string key)
                        {
                            var token = data[key];
                            if (token == null) return 0;
                            return long.TryParse(token.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out long val) ? val : 0;
                        }

                        result.Add(new HistoricalPrice
                        {
                            Date = date,
                            Open = ParseDouble("1. open"),
                            High = ParseDouble("2. high"),
                            Low = ParseDouble("3. low"),
                            Close = ParseDouble("4. close"),
                            Volume = ParseLong("6. volume") != 0 ? ParseLong("6. volume") : ParseLong("5. volume"),
                            AdjClose = data["5. adjusted close"] != null ? ParseDouble("5. adjusted close") : ParseDouble("4. close")
                        });
                    }
                }

                // Sort by date ascending
                result = result.OrderBy(h => h.Date).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error parsing Alpha Vantage response: {ex.Message}");
            }
            return result;
        }

        /// <summary>
        /// Converts a list of historical prices to a StockData object for charting
        /// </summary>
        public async Task<StockData> ConvertToStockData(List<HistoricalPrice> historicalPrices, string symbol, string range = "1y", string interval = "1d")
        {
            if (historicalPrices == null || !historicalPrices.Any())
            {
                return new StockData();
            }

            var stockData = new StockData
            {
                Prices = historicalPrices.Select(h => h.Close).ToList(),
                Dates = historicalPrices.Select(h => h.Date).ToList(),
                CandleData = historicalPrices.Select(h => new LiveCharts.Defaults.OhlcPoint(
                    h.Open,
                    h.High,
                    h.Low,
                    h.Close
                )).ToList()
            };

            var period = Math.Min(20, historicalPrices.Count);

            var prices = (await GetHistoricalPrices(symbol, range, interval)).Select(h => h.Close).ToList();

            if (prices.Count >= period)
            {
                var (upperBand, middleBand, lowerBand) = CalculateBollingerBands(prices, period, 2.0);
                stockData.UpperBand = upperBand;
                stockData.MiddleBand = middleBand;
                stockData.LowerBand = lowerBand;

                // Calculate RSI
                stockData.RSI = CalculateRSI(prices, 14);
            }

            return stockData;
        }

        private (List<double> upperBand, List<double> middleBand, List<double> lowerBand) CalculateBollingerBands(List<double> prices, int period, double stdDevMultiplier)
        {
            var middleBand = new List<double>();
            var upperBand = new List<double>();
            var lowerBand = new List<double>();

            // Add empty values for initial periods where we can't calculate
            for (int i = 0; i < period - 1; i++)
            {
                middleBand.Add(double.NaN);
                upperBand.Add(double.NaN);
                lowerBand.Add(double.NaN);
            }

            for (int i = period - 1; i < prices.Count; i++)
            {
                var periodPrices = prices.Skip(i - period + 1).Take(period).ToList();
                var average = periodPrices.Average();
                var stdDev = Math.Sqrt(periodPrices.Average(v => Math.Pow(v - average, 2)));

                middleBand.Add(average);
                upperBand.Add(average + stdDevMultiplier * stdDev);
                lowerBand.Add(average - stdDevMultiplier * stdDev);
            }

            return (upperBand, middleBand, lowerBand);
        }

        private List<double> CalculateRSI(List<double> prices, int period)
        {
            var rsiValues = new List<double>();

            // Add empty values for initial periods where we can't calculate
            for (int i = 0; i < period; i++)
            {
                rsiValues.Add(double.NaN);
            }

            if (prices.Count <= period)
            {
                return rsiValues;
            }

            List<double> gains = new List<double>();
            List<double> losses = new List<double>();

            // Calculate price changes
            for (int i = 1; i < prices.Count; i++)
            {
                double change = prices[i] - prices[i - 1];
                gains.Add(change > 0 ? change : 0);
                losses.Add(change < 0 ? -change : 0);
            }

            // Calculate initial average gain and loss
            double avgGain = gains.Take(period).Average();
            double avgLoss = losses.Take(period).Average();

            // Calculate first RSI
            double rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
            double rsi = 100 - 100 / (1 + rs);
            rsiValues.Add(rsi);

            // Calculate remaining RSI values
            for (int i = period + 1; i < prices.Count; i++)
            {
                avgGain = (avgGain * (period - 1) + gains[i - 1]) / period;
                avgLoss = (avgLoss * (period - 1) + losses[i - 1]) / period;

                rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                rsi = 100 - 100 / (1 + rs);
                rsiValues.Add(rsi);
            }

            return rsiValues;
        }
    }
}
