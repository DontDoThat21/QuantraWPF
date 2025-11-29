using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Quantra.Models;
using System.IO;
using System.Text.Json;
using System.Text;
using Quantra.CrossCutting.Monitoring;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    public class TechnicalIndicatorService : ITechnicalIndicatorService, IDisposable
    {
        private static readonly HttpClient client = new HttpClient { Timeout = TimeSpan.FromSeconds(25) };
        private readonly HistoricalDataService _historicalDataService;
        private readonly AlphaVantageService _alphaVantageService;
        private readonly string _apiKey;
        private readonly Dictionary<string, Func<string, string, Task<double>>> _indicators;

        // Dictionary of registered custom indicators
        private readonly Dictionary<string, IIndicator> _customIndicators;

        // Repository for custom indicator definitions (local in-memory fallback to avoid cross-project dependency)
        private readonly CustomIndicatorRepository _customIndicatorRepository;

        // Throttler for concurrent background tasks (local minimal implementation)
        private readonly ConcurrentTaskThrottler _taskThrottler;
        private readonly IMonitoringManager _monitoringManager;
        private bool _disposed = false;

        public TechnicalIndicatorService(AlphaVantageService alphaVantageService, UserSettingsService userSettingsService, LoggingService loggingService)
        {
            _alphaVantageService = alphaVantageService;

            // Get API key from database via AlphaVantageService
            _apiKey = AlphaVantageService.GetApiKey();
            if (string.IsNullOrWhiteSpace(_apiKey))
            {
                throw new InvalidOperationException("AlphaVantageApiKey not found in database settings profile.");
            }

            _historicalDataService = new HistoricalDataService(userSettingsService, loggingService);

            _indicators = new Dictionary<string, Func<string, string, Task<double>>>
            {
                { "TradingSignal", CalculateTradingSignal },
                { "BuySellSignal", CalculateBuySellSignal },
                // Add other indicators as needed
            };

            // Initialize custom indicators
            _customIndicators = new Dictionary<string, IIndicator>();
            _customIndicatorRepository = new CustomIndicatorRepository();

            // Initialize monitoring manager for performance profiling
            _monitoringManager = MonitoringManager.Instance;

            // Initialize task throttler with max degree of 6 for optimal performance
            _taskThrottler = new ConcurrentTaskThrottler(6);

            // Register this service with the ServiceLocator
            ServiceLocator.RegisterService<ITechnicalIndicatorService>(this);

            // Load saved custom indicators
            LoadSavedIndicatorsAsync().Wait();
        }


        public async Task<Dictionary<string, double>> CalculateIndicators(string symbol, string timeframe)
        {
            var results = new Dictionary<string, double>();
            foreach (var indicator in _indicators)
            {
                results[indicator.Key] = await indicator.Value(symbol, timeframe);
            }
            return results;
        }

        public async Task<bool> ValidateIndicators(Dictionary<string, double> indicators, string tradingAction)
        {
            if (indicators == null || !indicators.ContainsKey("TradingSignal") || !indicators.ContainsKey("BuySellSignal"))
                return false;

            var tradingSignal = indicators["TradingSignal"];
            var buySellSignal = indicators["BuySellSignal"];

            return tradingAction.ToUpper() switch
            {
                "BUY" => tradingSignal > 30 && buySellSignal > 0,
                "SELL" => tradingSignal < -30 && buySellSignal < 0,
                _ => false
            };
        }

        public async Task<double> GetTradingSignal(Dictionary<string, double> indicators)
        {
            if (indicators == null || !indicators.ContainsKey("TradingSignal"))
                return 0;

            return indicators["TradingSignal"];
        }

        private async Task<double> CalculateTradingSignal(string symbol, string timeframe)
        {
            // Implementation here
            return 0;
        }

        private async Task<double> CalculateBuySellSignal(string symbol, string timeframe)
        {
            // Implementation here
            return 0;
        }

        public async Task<(double macd, double signal)> GetMACD(string symbol, string timeframe)
        {
            try
            {
                string range = MapTimeframeToRange(timeframe);
                string interval = MapTimeframeToInterval(timeframe);

                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
                if (historicalData.Count < 35)
                    return (0, 0);

                var prices = historicalData.Select(h => h.Close).ToList();

                // Calculate MACD with standard settings (12, 26, 9)
                var ema12 = CalculateEMA(prices, 12);
                var ema26 = CalculateEMA(prices, 26);

                if (ema12.Count == 0 || ema26.Count == 0)
                    return (0, 0);

                // Calculate MACD line
                var macdLine = new List<double>();
                for (int i = 0; i < Math.Min(ema12.Count, ema26.Count); i++)
                {
                    macdLine.Add(ema12[i] - ema26[i]);
                }

                // Calculate signal line (9-day EMA of MACD line)
                var signalLine = CalculateEMA(macdLine, 9);

                if (macdLine.Count > 0 && signalLine.Count > 0)
                    return (macdLine.Last(), signalLine.Last());

                return (0, 0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error calculating MACD for {symbol}: {ex.Message}");
                return (0, 0);
            }
        }

        public async Task<double> GetVWAP(string symbol, string timeframe)
        {
            try
            {
                string range = MapTimeframeToRange(timeframe);
                string interval = MapTimeframeToInterval(timeframe);

                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
                if (historicalData.Count == 0)
                    return 0;

                // VWAP calculation
                double cumulativeTPV = 0; // Typical Price * Volume
                long cumulativeVolume = 0;

                foreach (var bar in historicalData)
                {
                    double typicalPrice = (bar.High + bar.Low + bar.Close) / 3;
                    cumulativeTPV += typicalPrice * bar.Volume;
                    cumulativeVolume += bar.Volume;
                }

                if (cumulativeVolume == 0)
                    return historicalData.Last().Close;

                return cumulativeTPV / cumulativeVolume;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error calculating VWAP for {symbol}: {ex.Message}");
                return 0;
            }
        }

        public async Task<double> GetROC(string symbol, string timeframe)
        {
            try
            {
                string range = MapTimeframeToRange(timeframe);
                string interval = MapTimeframeToInterval(timeframe);

                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
                if (historicalData.Count < 10)
                    return 0;

                int period = 10; // Standard ROC period
                double currentPrice = historicalData.Last().Close;
                double priceNPeriodsAgo = historicalData[historicalData.Count - period - 1].Close;

                return (currentPrice - priceNPeriodsAgo) / priceNPeriodsAgo * 100;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error calculating ROC for {symbol}: {ex.Message}");
                return 0;
            }
        }

        public async Task<(double high, double low)> GetHighsLows(string symbol, string timeframe)
        {
            try
            {
                string range = MapTimeframeToRange(timeframe);
                string interval = MapTimeframeToInterval(timeframe);

                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
                if (historicalData.Count == 0)
                    return (0, 0);

                double high = historicalData.Max(d => d.High);
                double low = historicalData.Min(d => d.Low);

                return (high, low);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error calculating highs/lows for {symbol}: {ex.Message}");
                return (0, 0);
            }
        }

        public async Task<(double bullPower, double bearPower)> GetBullBearPower(string symbol, string timeframe)
        {
            try
            {
                string range = MapTimeframeToRange(timeframe);
                string interval = MapTimeframeToInterval(timeframe);

                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
                if (historicalData.Count < 14)
                    return (0, 0);

                // Calculate 13-period EMA
                var prices = historicalData.Select(h => h.Close).ToList();
                var ema13 = CalculateEMA(prices, 13);

                if (ema13.Count == 0)
                    return (0, 0);

                double emaValue = ema13.Last();

                // Get today's high and low
                double high = historicalData.Last().High;
                double low = historicalData.Last().Low;

                double bullPower = high - emaValue; // Bull Power = High - EMA
                double bearPower = low - emaValue;  // Bear Power = Low - EMA

                return (bullPower, bearPower);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error calculating Bull/Bear Power for {symbol}: {ex.Message}");
                return (0, 0);
            }
        }

        public async Task<double> GetWilliamsR(string symbol, string timeframe)
        {
            try
            {
                string range = MapTimeframeToRange(timeframe);
                string interval = MapTimeframeToInterval(timeframe);

                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
                if (historicalData.Count < 14)
                    return 0;

                // Take the last 14 periods
                var lastNPeriods = historicalData.Skip(historicalData.Count - 14).ToList();

                double highestHigh = lastNPeriods.Max(d => d.High);
                double lowestLow = lastNPeriods.Min(d => d.Low);
                double currentClose = lastNPeriods.Last().Close;

                // Williams %R formula: ((Highest High - Close) / (Highest High - Lowest Low)) * -100
                if (highestHigh - lowestLow == 0)
                    return -50; // Middle value when there's no range

                return (highestHigh - currentClose) / (highestHigh - lowestLow) * -100;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error calculating Williams %R for {symbol}: {ex.Message}");
                return 0;
            }
        }

        private string MapTimeframeToRange(string timeframe)
        {
            switch (timeframe.ToLower())
            {
                case "1min":
                case "5min":
                case "15min":
                case "30min":
                    return "1d";
                case "1hour":
                    return "5d";
                case "4hour":
                    return "1mo";
                case "1day":
                    return "1y";
                case "1week":
                    return "2y";
                case "1month":
                    return "5y";
                default:
                    return "1mo";
            }
        }

        private string MapTimeframeToInterval(string timeframe)
        {
            switch (timeframe.ToLower())
            {
                case "1min":
                    return "1m";
                case "5min":
                    return "5m";
                case "15min":
                    return "15m";
                case "30min":
                    return "30m";
                case "1hour":
                    return "60m";
                case "4hour":
                    return "1d";
                case "1day":
                    return "1d";
                case "1week":
                    return "1wk";
                case "1month":
                    return "1mo";
                default:
                    return "1d";
            }
        }

        /// <summary>
        /// Gets comprehensive indicator data for a symbol for use in trading prediction models
        /// </summary>
        /// <param name="symbol">Stock symbol to analyze</param>
        /// <param name="timeframe">Timeframe for analysis (e.g. "1day", "1week")</param>
        /// <returns>Dictionary of indicator names and values</returns>
        public async Task<Dictionary<string, double>> GetIndicatorsForPrediction(string symbol, string timeframe)
        {
            var (result, _) = await _monitoringManager.RecordExecutionTimeAsync($"GetIndicatorsForPrediction_{symbol}_{timeframe}", async () =>
            {
                var result = new Dictionary<string, double>();
                try
                {
                    var (rsi, rsiDuration) = await _monitoringManager.RecordExecutionTimeAsync($"GetRSI_{symbol}_{timeframe}", async () => await GetRSI(symbol, timeframe));
                    var (adx, adxDuration) = await _monitoringManager.RecordExecutionTimeAsync($"GetADX_{symbol}_{timeframe}", async () => await GetADX(symbol, timeframe));
                    var (atr, atrDuration) = await _monitoringManager.RecordExecutionTimeAsync($"GetATR_{symbol}_{timeframe}", async () => await GetATR(symbol, timeframe));
                    var (momentum, momentumDuration) = await _monitoringManager.RecordExecutionTimeAsync($"GetMomentum_{symbol}_{timeframe}", async () => await GetMomentum(symbol, timeframe));
                    var (stochastic, stochasticDuration) = await _monitoringManager.RecordExecutionTimeAsync($"GetStochastic_{symbol}_{timeframe}", async () => await GetStochastic(symbol, timeframe));
                    var (obv, obvDuration) = await _monitoringManager.RecordExecutionTimeAsync($"GetOBV_{symbol}_{timeframe}", async () => await GetOBV(symbol, timeframe));
                    var (mfi, mfiDuration) = await _monitoringManager.RecordExecutionTimeAsync($"GetMFI_{symbol}_{timeframe}", async () => await GetMFI(symbol, timeframe));

                    // Extract k and d from stochastic result
                    var (k, d) = stochastic;

                    // Add core indicators
                    result["RSI"] = rsi;
                    result["ADX"] = adx;
                    result["ATR"] = atr;
                    result["MomentumScore"] = momentum;
                    result["StochK"] = k;
                    result["StochD"] = d;
                    result["OBV"] = obv;
                    result["MFI"] = mfi;

                    // Add more derived indicators
                    result["TradingSignal"] = CalculateTradingSignal(rsi, adx, momentum);
                    result["BuySellSignal"] = CalculateBuySellSignal(rsi, k, d);
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", $"Failed to get indicators for {symbol}", ex.ToString());
                }

                return result;
            });

            return result;
        }

        /// <summary>
        /// Gets comprehensive indicator values for multiple symbols in batches for better performance
        /// </summary>
        /// <param name="symbols">List of symbols to analyze</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <returns>Dictionary mapping symbols to their indicator values</returns>
        public async Task<Dictionary<string, Dictionary<string, double>>> GetIndicatorsForPredictionBatchAsync(
            List<string> symbols, string timeframe = "5min")
        {
            try
            {
                // Split into batches of 5 symbols to respect API rate limits
                int batchSize = 5;
                var result = new Dictionary<string, Dictionary<string, double>>();

                //DatabaseMonolith.Log("Info", $"Fetching indicators for {symbols.Count} symbols in batches of {batchSize} with throttling");

                for (int i = 0; i < symbols.Count; i += batchSize)
                {
                    var batch = symbols.Skip(i).Take(batchSize).ToList();

                    // Process batch concurrently with throttling to prevent thread pool exhaustion
                    var batchTaskFactories = batch.Select(symbol =>
                        new Func<Task<KeyValuePair<string, Dictionary<string, double>>>>(() =>
                            ProcessSymbolForBatch(symbol, timeframe)));

                    var batchResults = await _taskThrottler.ExecuteThrottledAsync(batchTaskFactories);

                    // Combine results
                    foreach (var kvp in batchResults)
                    {
                        result[kvp.Key] = kvp.Value;
                    }

                    // Respect API rate limits between batches
                    if (i + batchSize < symbols.Count)
                    {
                        await Task.Delay(2000); // 2-second delay between batches
                    }
                }

                //DatabaseMonolith.Log("Info", $"Successfully fetched indicators for {result.Count} symbols in batch with throttling");
                return result;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error batching technical indicator requests: {ex.Message}", ex.ToString());
                return new Dictionary<string, Dictionary<string, double>>();
            }
        }

        /// <summary>
        /// Helper method to process a single symbol for batch operations
        /// </summary>
        /// <param name="symbol">Symbol to process</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <returns>Key-value pair with symbol and its indicators</returns>
        private async Task<KeyValuePair<string, Dictionary<string, double>>> ProcessSymbolForBatch(string symbol, string timeframe)
        {
            try
            {
                var indicators = await GetIndicatorsForPrediction(symbol, timeframe);
                return new KeyValuePair<string, Dictionary<string, double>>(symbol, indicators);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Failed to get indicators for symbol {symbol}", ex.ToString());
                return new KeyValuePair<string, Dictionary<string, double>>(symbol, new Dictionary<string, double>());
            }
        }

        /// <summary>
        /// Gets a comprehensive set of indicator values for algorithmic trading decisions
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Dictionary with indicator names and values</returns>
        public async Task<Dictionary<string, double>> GetAlgorithmicTradingSignals(string symbol)
        {
            var result = new Dictionary<string, double>();
            try
            {
                // Get technical indicators with 1min timeframe for immediate signals
                var rsi = await GetRSI(symbol, "1min");
                var adx = await GetADX(symbol, "1min");
                var momentum = await GetMomentum(symbol, "1min");
                var (k, d) = await GetStochastic(symbol, "1min");
                var obv = await GetOBV(symbol, "1min");
                var mfi = await GetMFI(symbol, "1min");

                // Calculate composite signals
                double shortTermSignal = CalculateShortTermSignal(rsi, k, d);
                double trendStrength = CalculateTrendStrength(adx, momentum);
                double overallScore = (shortTermSignal + trendStrength) / 2.0;

                // Add signals to result
                result["ShortTermSignal"] = shortTermSignal;
                result["TrendStrength"] = trendStrength;
                result["OverallScore"] = overallScore;
                result["TradingSignal"] = Math.Sign(overallScore) * Math.Min(100, Math.Abs(overallScore * 100));
                result["OBV"] = obv; // Add OBV to the signals
                result["MFI"] = mfi; // Add MFI to the signals
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to get algorithmic signals for {symbol}", ex.ToString());
            }

            return result;
        }

        private double CalculateShortTermSignal(double rsi, double k, double d)
        {
            double signal = 0;

            // RSI signals (-1 to 1)
            if (rsi < 30) signal += 1;      // Oversold
            else if (rsi > 70) signal -= 1;  // Overbought
            else signal += (50 - rsi) / 20;  // Neutral zone signal

            // Stochastic signals
            if (k < 20 && k > d) signal += 0.5;  // Bullish stoch
            if (k > 80 && k < d) signal -= 0.5;  // Bearish stoch

            return Math.Max(-1, Math.Min(1, signal));  // Clamp between -1 and 1
        }

        private double CalculateTrendStrength(double adx, double momentum)
        {
            double strength = 0;

            // ADX trend strength (0 to 1)
            strength += adx > 25 ? (adx - 25) / 75 : 0;

            // Momentum contribution (-1 to 1)
            strength += Math.Sign(momentum) * Math.Min(1, Math.Abs(momentum) / 100);

            return Math.Max(-1, Math.Min(1, strength / 2));  // Average and clamp
        }

        private double CalculateTradingSignal(double rsi, double adx, double momentum)
        {
            double signal = 0;

            // RSI contribution
            if (rsi < 30) signal += 30;        // Oversold
            else if (rsi > 70) signal -= 30;   // Overbought
            else signal += (50 - rsi) / 2;     // Neutral zone contribution

            // ADX contribution (trend strength)
            if (adx > 25)
            {
                double trendMagnitude = Math.Min(1, (adx - 25) / 25);  // 0 to 1 based on ADX 25-50
                signal *= 1 + trendMagnitude;  // Amplify signal based on trend strength
            }

            // Momentum contribution
            signal += momentum * 0.3;  // Scale momentum's influence

            return Math.Max(-100, Math.Min(100, signal));  // Clamp to -100 to 100
        }

        private double CalculateBuySellSignal(double rsi, double k, double d)
        {
            double signal = 0;

            // RSI signals
            if (rsi < 30) signal += 0.5;       // Oversold - buy signal
            else if (rsi > 70) signal -= 0.5;  // Overbought - sell signal

            // Stochastic signals
            if (k < 20 && k > d) signal += 0.5;  // Bullish crossover
            if (k > 80 && k < d) signal -= 0.5;  // Bearish crossover

            return Math.Max(-1, Math.Min(1, signal));  // Clamp to -1 to 1
        }

        public async Task<double> GetRSI(string symbol, string interval = "1day")
        {
            try
            {
                string range = MapTimeframeToRange(interval);
                string mappedInterval = MapTimeframeToInterval(interval);

                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 15) // Need at least 15 data points for RSI calculation
                    return 50; // Neutral default

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Extract closing prices
                var closingPrices = historicalData.Select(h => h.Close).ToList();

                // Calculate RSI using internal method
                var rsiValues = CalculateRSI(closingPrices, 14);

                // Return the latest RSI value
                var latestRsi = rsiValues.LastOrDefault(r => !double.IsNaN(r));
                return double.IsNaN(latestRsi) ? 50 : latestRsi;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate RSI for {symbol}", ex.ToString());
                return 50; // Neutral default
            }
        }

        public async Task<double> GetADX(string symbol, string interval = "1day")
        {
            try
            {
                string range = MapTimeframeToRange(interval);
                string mappedInterval = MapTimeframeToInterval(interval);

                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 30) // Need enough data for ADX calculation
                    return 25; // Neutral default

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Extract price data
                var highPrices = historicalData.Select(h => h.High).ToList();
                var lowPrices = historicalData.Select(h => h.Low).ToList();
                var closePrices = historicalData.Select(h => h.Close).ToList();

                // Calculate ADX using internal method
                var adxValues = CalculateADX(highPrices, lowPrices, closePrices, 14);

                // Return the latest ADX value
                var latestAdx = adxValues.LastOrDefault(a => !double.IsNaN(a));
                return double.IsNaN(latestAdx) ? 25 : latestAdx;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate ADX for {symbol}", ex.ToString());
                return 25; // Neutral default
            }
        }

        public async Task<double> GetATR(string symbol, string interval = "1day")
        {
            try
            {
                string range = MapTimeframeToRange(interval);
                string mappedInterval = MapTimeframeToInterval(interval);

                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 16) // Need enough data for ATR calculation
                    return 1.0; // Default value

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Extract price data
                var highPrices = historicalData.Select(h => h.High).ToList();
                var lowPrices = historicalData.Select(h => h.Low).ToList();
                var closePrices = historicalData.Select(h => h.Close).ToList();

                // Calculate ATR using internal method
                var atrValues = CalculateATR(highPrices, lowPrices, closePrices, 14);

                // Return the latest ATR value
                var latestAtr = atrValues.LastOrDefault(a => !double.IsNaN(a));
                return double.IsNaN(latestAtr) ? 1.0 : latestAtr;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate ATR for {symbol}", ex.ToString());
                return 1.0; // Default value
            }
        }

        public async Task<double> GetMomentum(string symbol, string interval = "1day")
        {
            try
            {
                string range = MapTimeframeToRange(interval);
                string mappedInterval = MapTimeframeToInterval(interval);

                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 12) // Need enough data for momentum calculation
                    return 0; // Neutral default

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Extract closing prices
                var closingPrices = historicalData.Select(h => h.Close).ToList();

                // Calculate ROC (Rate of Change) as momentum using internal method
                var rocValues = CalculateROC(closingPrices, 10);

                // Return the latest ROC value
                var latestRoc = rocValues.LastOrDefault(r => !double.IsNaN(r));
                return double.IsNaN(latestRoc) ? 0 : latestRoc;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate momentum for {symbol}", ex.ToString());
                return 0; // Neutral default
            }
        }

        public async Task<(double K, double D)> GetStochastic(string symbol, string interval = "1day")
        {
            try
            {
                string range = MapTimeframeToRange(interval);
                string mappedInterval = MapTimeframeToInterval(interval);

                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 16) // Need enough data for stochastic calculation
                    return (50, 50); // Neutral default

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Extract price data
                var highPrices = historicalData.Select(h => h.High).ToList();
                var lowPrices = historicalData.Select(h => h.Low).ToList();
                var closePrices = historicalData.Select(h => h.Close).ToList();

                // Calculate Stochastic using internal method (14-period, 3-smoothing, 3-D period)
                var stochValues = CalculateStochastic(highPrices, lowPrices, closePrices, 14, 3, 3);

                // Return the latest K and D values
                var latestK = stochValues.K.LastOrDefault(k => !double.IsNaN(k));
                var latestD = stochValues.D.LastOrDefault(d => !double.IsNaN(d));

                return (double.IsNaN(latestK) ? 50 : latestK, double.IsNaN(latestD) ? 50 : latestD);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate stochastic for {symbol}", ex.ToString());
                return (50, 50); // Neutral default
            }
        }

        // Simulating main branch might have a different implementation or missing OBV
        public async Task<double> GetMFI(string symbol, string interval = "1day")
        {
            try
            {
                string range = MapTimeframeToRange(interval);
                string mappedInterval = MapTimeframeToInterval(interval);

                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 14)
                    return 50; // Default value

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Take the last 14 periods
                var periods = historicalData.Skip(Math.Max(0, historicalData.Count - 14)).ToList();

                double positiveMoneyFlow = 0;
                double negativeMoneyFlow = 0;

                for (int i = 1; i < periods.Count; i++)
                {
                    // Calculate typical price
                    double currentTypicalPrice = (periods[i].High + periods[i].Low + periods[i].Close) / 3;
                    double prevTypicalPrice = (periods[i - 1].High + periods[i - 1].Low + periods[i - 1].Close) / 3;

                    // Calculate raw money flow
                    double rawMoneyFlow = currentTypicalPrice * periods[i].Volume;

                    // Add to positive/negative money flow
                    if (currentTypicalPrice > prevTypicalPrice)
                        positiveMoneyFlow += rawMoneyFlow;
                    else if (currentTypicalPrice < prevTypicalPrice)
                        negativeMoneyFlow += rawMoneyFlow;
                }

                // Calculate money flow ratio
                double moneyFlowRatio = negativeMoneyFlow == 0 ? 100 : positiveMoneyFlow / negativeMoneyFlow;

                // Calculate MFI
                double mfi = 100 - 100 / (1 + moneyFlowRatio);

                return mfi;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate MFI for {symbol}", ex.ToString());
                return 50; // Default value
            }
        }

        public async Task<(double StochK, double StochD)> GetSTOCH(string symbol, string timeframe)
        {
            try
            {
                string range = MapTimeframeToRange(timeframe);
                string mappedInterval = MapTimeframeToInterval(timeframe);

                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 16) // Need enough data for stochastic calculation
                    return (50, 50); // Neutral default

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Extract price data
                var highPrices = historicalData.Select(h => h.High).ToList();
                var lowPrices = historicalData.Select(h => h.Low).ToList();
                var closePrices = historicalData.Select(h => h.Close).ToList();

                // Calculate Stochastic using internal method (14-period, 3-smoothing, 3-D period)
                var stochValues = CalculateStochastic(highPrices, lowPrices, closePrices, 14, 3, 3);

                // Return the latest K and D values
                var latestK = stochValues.K.LastOrDefault(k => !double.IsNaN(k));
                var latestD = stochValues.D.LastOrDefault(d => !double.IsNaN(d));

                return (double.IsNaN(latestK) ? 50 : latestK, double.IsNaN(latestD) ? 50 : latestD);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate STOCH for {symbol}", ex.ToString());
                return (50, 50); // Neutral default
            }
        }

        public async Task<double> GetSTOCHRSI(string symbol, string timeframe)
        {
            try
            {
                // Calculate Stochastic RSI using RSI values
                var closingPrices = await _alphaVantageService.GetHistoricalClosingPricesAsync(symbol, 20); // 14+6 for window
                if (closingPrices == null || closingPrices.Count < 15)
                    return 0.5; // Neutral
                var rsiList = new List<double>();
                for (int i = 0; i < closingPrices.Count - 13; i++)
                {
                    var window = closingPrices.Skip(i).Take(14).ToList();
                    double avgGain = 0, avgLoss = 0;
                    for (int j = 1; j < window.Count; j++)
                    {
                        double change = window[j] - window[j - 1];
                        if (change > 0) avgGain += change;
                        else avgLoss -= change;
                    }
                    avgGain /= 14;
                    avgLoss /= 14;
                    double rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                    double rsi = 100 - 100 / (1 + rs);
                    rsiList.Add(rsi);
                }
                double minRsi = rsiList.Min();
                double maxRsi = rsiList.Max();
                double lastRsi = rsiList.Last();
                if (maxRsi - minRsi == 0) return 0.5;
                return (lastRsi - minRsi) / (maxRsi - minRsi);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to get StochRSI for {symbol}", ex.ToString());
                return 0.5; // Neutral
            }
        }

        public async Task<double> GetUltimateOscillator(string symbol, string timeframe)
        {
            try
            {
                string range = MapTimeframeToRange(timeframe);
                string mappedInterval = MapTimeframeToInterval(timeframe);

                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 30) // Need enough data for Ultimate Oscillator calculation
                    return 50; // Neutral default

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Extract price data
                var highPrices = historicalData.Select(h => h.High).ToList();
                var lowPrices = historicalData.Select(h => h.Low).ToList();
                var closePrices = historicalData.Select(h => h.Close).ToList();

                // Calculate Ultimate Oscillator using internal method
                var uoValues = CalculateUltimateOscillator(highPrices, lowPrices, closePrices, 7, 14, 28);

                // Return the latest Ultimate Oscillator value
                var latestUo = uoValues.LastOrDefault(u => !double.IsNaN(u));
                return double.IsNaN(latestUo) ? 50 : latestUo;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate Ultimate Oscillator for {symbol}", ex.ToString());
                return 50; // Neutral default
            }
        }

        /// <summary>
        /// Calculates On-Balance Volume (OBV), a momentum indicator that uses volume flow to predict changes in stock price
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <param name="interval">Data interval (1day, 1min, etc.)</param>
        /// <returns>The current OBV value</returns>
        public async Task<double> GetOBV(string symbol, string interval = "1day")
        {
            try
            {
                string range = MapTimeframeToRange(interval);
                string mappedInterval = MapTimeframeToInterval(interval);
                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 2)
                    return 0;
                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();
                // Calculate OBV
                double obv = 0; // Start OBV at 0
                for (int i = 1; i < historicalData.Count; i++)
                {
                    var currentClose = historicalData[i].Close;
                    var previousClose = historicalData[i - 1].Close;
                    var currentVolume = historicalData[i].Volume;
                    if (currentClose > previousClose)
                    {
                        // Price up - add volume
                        obv += currentVolume;
                    }
                    else if (currentClose < previousClose)
                    {
                        // Price down - subtract volume
                        obv -= currentVolume;
                    }
                    // Price unchanged - OBV remains the same (no change)
                }
                return obv;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate OBV for {symbol}", ex.ToString());
                return 0; // Default value
            }
        }

        public async Task<double> GetParabolicSAR(string symbol, string interval = "1day")
        {
            try
            {
                string range = MapTimeframeToRange(interval);
                string intervalMapped = MapTimeframeToInterval(interval);

                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, intervalMapped);
                if (historicalData.Count < 3)
                    return 0;

                // Use default parameters for SAR calculation
                double accelerationFactor = 0.02;
                double maxAccelerationFactor = 0.2;

                // Calculate the SAR values
                var (sarValues, _, _) = CalculateParabolicSAR(historicalData, accelerationFactor, maxAccelerationFactor);

                // Return the latest valid SAR value
                if (sarValues != null && sarValues.Count > 0)
                {
                    return sarValues.Last();
                }

                return 0;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to get Parabolic SAR for {symbol}: {ex.Message}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Calculate Parabolic SAR values for a price series
        /// </summary>
        /// <param name="prices">Historical price data</param>
        /// <param name="initialAF">Initial acceleration factor</param>
        /// <param name="maxAF">Maximum acceleration factor</param>
        /// <returns>List of SAR values, extreme points, and uptrend status</returns>
        private (List<double> sarValues, List<double> extremePoints, List<bool> isUptrend)
            CalculateParabolicSAR(List<HistoricalPrice> prices, double initialAF, double maxAF)
        {
            if (prices.Count < 3)
                return (null, null, null);

            var sarValues = new List<double>();
            var extremePoints = new List<double>();
            var isUptrend = new List<bool>();

            // Initialize with placeholders for the first two entries
            sarValues.Add(double.NaN);
            sarValues.Add(double.NaN);
            extremePoints.Add(double.NaN);
            extremePoints.Add(double.NaN);
            isUptrend.Add(false);
            isUptrend.Add(false);

            // Determine initial trend (based on closing prices)
            bool currentUptrend = prices[1].Close > prices[0].Close;

            // Initial SAR value
            double sar = currentUptrend ? prices[0].Low : prices[0].High;

            // Initial extreme point
            double ep = currentUptrend ? prices[1].High : prices[1].Low;

            // Initial acceleration factor
            double af = initialAF;

            // Calculate SAR for each subsequent period
            for (int i = 2; i < prices.Count; i++)
            {
                // Prior SAR
                double priorSAR = sar;

                // Calculate current SAR
                sar = priorSAR + af * (ep - priorSAR);

                // Ensure SAR doesn't go beyond price action limits
                if (currentUptrend)
                {
                    // In uptrend, SAR must be below the current Low and previous Low
                    sar = Math.Min(sar, Math.Min(prices[i - 1].Low, prices[i - 2].Low));
                }
                else
                {
                    // In downtrend, SAR must be above the current High and previous High
                    sar = Math.Max(sar, Math.Max(prices[i - 1].High, prices[i - 2].High));
                }

                // Check for trend reversal
                bool potentialReversal = currentUptrend && prices[i].Low < sar ||
                                        !currentUptrend && prices[i].High > sar;

                if (potentialReversal)
                {
                    // Reverse the trend
                    currentUptrend = !currentUptrend;

                    // Reset acceleration factor
                    af = initialAF;

                    // Set new extreme point
                    ep = currentUptrend ? prices[i].High : prices[i].Low;

                    // Set new SAR at the prior extreme point
                    sar = ep;
                }
                else
                {
                    // Trend continues
                    if (currentUptrend)
                    {
                        // Update extreme point if a new high is found
                        if (prices[i].High > ep)
                        {
                            ep = prices[i].High;
                            af = Math.Min(af + initialAF, maxAF); // Increase acceleration factor
                        }
                    }
                    else
                    {
                        // Update extreme point if a new low is found
                        if (prices[i].Low < ep)
                        {
                            ep = prices[i].Low;
                            af = Math.Min(af + initialAF, maxAF); // Increase acceleration factor
                        }
                    }
                }

                sarValues.Add(sar);
                extremePoints.Add(ep);
                isUptrend.Add(currentUptrend);
            }

            return (sarValues, extremePoints, isUptrend);
        }

        public async Task<double> GetCCI(string symbol, string interval = "1day")
        {
            try
            {
                string range = MapTimeframeToRange(interval);
                string mappedInterval = MapTimeframeToInterval(interval);

                // Get historical price data
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, mappedInterval);
                if (historicalData.Count < 22) // Need enough data for CCI calculation (20 period default)
                    return 0; // Neutral default

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Extract price data
                var highPrices = historicalData.Select(h => h.High).ToList();
                var lowPrices = historicalData.Select(h => h.Low).ToList();
                var closePrices = historicalData.Select(h => h.Close).ToList();

                // Calculate CCI using internal method
                var cciValues = CalculateCCI(highPrices, lowPrices, closePrices, 20);

                // Return the latest CCI value
                var latestCci = cciValues.LastOrDefault(c => !double.IsNaN(c));
                return double.IsNaN(latestCci) ? 0 : latestCci;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate CCI for {symbol}", ex.ToString());
                return 0; // Neutral default
            }
        }

        #region Indicator Correlation Analysis

        /// <summary>
        /// Calculates correlation between two technical indicators
        /// </summary>
        /// <param name="symbol">Symbol to analyze</param>
        /// <param name="firstIndicator">First indicator type (e.g., RSI, MACD, etc.)</param>
        /// <param name="secondIndicator">Second indicator type</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <param name="dataPoints">Number of historical data points to use</param>
        /// <returns>Correlation result</returns>
        public async Task<IndicatorCorrelationResult> CalculateIndicatorCorrelation(
            string symbol, string firstIndicator, string secondIndicator,
            string timeframe = "1day", int dataPoints = 30)
        {
            try
            {
                // Get historical data for both indicators
                var firstData = await GetHistoricalIndicatorData(symbol, firstIndicator, timeframe, dataPoints);
                var secondData = await GetHistoricalIndicatorData(symbol, secondIndicator, timeframe, dataPoints);

                if (firstData.Count == 0 || secondData.Count == 0)
                {
                    return new IndicatorCorrelationResult
                    {
                        Symbol = symbol,
                        FirstIndicator = firstIndicator,
                        SecondIndicator = secondIndicator,
                        CorrelationCoefficient = 0,
                        ConfidenceLevel = 0,
                        DataPointsCount = 0,
                        Timeframe = timeframe
                    };
                }

                // Calculate correlation
                var correlation = CalculatePearsonCorrelation(firstData, secondData);

                // Calculate confidence level based on sample size
                double confidenceLevel = CalculateConfidenceLevel(correlation, Math.Min(firstData.Count, secondData.Count));

                return new IndicatorCorrelationResult
                {
                    Symbol = symbol,
                    FirstIndicator = firstIndicator,
                    SecondIndicator = secondIndicator,
                    CorrelationCoefficient = correlation,
                    ConfidenceLevel = confidenceLevel,
                    DataPointsCount = Math.Min(firstData.Count, secondData.Count),
                    FirstIndicatorValues = firstData,
                    SecondIndicatorValues = secondData,
                    Timeframe = timeframe
                };
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate indicator correlation for {symbol}: {firstIndicator} vs {secondIndicator}", ex.ToString());

                return new IndicatorCorrelationResult
                {
                    Symbol = symbol,
                    FirstIndicator = firstIndicator,
                    SecondIndicator = secondIndicator,
                    CorrelationCoefficient = 0,
                    ConfidenceLevel = 0,
                    DataPointsCount = 0,
                    Timeframe = timeframe
                };
            }
        }

        /// <summary>
        /// Calculates correlation between all relevant indicator pairs
        /// </summary>
        /// <param name="symbol">Symbol to analyze</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <param name="dataPoints">Number of historical data points to use</param>
        /// <returns>List of correlation results</returns>
        public async Task<List<IndicatorCorrelationResult>> CalculateAllIndicatorCorrelations(
            string symbol, string timeframe = "1day", int dataPoints = 30)
        {
            var results = new List<IndicatorCorrelationResult>();

            try
            {
                // Define key indicators to analyze
                var indicators = new List<string> { "RSI", "MACD", "ADX", "StochRSI", "VWAP", "OBV", "MFI", "BB_Width" };

                // Create task factories for all indicator pairs
                var correlationTaskFactories = new List<Func<Task<IndicatorCorrelationResult>>>();

                for (int i = 0; i < indicators.Count; i++)
                {
                    for (int j = i + 1; j < indicators.Count; j++)
                    {
                        var firstIndicator = indicators[i];
                        var secondIndicator = indicators[j];

                        correlationTaskFactories.Add(() =>
                            CalculateIndicatorCorrelation(symbol, firstIndicator, secondIndicator, timeframe, dataPoints));
                    }
                }

                // Execute all correlation calculations with throttling
                var correlationResults = await _taskThrottler.ExecuteThrottledAsync(correlationTaskFactories);
                results.AddRange(correlationResults);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate all indicator correlations for {symbol}", ex.ToString());
            }

            return results;
        }

        /// <summary>
        /// Identifies confirmation patterns among multiple indicators
        /// </summary>
        /// <param name="symbol">Symbol to analyze</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <returns>List of confirmation patterns</returns>
        public async Task<List<IndicatorConfirmationPattern>> FindConfirmationPatterns(string symbol, string timeframe = "1day")
        {
            var patterns = new List<IndicatorConfirmationPattern>();

            try
            {
                // Get current indicator values for analysis
                var indicators = await GetIndicatorsForPrediction(symbol, timeframe);

                // Get historical indicator correlations for additional context
                var correlations = await CalculateAllIndicatorCorrelations(symbol, timeframe, 20);

                // Identify trend-momentum confirmation pattern
                var trendMomentumPattern = IdentifyTrendMomentumPattern(symbol, indicators, correlations);
                if (trendMomentumPattern != null)
                {
                    patterns.Add(trendMomentumPattern);
                }

                // Identify overbought/oversold confirmation pattern
                var overextendedPattern = IdentifyOverextendedPattern(symbol, indicators, correlations);
                if (overextendedPattern != null)
                {
                    patterns.Add(overextendedPattern);
                }

                // Identify volume-price confirmation pattern
                var volumePricePattern = IdentifyVolumePricePattern(symbol, indicators, correlations);
                if (volumePricePattern != null)
                {
                    patterns.Add(volumePricePattern);
                }

                // Add more pattern identifications as needed
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to find confirmation patterns for {symbol}", ex.ToString());
            }

            return patterns;
        }

        /// <summary>
        /// Gets historical data for a specific indicator
        /// </summary>
        /// <param name="symbol">Symbol to analyze</param>
        /// <param name="indicatorType">Type of indicator</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <param name="dataPoints">Number of data points to retrieve</param>
        /// <returns>List of historical indicator values</returns>
        private async Task<List<double>> GetHistoricalIndicatorData(
            string symbol, string indicatorType, string timeframe, int dataPoints)
        {
            try
            {
                // First check if we can use DatabaseMonolith's method for certain indicators
                if (indicatorType == "RSI" || indicatorType == "MACD" || indicatorType == "ADX" ||
                    indicatorType == "ROC" || indicatorType == "BB_Width")
                {
                    return await _alphaVantageService.GetHistoricalIndicatorData(symbol, indicatorType);
                }

                // For indicators not directly available from DatabaseMonolith, 
                // we need custom handling for each indicator type
                switch (indicatorType.ToUpperInvariant())
                {
                    case "STOCHRSI":
                        // For StochRSI, get RSI values first then calculate StochRSI
                        var rsiValues = await _alphaVantageService.GetHistoricalIndicatorData(symbol, "RSI");
                        return CalculateStochRSIFromRSI(rsiValues);

                    case "VWAP":
                        // Get price and volume data and calculate VWAP
                        string range = MapTimeframeToRange(timeframe);
                        string interval = MapTimeframeToInterval(timeframe);

                        var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
                        if (historicalData.Count == 0)
                            return new List<double>();

                        return CalculateHistoricalVWAP(historicalData);

                    case "OBV":
                        // Calculate historical OBV values
                        range = MapTimeframeToRange(timeframe);
                        interval = MapTimeframeToInterval(timeframe);

                        historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
                        if (historicalData.Count == 0)
                            return new List<double>();

                        return CalculateHistoricalOBV(historicalData);

                    case "MFI":
                        // Calculate historical MFI values
                        range = MapTimeframeToRange(timeframe);
                        interval = MapTimeframeToInterval(timeframe);

                        historicalData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);
                        if (historicalData.Count < 14)
                            return new List<double>();

                        return CalculateHistoricalMFI(historicalData);

                    default:
                        // For unknown indicators, return empty list
                        //DatabaseMonolith.Log("Warning", $"Unknown indicator type for historical data: {indicatorType}", "");
                        return new List<double>();
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to get historical indicator data for {indicatorType}", ex.ToString());
                return new List<double>();
            }
        }

        /// <summary>
        /// Identifies confirmation patterns based on trend and momentum indicators
        /// </summary>
        private IndicatorConfirmationPattern IdentifyTrendMomentumPattern(
            string symbol, Dictionary<string, double> indicators, List<IndicatorCorrelationResult> correlations)
        {
            try
            {
                if (indicators == null || !indicators.ContainsKey("ADX") ||
                    !indicators.ContainsKey("RSI") || !indicators.ContainsKey("MomentumScore"))
                {
                    return null;
                }

                double adx = indicators["ADX"];
                double rsi = indicators["RSI"];
                double momentum = indicators["MomentumScore"];

                // Only consider strong trends
                if (adx < 25)
                {
                    return null;
                }

                string direction;
                double strength;

                // Determine signal direction
                if (rsi > 60 && momentum > 0)
                {
                    direction = "Bullish";
                    strength = (rsi - 60) / 40.0 * (momentum / 100.0) * (adx / 100.0);
                }
                else if (rsi < 40 && momentum < 0)
                {
                    direction = "Bearish";
                    strength = (40 - rsi) / 40.0 * (Math.Abs(momentum) / 100.0) * (adx / 100.0);
                }
                else
                {
                    return null; // No clear pattern
                }

                // Find relevant correlations
                var supportingCorrelations = correlations.Where(c =>
                    c.FirstIndicator == "ADX" && c.SecondIndicator == "RSI" ||
                    c.FirstIndicator == "RSI" && c.SecondIndicator == "MACD")
                    .ToList();

                // Create pattern
                var pattern = new IndicatorConfirmationPattern
                {
                    Symbol = symbol,
                    PatternType = "Trend-Momentum Confirmation",
                    SignalDirection = direction,
                    ConfirmationStrength = Math.Min(1.0, strength),
                    ConfirmingIndicators = new List<string> { "ADX", "RSI", "Momentum" },
                    IndicatorValues = new Dictionary<string, double>
                    {
                        { "ADX", adx },
                        { "RSI", rsi },
                        { "Momentum", momentum }
                    },
                    SupportingCorrelations = supportingCorrelations,
                    TimeHorizon = adx > 40 ? "Medium-term" : "Short-term",
                    Reliability = adx / 100.0
                };

                return pattern;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error identifying trend-momentum pattern for {symbol}", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Identifies confirmation patterns for overbought/oversold conditions
        /// </summary>
        private IndicatorConfirmationPattern IdentifyOverextendedPattern(
            string symbol, Dictionary<string, double> indicators, List<IndicatorCorrelationResult> correlations)
        {
            try
            {
                if (indicators == null || !indicators.ContainsKey("RSI") ||
                    !indicators.ContainsKey("StochK") || !indicators.ContainsKey("StochD"))
                {
                    return null;
                }

                double rsi = indicators["RSI"];
                double stochK = indicators["StochK"];
                double stochD = indicators["StochD"];
                double mfi = indicators.ContainsKey("MFI") ? indicators["MFI"] : 50;

                string direction;
                double strength;

                // Check for oversold condition
                if (rsi < 30 && stochK < 20 && stochD < 20 && mfi < 30)
                {
                    direction = "Bullish";
                    strength = ((30 - rsi) / 30.0 + (20 - stochK) / 20.0 + (30 - mfi) / 30.0) / 3.0;
                }
                // Check for overbought condition
                else if (rsi > 70 && stochK > 80 && stochD > 80 && mfi > 70)
                {
                    direction = "Bearish";
                    strength = ((rsi - 70) / 30.0 + (stochK - 80) / 20.0 + (mfi - 70) / 30.0) / 3.0;
                }
                else
                {
                    return null; // No clear overextended pattern
                }

                // Find supporting correlations
                var supportingCorrelations = correlations.Where(c =>
                    c.FirstIndicator == "RSI" && c.SecondIndicator == "MFI" ||
                    c.FirstIndicator == "RSI" && c.SecondIndicator == "StochRSI")
                    .ToList();

                // Create pattern
                var pattern = new IndicatorConfirmationPattern
                {
                    Symbol = symbol,
                    PatternType = direction == "Bullish" ? "Oversold Confirmation" : "Overbought Confirmation",
                    SignalDirection = direction,
                    ConfirmationStrength = Math.Min(1.0, strength),
                    ConfirmingIndicators = new List<string> { "RSI", "Stochastic", "MFI" },
                    IndicatorValues = new Dictionary<string, double>
                    {
                        { "RSI", rsi },
                        { "StochK", stochK },
                        { "StochD", stochD },
                        { "MFI", mfi }
                    },
                    SupportingCorrelations = supportingCorrelations,
                    TimeHorizon = "Short-term",
                    Reliability = strength * 0.8 // Slightly lower reliability for mean reversion patterns
                };

                return pattern;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error identifying overextended pattern for {symbol}", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Identifies price-volume confirmation patterns
        /// </summary>
        private IndicatorConfirmationPattern IdentifyVolumePricePattern(
            string symbol, Dictionary<string, double> indicators, List<IndicatorCorrelationResult> correlations)
        {
            try
            {
                if (indicators == null || !indicators.ContainsKey("OBV") ||
                    !indicators.ContainsKey("MFI") || !indicators.ContainsKey("MomentumScore"))
                {
                    return null;
                }

                double obv = indicators["OBV"];
                double mfi = indicators["MFI"];
                double momentum = indicators["MomentumScore"];

                // Get historical data for OBV to determine trend
                string range = MapTimeframeToRange("1day");
                string interval = MapTimeframeToInterval("1day");
                var historicalData = _historicalDataService.GetHistoricalPrices(symbol, range, interval).Result;

                if (historicalData.Count < 10)
                    return null;

                // Analyze OBV trend
                var obvHistory = CalculateHistoricalOBV(historicalData);
                if (obvHistory.Count < 5)
                    return null;

                double obvTrend = obvHistory[obvHistory.Count - 1] - obvHistory[obvHistory.Count - 5];

                string direction;
                double strength;

                // Check for bullish volume confirmation
                if (obvTrend > 0 && mfi > 50 && momentum > 0)
                {
                    direction = "Bullish";
                    strength = obvTrend / 1000000.0 * (mfi - 50) / 50.0 * momentum / 100.0;
                    strength = Math.Min(1.0, Math.Max(0.1, strength));
                }
                // Check for bearish volume confirmation
                else if (obvTrend < 0 && mfi < 50 && momentum < 0)
                {
                    direction = "Bearish";
                    strength = Math.Abs(obvTrend) / 1000000.0 * (50 - mfi) / 50.0 * Math.Abs(momentum) / 100.0;
                    strength = Math.Min(1.0, Math.Max(0.1, strength));
                }
                else
                {
                    return null; // No clear volume-price confirmation
                }

                // Find supporting correlations
                var supportingCorrelations = correlations.Where(c =>
                    c.FirstIndicator == "OBV" && c.SecondIndicator == "MFI")
                    .ToList();

                // Create pattern
                var pattern = new IndicatorConfirmationPattern
                {
                    Symbol = symbol,
                    PatternType = "Volume-Price Confirmation",
                    SignalDirection = direction,
                    ConfirmationStrength = strength,
                    ConfirmingIndicators = new List<string> { "OBV", "MFI", "Momentum" },
                    IndicatorValues = new Dictionary<string, double>
                    {
                        { "OBV", obv },
                        { "OBV_Trend", obvTrend },
                        { "MFI", mfi },
                        { "Momentum", momentum }
                    },
                    SupportingCorrelations = supportingCorrelations,
                    TimeHorizon = "Medium-term",
                    Reliability = 0.7 * strength
                };

                return pattern;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error identifying volume-price pattern for {symbol}", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Calculate Pearson correlation coefficient between two indicator value series
        /// </summary>
        private double CalculatePearsonCorrelation(List<double> x, List<double> y)
        {
            int n = Math.Min(x.Count, y.Count);

            if (n <= 1)
                return 0;

            // Trim lists to same length
            var x1 = x.Take(n).ToList();
            var y1 = y.Take(n).ToList();

            double sumX = 0;
            double sumY = 0;
            double sumXY = 0;
            double sumX2 = 0;
            double sumY2 = 0;

            for (int i = 0; i < n; i++)
            {
                sumX += x1[i];
                sumY += y1[i];
                sumXY += x1[i] * y1[i];
                sumX2 += x1[i] * x1[i];
                sumY2 += y1[i] * y1[i];
            }

            // Calculate correlation coefficient
            double numerator = n * sumXY - sumX * sumY;
            double denominator = Math.Sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

            if (denominator == 0)
                return 0;

            return numerator / denominator;
        }

        /// <summary>
        /// Calculate confidence level for a correlation based on sample size
        /// </summary>
        private double CalculateConfidenceLevel(double correlation, int sampleSize)
        {
            if (sampleSize <= 5)
                return 0.3; // Very low confidence with small samples

            double r2 = correlation * correlation;
            double absoluteCorr = Math.Abs(correlation);

            // Adjust confidence based on sample size and correlation strength
            if (sampleSize > 30)
                return Math.Min(1.0, 0.7 + absoluteCorr * 0.3);
            else if (sampleSize > 20)
                return Math.Min(0.9, 0.6 + absoluteCorr * 0.3);
            else if (sampleSize > 10)
                return Math.Min(0.8, 0.5 + absoluteCorr * 0.3);
            else
                return Math.Min(0.6, 0.4 + absoluteCorr * 0.2);
        }

        /// <summary>
        /// Calculate Stochastic RSI values from RSI values
        /// </summary>
        private List<double> CalculateStochRSIFromRSI(List<double> rsiValues)
        {
            var stochRsi = new List<double>();

            if (rsiValues.Count < 14)
                return stochRsi;

            for (int i = 13; i < rsiValues.Count; i++)
            {
                // Get the last 14 RSI values
                var window = rsiValues.Skip(i - 13).Take(14).ToList();

                double minRsi = window.Min();
                double maxRsi = window.Max();
                double currentRsi = window.Last();

                if (maxRsi - minRsi == 0)
                    stochRsi.Add(0.5); // Avoid division by zero
                else
                    stochRsi.Add((currentRsi - minRsi) / (maxRsi - minRsi));
            }

            return stochRsi;
        }

        /// <summary>
        /// Calculate historical VWAP values
        /// </summary>
        private List<double> CalculateHistoricalVWAP(List<HistoricalPrice> prices)
        {
            var vwapValues = new List<double>();

            if (prices.Count == 0)
                return vwapValues;

            double cumulativeTPV = 0;
            long cumulativeVolume = 0;

            foreach (var bar in prices)
            {
                double typicalPrice = (bar.High + bar.Low + bar.Close) / 3;
                cumulativeTPV += typicalPrice * bar.Volume;
                cumulativeVolume += bar.Volume;

                if (cumulativeVolume == 0)
                    vwapValues.Add(bar.Close);
                else
                    vwapValues.Add(cumulativeTPV / cumulativeVolume);
            }

            return vwapValues;
        }

        /// <summary>
        /// Calculate historical OBV values
        /// </summary>
        private List<double> CalculateHistoricalOBV(List<HistoricalPrice> prices)
        {
            var obvValues = new List<double>();

            if (prices.Count < 2)
                return obvValues;

            // First OBV value is arbitrary, set it to 0
            obvValues.Add(0);
            double obv = 0;

            for (int i = 1; i < prices.Count; i++)
            {
                var currentClose = prices[i].Close;
                var previousClose = prices[i - 1].Close;
                var currentVolume = prices[i].Volume;

                if (currentClose > previousClose)
                    obv += currentVolume;
                else if (currentClose < previousClose)
                    obv -= currentVolume;
                // Price unchanged - OBV remains the same

                obvValues.Add(obv);
            }

            return obvValues;
        }

        /// <summary>
        /// Calculate historical MFI values
        /// </summary>
        private List<double> CalculateHistoricalMFI(List<HistoricalPrice> prices)
        {
            var mfiValues = new List<double>();

            if (prices.Count < 14)
                return mfiValues;

            // For the first 13 bars, we don't have enough history for MFI
            for (int i = 0; i < 13; i++)
            {
                mfiValues.Add(50); // Neutral default
            }

            // Calculate MFI for each bar with sufficient history
            for (int i = 13; i < prices.Count; i++)
            {
                double positiveMoneyFlow = 0;
                double negativeMoneyFlow = 0;

                for (int j = i - 13; j <= i; j++)
                {
                    if (j == i - 13) continue; // Skip first bar in window

                    // Calculate typical price
                    double currentTypical = (prices[j].High + prices[j].Low + prices[j].Close) / 3;
                    double prevTypical = (prices[j - 1].High + prices[j - 1].Low + prices[j - 1].Close) / 3;
                    double rawMoneyFlow = currentTypical * prices[j].Volume;

                    if (currentTypical > prevTypical)
                        positiveMoneyFlow += rawMoneyFlow;
                    else if (currentTypical < prevTypical)
                        negativeMoneyFlow += rawMoneyFlow;
                }

                // Calculate money flow ratio
                double moneyFlowRatio = negativeMoneyFlow == 0 ? 100 : positiveMoneyFlow / negativeMoneyFlow;

                // Calculate MFI
                double mfi = 100 - 100 / (1 + moneyFlowRatio);
                mfiValues.Add(mfi);
            }

            return mfiValues;
        }

        #endregion

        #region Custom Indicators Implementation

        /// <summary>
        /// Load saved custom indicators from the repository
        /// </summary>
        private async Task LoadSavedIndicatorsAsync()
        {
            try
            {
                var indicatorDefinitions = await _customIndicatorRepository.GetAllIndicatorsAsync();

                foreach (var definition in indicatorDefinitions)
                {
                    try
                    {
                        // Create and register indicator from definition
                        var indicator = CustomIndicatorFactory.CreateIndicator(definition);
                        if (indicator != null)
                        {
                            _customIndicators[definition.Id] = indicator;
                        }
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Error", $"Failed to load indicator {definition.Id}", ex.ToString());
                    }
                }

                //DatabaseMonolith.Log("Info", $"Loaded {_customIndicators.Count} custom indicators", "");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to load custom indicators", ex.ToString());
            }
        }

        /// <summary>
        /// Get an indicator by its ID
        /// </summary>
        public async Task<IIndicator> GetIndicatorAsync(string indicatorId)
        {
            if (string.IsNullOrWhiteSpace(indicatorId))
                throw new ArgumentNullException(nameof(indicatorId));

            // Check if the indicator is already loaded
            if (_customIndicators.TryGetValue(indicatorId, out var indicator))
                return indicator;

            // Try to load it from the repository
            var definition = await _customIndicatorRepository.GetIndicatorAsync(indicatorId);
            if (definition == null)
                return null;

            try
            {
                // Create indicator from definition
                indicator = CustomIndicatorFactory.CreateIndicator(definition);
                if (indicator != null)
                {
                    _customIndicators[indicatorId] = indicator;
                    return indicator;
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to create indicator {indicatorId}", ex.ToString());
            }

            return null;
        }

        /// <summary>
        /// Register a new indicator
        /// </summary>
        public async Task<bool> RegisterIndicatorAsync(IIndicator indicator)
        {
            if (indicator == null)
                throw new ArgumentNullException(nameof(indicator));

            if (string.IsNullOrWhiteSpace(indicator.Id))
                throw new ArgumentException("Indicator ID cannot be null or empty", nameof(indicator));

            // Check if the indicator is already registered
            if (_customIndicators.ContainsKey(indicator.Id))
                return false;

            // Add to in-memory dictionary
            _customIndicators[indicator.Id] = indicator;

            // If it's a CustomIndicator, save it to the repository
            if (indicator is CustomIndicator customIndicator)
            {
                // Convert to definition and save
                var definition = new CustomIndicatorDefinition
                {
                    Id = indicator.Id,
                    Name = indicator.Name,
                    Description = indicator.Description,
                    Category = indicator.Category,
                    Dependencies = indicator.GetDependencies().ToList(),
                    IndicatorType = indicator is CompositeIndicator ? "Composite" : "Custom"
                };

                // Save parameters
                foreach (var param in indicator.Parameters)
                {
                    definition.Parameters[param.Key] = new IndicatorParameterDefinition
                    {
                        Name = param.Value.Name,
                        Description = param.Value.Description,
                        DefaultValue = param.Value.DefaultValue,
                        Value = param.Value.Value,
                        ParameterType = param.Value.ParameterType.Name,
                        MinValue = param.Value.MinValue,
                        MaxValue = param.Value.MaxValue,
                        IsOptional = param.Value.IsOptional,
                        Options = param.Value.Options
                    };
                }

                // Save to repository
                await _customIndicatorRepository.SaveIndicatorAsync(definition);
            }

            return true;
        }

        /// <summary>
        /// Unregister an indicator
        /// </summary>
        public async Task<bool> UnregisterIndicatorAsync(string indicatorId)
        {
            if (string.IsNullOrWhiteSpace(indicatorId))
                throw new ArgumentNullException(nameof(indicatorId));

            // Remove from in-memory dictionary
            if (!_customIndicators.Remove(indicatorId))
                return false;

            // Remove from repository
            _customIndicatorRepository.DeleteIndicator(indicatorId);

            return true;
        }

        /// <summary>
        /// Get all registered indicators
        /// </summary>
        public async Task<List<IIndicator>> GetAllIndicatorsAsync()
        {
            return _customIndicators.Values.ToList();
        }

        /// <summary>
        /// Calculate a custom indicator for a symbol
        /// </summary>
        public async Task<Dictionary<string, double>> CalculateCustomIndicatorAsync(string indicatorId, string symbol, string timeframe)
        {
            var indicator = await GetIndicatorAsync(indicatorId);
            if (indicator == null)
                throw new InvalidOperationException($"Indicator {indicatorId} not found");

            // Get historical data for calculation
            List<HistoricalPrice> historicalData;

            try
            {
                historicalData = await _historicalDataService.GetHistoricalPrices(symbol, timeframe);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to get historical data for {symbol}", ex.ToString());
                throw new InvalidOperationException("Failed to get historical data", ex);
            }

            // Calculate the indicator
            return await indicator.CalculateAsync(historicalData);
        }

        /// <summary>
        /// Get an indicator definition by ID
        /// </summary>
        public async Task<CustomIndicatorDefinition> GetIndicatorDefinitionAsync(string indicatorId)
        {
            if (string.IsNullOrWhiteSpace(indicatorId))
                throw new ArgumentNullException(nameof(indicatorId));

            return await _customIndicatorRepository.GetIndicatorAsync(indicatorId);
        }

        /// <summary>
        /// Save an indicator definition
        /// </summary>
        public async Task<bool> SaveIndicatorDefinitionAsync(CustomIndicatorDefinition definition)
        {
            if (definition == null)
                throw new ArgumentNullException(nameof(definition));

            // Save to repository
            var result = await _customIndicatorRepository.SaveIndicatorAsync(definition);

            // If successful and indicator is already loaded, update it
            if (result && _customIndicators.ContainsKey(definition.Id))
            {
                try
                {
                    // Create and update indicator from definition
                    var indicator = CustomIndicatorFactory.CreateIndicator(definition);
                    if (indicator != null)
                    {
                        _customIndicators[definition.Id] = indicator;
                    }
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", $"Failed to update indicator {definition.Id}", ex.ToString());
                    // Don't fail the operation, the definition was saved successfully
                }
            }

            return result;
        }

        /// <summary>
        /// Search for indicators
        /// </summary>
        public async Task<List<CustomIndicatorDefinition>> SearchIndicatorsAsync(string searchTerm, string category = null)
        {
            return await _customIndicatorRepository.SearchIndicatorsAsync(searchTerm, category);
        }

        #endregion

        /// <summary>
        /// Gets the correlation between two indicators
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="timeframe">Timeframe for analysis</param>
        /// <param name="indicator1">First indicator name</param>
        /// <param name="indicator2">Second indicator name</param>
        /// <param name="period">Number of periods to analyze</param>
        /// <returns>Correlation coefficient (-1 to 1)</returns>
        public async Task<double> GetIndicatorCorrelation(string symbol, string timeframe, string indicator1, string indicator2, int period = 30)
        {
            // This method previously referenced Quantra.Modules.IndicatorCorrelationAnalysis, which does not exist.
            // You should implement the correlation logic directly or use existing methods.
            // For now, fallback to CalculateIndicatorCorrelation and return the coefficient.
            var result = await CalculateIndicatorCorrelation(symbol, indicator1, indicator2, timeframe, period);
            return result.CorrelationCoefficient;
        }

        #region Visualization Framework Methods

        // Methods for visualization framework
        public (List<double> Upper, List<double> Middle, List<double> Lower) CalculateBollingerBands(List<double> prices, int period, double stdDevMultiplier)
        {
            var result = (Upper: new List<double>(), Middle: new List<double>(), Lower: new List<double>());

            // Calculate Simple Moving Average (SMA)
            var sma = CalculateSMA(prices, period);
            result.Middle = sma;

            // Calculate standard deviation for each window
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period - 1)
                {
                    // Not enough data for full window
                    result.Upper.Add(double.NaN);
                    result.Lower.Add(double.NaN);
                    continue;
                }

                // Get window of prices for calculating std dev
                var window = prices.Skip(i - period + 1).Take(period).ToList();
                var mean = sma[i];
                var stdDev = Math.Sqrt(window.Average(v => Math.Pow(v - mean, 2)));

                // Calculate upper and lower bands
                result.Upper.Add(mean + stdDevMultiplier * stdDev);
                result.Lower.Add(mean - stdDevMultiplier * stdDev);
            }

            return result;
        }

        public List<double> CalculateSMA(List<double> prices, int period)
        {
            var result = new List<double>();

            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period - 1)
                {
                    // Not enough data for full window
                    result.Add(double.NaN);
                    continue;
                }

                // Calculate SMA for this window
                var sum = 0.0;
                for (int j = i - period + 1; j <= i; j++)
                {
                    sum += prices[j];
                }
                result.Add(sum / period);
            }

            return result;
        }

        public List<double> CalculateEMA(List<double> prices, int period)
        {
            var result = new List<double>();

            // First EMA value is SMA
            var sma = CalculateSMA(prices, period);

            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period - 1)
                {
                    // Not enough data for full window
                    result.Add(double.NaN);
                    continue;
                }

                if (i == period - 1)
                {
                    // First EMA is SMA
                    result.Add(sma[i]);
                    continue;
                }

                // Calculate EMA: EMA = Price * k + EMA(previous) * (1-k)
                // where k = 2/(period+1)
                double multiplier = 2.0 / (period + 1);
                double ema = prices[i] * multiplier + result[i - 1] * (1 - multiplier);
                result.Add(ema);
            }

            return result;
        }

        public List<double> CalculateVWAP(List<double> highPrices, List<double> lowPrices, List<double> closePrices, List<double> volumes)
        {
            var result = new List<double>();

            // Ensure all input lists are the same length
            int length = Math.Min(Math.Min(highPrices.Count, lowPrices.Count), Math.Min(closePrices.Count, volumes.Count));

            double cumulativeVolume = 0;
            double cumulativePV = 0; // Price * Volume

            for (int i = 0; i < length; i++)
            {
                // Typical price = (high + low + close) / 3
                double typicalPrice = (highPrices[i] + lowPrices[i] + closePrices[i]) / 3;

                // Cumulative values
                cumulativeVolume += volumes[i];
                cumulativePV += typicalPrice * volumes[i];

                // VWAP = Cumulative PV / Cumulative Volume
                double vwap = cumulativePV / cumulativeVolume;
                result.Add(vwap);
            }

            return result;
        }

        public List<double> CalculateRSI(List<double> prices, int period)
        {
            var result = new List<double>();

            // Need at least period+1 data points to calculate first RSI
            if (prices.Count <= period)
            {
                // Not enough data
                for (int i = 0; i < prices.Count; i++)
                {
                    result.Add(double.NaN);
                }
                return result;
            }

            // Calculate price changes
            var priceChanges = new List<double>();
            for (int i = 1; i < prices.Count; i++)
            {
                priceChanges.Add(prices[i] - prices[i - 1]);
            }

            // First gains and losses (for period)
            var gains = new List<double>();
            var losses = new List<double>();

            for (int i = 0; i < priceChanges.Count; i++)
            {
                double change = priceChanges[i];
                gains.Add(change > 0 ? change : 0);
                losses.Add(change < 0 ? Math.Abs(change) : 0);

                // Add NaN for initial values
                if (i < period - 1)
                {
                    result.Add(double.NaN);
                }
                else
                {
                    // Calculate average gains and average losses for this period
                    double avgGain = gains.Skip(i - period + 1).Take(period).Average();
                    double avgLoss = losses.Skip(i - period + 1).Take(period).Average();

                    // Calculate RS and RSI
                    double rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                    double rsi = 100 - 100 / (1 + rs);
                    result.Add(rsi);
                }
            }

            return result;
        }

        public (List<double> MacdLine, List<double> SignalLine, List<double> Histogram) CalculateMACD(List<double> prices, int fastPeriod, int slowPeriod, int signalPeriod)
        {
            var result = (MacdLine: new List<double>(), SignalLine: new List<double>(), Histogram: new List<double>());

            // Calculate EMAs
            var fastEMA = CalculateEMA(prices, fastPeriod);
            var slowEMA = CalculateEMA(prices, slowPeriod);

            // Calculate MACD line
            var macdLine = new List<double>();
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < slowPeriod - 1)
                {
                    // Not enough data for slow EMA
                    macdLine.Add(double.NaN);
                }
                else
                {
                    // MACD = Fast EMA - Slow EMA
                    macdLine.Add(fastEMA[i] - slowEMA[i]);
                }
            }

            // Calculate signal line (EMA of MACD line)
            var signalLine = CalculateEMA(macdLine, signalPeriod);

            // Calculate histogram (MACD - Signal)
            var histogram = new List<double>();
            for (int i = 0; i < macdLine.Count; i++)
            {
                if (i < slowPeriod + signalPeriod - 2)
                {
                    // Not enough data for signal line
                    histogram.Add(double.NaN);
                }
                else
                {
                    // Histogram = MACD - Signal
                    histogram.Add(macdLine[i] - signalLine[i]);
                }
            }

            result.MacdLine = macdLine;
            result.SignalLine = signalLine;
            result.Histogram = histogram;

            return result;
        }

        public (List<double> K, List<double> D) CalculateStochastic(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int kPeriod, int kSmoothing, int dPeriod)
        {
            var result = (K: new List<double>(), D: new List<double>());

            // Ensure all input lists are the same length
            int length = Math.Min(Math.Min(highPrices.Count, lowPrices.Count), closePrices.Count);

            // Calculate %K for each point
            var rawK = new List<double>();
            for (int i = 0; i < length; i++)
            {
                if (i < kPeriod - 1)
                {
                    // Not enough data for full window
                    rawK.Add(double.NaN);
                    continue;
                }

                // Find highest high and lowest low over period
                var highestHigh = double.MinValue;
                var lowestLow = double.MaxValue;

                for (int j = i - kPeriod + 1; j <= i; j++)
                {
                    highestHigh = Math.Max(highestHigh, highPrices[j]);
                    lowestLow = Math.Min(lowestLow, lowPrices[j]);
                }

                // Calculate raw %K
                double currentClose = closePrices[i];
                double stochK = highestHigh == lowestLow ? 50 : (currentClose - lowestLow) / (highestHigh - lowestLow) * 100;
                rawK.Add(stochK);
            }

            // Calculate smoothed %K (optional smoothing)
            var smoothedK = kSmoothing > 1 ? CalculateSMA(rawK, kSmoothing) : rawK;
            result.K = smoothedK;

            // Calculate %D (SMA of %K)
            result.D = CalculateSMA(smoothedK, dPeriod);

            return result;
        }

        public List<double> CalculateStochRSI(List<double> prices, int rsiPeriod, int stochPeriod, int kPeriod, int dPeriod)
        {
            var result = new List<double>();

            // Calculate RSI values
            var rsiValues = CalculateRSI(prices, rsiPeriod);

            // Apply Stochastic to RSI values
            for (int i = 0; i < rsiValues.Count; i++)
            {
                if (i < rsiPeriod + stochPeriod - 1)
                {
                    // Not enough data for full window
                    result.Add(double.NaN);
                    continue;
                }

                // Find highest high and lowest low RSI over period
                var highestRsi = double.MinValue;
                var lowestRsi = double.MaxValue;

                for (int j = i - stochPeriod + 1; j <= i; j++)
                {
                    if (!double.IsNaN(rsiValues[j]))
                    {
                        highestRsi = Math.Max(highestRsi, rsiValues[j]);
                        lowestRsi = Math.Min(lowestRsi, rsiValues[j]);
                    }
                }

                // Calculate StochRSI
                double currentRsi = rsiValues[i];
                double stochRsi = highestRsi == lowestRsi ? 0.5 : (currentRsi - lowestRsi) / (highestRsi - lowestRsi);
                result.Add(stochRsi);
            }

            return result;
        }

        public List<double> CalculateWilliamsR(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period)
        {
            var result = new List<double>();

            // Ensure all input lists are the same length
            int length = Math.Min(Math.Min(highPrices.Count, lowPrices.Count), closePrices.Count);

            for (int i = 0; i < length; i++)
            {
                if (i < period - 1)
                {
                    // Not enough data for full window
                    result.Add(double.NaN);
                    continue;
                }

                // Find highest high and lowest low over period
                var highestHigh = double.MinValue;
                var lowestLow = double.MaxValue;

                for (int j = i - period + 1; j <= i; j++)
                {
                    highestHigh = Math.Max(highestHigh, highPrices[j]);
                    lowestLow = Math.Min(lowestLow, lowPrices[j]);
                }

                // Calculate Williams %R
                double currentClose = closePrices[i];
                double williamsR = highestHigh == lowestLow ? -50 : (highestHigh - currentClose) / (highestHigh - lowestLow) * -100;
                result.Add(williamsR);
            }

            return result;
        }

        public List<double> CalculateCCI(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period)
        {
            var result = new List<double>();

            // Ensure all input lists are the same length
            int length = Math.Min(Math.Min(highPrices.Count, lowPrices.Count), closePrices.Count);

            // Calculate typical prices: (H+L+C)/3
            var typicalPrices = new List<double>();
            for (int i = 0; i < length; i++)
            {
                typicalPrices.Add((highPrices[i] + lowPrices[i] + closePrices[i]) / 3);
            }

            // Calculate CCI
            for (int i = 0; i < length; i++)
            {
                if (i < period - 1)
                {
                    // Not enough data for full window
                    result.Add(double.NaN);
                    continue;
                }

                // Calculate SMA of typical prices
                var sma = 0.0;
                for (int j = i - period + 1; j <= i; j++)
                {
                    sma += typicalPrices[j];
                }
                sma /= period;

                // Calculate mean deviation
                var meanDev = 0.0;
                for (int j = i - period + 1; j <= i; j++)
                {
                    meanDev += Math.Abs(typicalPrices[j] - sma);
                }
                meanDev /= period;

                // Calculate CCI
                double cci = meanDev == 0 ? 0 : (typicalPrices[i] - sma) / (0.015 * meanDev);
                result.Add(cci);
            }

            return result;
        }

        public List<double> CalculateADX(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period = 14)
        {
            var result = new List<double>();

            // Ensure all input lists are the same length
            int length = Math.Min(Math.Min(highPrices.Count, lowPrices.Count), closePrices.Count);

            if (length < period + 1)
            {
                for (int i = 0; i < length; i++)
                {
                    result.Add(double.NaN);
                }
                return result;
            }

            // Calculate True Range (TR) and Directional Movement (+DM, -DM)
            var trueRanges = new List<double>();
            var plusDMs = new List<double>();
            var minusDMs = new List<double>();

            for (int i = 1; i < length; i++)
            {
                // True Range
                double tr1 = highPrices[i] - lowPrices[i];
                double tr2 = Math.Abs(highPrices[i] - closePrices[i - 1]);
                double tr3 = Math.Abs(lowPrices[i] - closePrices[i - 1]);
                double tr = Math.Max(tr1, Math.Max(tr2, tr3));
                trueRanges.Add(tr);

                // Directional Movement
                double highDiff = highPrices[i] - highPrices[i - 1];
                double lowDiff = lowPrices[i - 1] - lowPrices[i];

                double plusDM = highDiff > lowDiff && highDiff > 0 ? highDiff : 0;
                double minusDM = lowDiff > highDiff && lowDiff > 0 ? lowDiff : 0;

                plusDMs.Add(plusDM);
                minusDMs.Add(minusDM);
            }

            // Calculate smoothed versions (using EMA)
            var smoothedTRs = CalculateEMA(trueRanges, period);
            var smoothedPlusDMs = CalculateEMA(plusDMs, period);
            var smoothedMinusDMs = CalculateEMA(minusDMs, period);

            // Calculate +DI and -DI
            var plusDIs = new List<double>();
            var minusDIs = new List<double>();

            for (int i = 0; i < smoothedTRs.Count; i++)
            {
                double plusDI = smoothedTRs[i] == 0 ? 0 : smoothedPlusDMs[i] / smoothedTRs[i] * 100;
                double minusDI = smoothedTRs[i] == 0 ? 0 : smoothedMinusDMs[i] / smoothedTRs[i] * 100;

                plusDIs.Add(plusDI);
                minusDIs.Add(minusDI);
            }

            // Calculate DX
            var dxValues = new List<double>();
            for (int i = 0; i < plusDIs.Count; i++)
            {
                double diSum = plusDIs[i] + minusDIs[i];
                double dx = diSum == 0 ? 0 : Math.Abs(plusDIs[i] - minusDIs[i]) / diSum * 100;
                dxValues.Add(dx);
            }

            // Calculate ADX (EMA of DX)
            var adxValues = CalculateEMA(dxValues, period);

            // Pad with NaN for initial periods
            for (int i = 0; i < period; i++)
            {
                result.Add(double.NaN);
            }

            result.AddRange(adxValues);

            return result;
        }

        public List<double> CalculateATR(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period = 14)
        {
            var result = new List<double>();

            // Ensure all input lists are the same length
            int length = Math.Min(Math.Min(highPrices.Count, lowPrices.Count), closePrices.Count);

            if (length < 2)
            {
                for (int i = 0; i < length; i++)
                {
                    result.Add(double.NaN);
                }
                return result;
            }

            // Calculate True Range for each period
            var trueRanges = new List<double>();

            // First period - just high-low
            result.Add(highPrices[0] - lowPrices[0]);

            for (int i = 1; i < length; i++)
            {
                double tr1 = highPrices[i] - lowPrices[i];
                double tr2 = Math.Abs(highPrices[i] - closePrices[i - 1]);
                double tr3 = Math.Abs(lowPrices[i] - closePrices[i - 1]);
                double tr = Math.Max(tr1, Math.Max(tr2, tr3));
                trueRanges.Add(tr);
            }

            // Calculate ATR using EMA of True Range
            if (trueRanges.Count >= period)
            {
                var atrValues = CalculateEMA(trueRanges, period);
                result.AddRange(atrValues);
            }
            else
            {
                // Not enough data for full ATR calculation
                for (int i = 1; i < length; i++)
                {
                    result.Add(double.NaN);
                }
            }

            return result;
        }

        public List<double> CalculateROC(List<double> prices, int period = 10)
        {
            var result = new List<double>();

            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period)
                {
                    // Not enough data for ROC calculation
                    result.Add(double.NaN);
                }
                else
                {
                    // ROC = ((current - previous) / previous) * 100
                    double current = prices[i];
                    double previous = prices[i - period];

                    if (previous == 0)
                    {
                        result.Add(0);
                    }
                    else
                    {
                        double roc = (current - previous) / previous * 100;
                        result.Add(roc);
                    }
                }
            }

            return result;
        }

        public List<double> CalculateUltimateOscillator(List<double> highPrices, List<double> lowPrices, List<double> closePrices,
            int period1 = 7, int period2 = 14, int period3 = 28)
        {
            var result = new List<double>();

            // Ensure all input lists are the same length
            int length = Math.Min(Math.Min(highPrices.Count, lowPrices.Count), closePrices.Count);

            if (length < period3 + 1)
            {
                for (int i = 0; i < length; i++)
                {
                    result.Add(double.NaN);
                }
                return result;
            }

            // Calculate Buying Pressure (BP) and True Range (TR)
            var buyingPressures = new List<double>();
            var trueRanges = new List<double>();

            for (int i = 1; i < length; i++)
            {
                // Buying Pressure = Close - Min(Low, Previous Close)
                double minLow = Math.Min(lowPrices[i], closePrices[i - 1]);
                double bp = closePrices[i] - minLow;
                buyingPressures.Add(bp);

                // True Range
                double tr1 = highPrices[i] - lowPrices[i];
                double tr2 = Math.Abs(highPrices[i] - closePrices[i - 1]);
                double tr3 = Math.Abs(lowPrices[i] - closePrices[i - 1]);
                double tr = Math.Max(tr1, Math.Max(tr2, tr3));
                trueRanges.Add(tr);
            }

            // Calculate Ultimate Oscillator
            for (int i = period3 - 1; i < buyingPressures.Count; i++)
            {
                // Sum BP and TR for each period
                double bp1Sum = 0, tr1Sum = 0;
                double bp2Sum = 0, tr2Sum = 0;
                double bp3Sum = 0, tr3Sum = 0;

                // Period 1
                for (int j = i - period1 + 1; j <= i; j++)
                {
                    bp1Sum += buyingPressures[j];
                    tr1Sum += trueRanges[j];
                }

                // Period 2
                for (int j = i - period2 + 1; j <= i; j++)
                {
                    bp2Sum += buyingPressures[j];
                    tr2Sum += trueRanges[j];
                }

                // Period 3
                for (int j = i - period3 + 1; j <= i; j++)
                {
                    bp3Sum += buyingPressures[j];
                    tr3Sum += trueRanges[j];
                }

                // Calculate averages
                double avg1 = tr1Sum == 0 ? 0 : bp1Sum / tr1Sum;
                double avg2 = tr2Sum == 0 ? 0 : bp2Sum / tr2Sum;
                double avg3 = tr3Sum == 0 ? 0 : bp3Sum / tr3Sum;

                // Ultimate Oscillator formula
                double uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7;
                result.Add(uo);
            }

            // Pad with NaN for initial periods
            var paddedResult = new List<double>();
            for (int i = 0; i < period3; i++)
            {
                paddedResult.Add(double.NaN);
            }
            paddedResult.AddRange(result);

            return paddedResult;
        }

        #endregion

        /// <summary>
        /// Disposes the service and releases resources
        /// </summary>
        public void Dispose()
        {
            _taskThrottler?.Dispose();
        }

        // Minimal local implementations to avoid cross-project dependencies
        private class ConcurrentTaskThrottler : IDisposable
        {
            private readonly SemaphoreSlim _semaphore;
            public ConcurrentTaskThrottler(int maxDegreeOfParallelism = 6)
            {
                if (maxDegreeOfParallelism <= 0) maxDegreeOfParallelism = 1;
                _semaphore = new SemaphoreSlim(maxDegreeOfParallelism, maxDegreeOfParallelism);
            }
            public async Task<T[]> ExecuteThrottledAsync<T>(IEnumerable<Func<Task<T>>> taskFactories, CancellationToken cancellationToken = default)
            {
                if (taskFactories == null) return Array.Empty<T>();
                var tasks = taskFactories.Select(factory => Run(factory, cancellationToken));
                return await Task.WhenAll(tasks);
            }
            private async Task<T> Run<T>(Func<Task<T>> factory, CancellationToken ct)
            {
                await _semaphore.WaitAsync(ct);
                try { return await factory(); }
                finally { _semaphore.Release(); }
            }
            public void Dispose()
            {
                _semaphore?.Dispose();
            }
        }

        private class CustomIndicatorRepository
        {
            private readonly Dictionary<string, CustomIndicatorDefinition> _defs = new Dictionary<string, CustomIndicatorDefinition>();

            public Task<CustomIndicatorDefinition> GetIndicatorAsync(string id)
            {
                if (string.IsNullOrWhiteSpace(id)) throw new ArgumentNullException(nameof(id));
                _defs.TryGetValue(id, out var def);
                return Task.FromResult(def);
            }

            public Task<bool> SaveIndicatorAsync(CustomIndicatorDefinition definition)
            {
                if (definition == null) throw new ArgumentNullException(nameof(definition));
                if (string.IsNullOrWhiteSpace(definition.Id)) definition.Id = Guid.NewGuid().ToString();
                if (definition.Parameters == null) definition.Parameters = new Dictionary<string, IndicatorParameterDefinition>();
                _defs[definition.Id] = definition;
                return Task.FromResult(true);
            }

            public bool DeleteIndicator(string id)
            {
                if (string.IsNullOrWhiteSpace(id)) return false;
                return _defs.Remove(id);
            }

            public Task<List<CustomIndicatorDefinition>> GetAllIndicatorsAsync()
            {
                return Task.FromResult(_defs.Values.ToList());
            }

            public Task<List<CustomIndicatorDefinition>> SearchIndicatorsAsync(string searchTerm, string category = null)
            {
                var q = _defs.Values.AsEnumerable();
                if (!string.IsNullOrWhiteSpace(searchTerm))
                {
                    q = q.Where(i => (i.Name?.IndexOf(searchTerm, StringComparison.OrdinalIgnoreCase) ?? -1) >= 0 ||
                                     (i.Description?.IndexOf(searchTerm, StringComparison.OrdinalIgnoreCase) ?? -1) >= 0);
                }
                if (!string.IsNullOrWhiteSpace(category))
                {
                    q = q.Where(i => string.Equals(i.Category, category, StringComparison.OrdinalIgnoreCase));
                }
                return Task.FromResult(q.ToList());
            }
        }
    }
}
