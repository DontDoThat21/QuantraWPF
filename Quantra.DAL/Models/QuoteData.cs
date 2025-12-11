using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;
using LiveCharts; // For ChartValues<T>
using LiveCharts.Defaults; // For OhlcPoint
using Quantra.Utilities; // UiThreadHelper

namespace Quantra
{
    // ...existing code...

    // Ensure this class exists and is public
    public class QuoteData : IDisposable
    {
        private bool _disposed = false;
        // Basic stock info
        public string Symbol { get; set; }
        public string Name { get; set; } // Company name

        // Price and change info
        public double Price { get; set; }
        public double Change { get; set; }
        public double ChangePercent { get; set; }

        // Market info
        public double DayHigh { get; set; }
        public double DayLow { get; set; }
        public double MarketCap { get; set; }
        public double Volume { get; set; }
        public string Sector { get; set; }

        // Technical indicators for grid display
        public double RSI { get; set; }
        public double PERatio { get; set; }
        public double VWAP { get; set; }

        // Date/time info
        public DateTime Date { get; set; }
        public DateTime LastUpdated { get; set; }
        public DateTime LastAccessed { get; set; }
        public DateTime Timestamp { get; set; }
        public DateTime? CacheTime { get; set; } // From database cache metadata

        // Prediction properties
        public double? PredictedPrice { get; set; }
        public string PredictedAction { get; set; } // BUY, SELL, HOLD
        public double? PredictionConfidence { get; set; } // 0.0 to 1.0
        public DateTime? PredictionTimestamp { get; set; }
        public string PredictionModelVersion { get; set; }

        // Option chain data
        public List<OptionData> OptionChain { get; set; } = new List<OptionData>();
        public DateTime? OptionDataFetchTimestamp { get; set; }
        public TimeSpan? OptionDataCacheWindow { get; set; }

        // Chart/band/indicator values
        public ChartValues<double> StockPriceValues { get; set; } = new ChartValues<double>();
        public ChartValues<double> UpperBandValues { get; set; } = new ChartValues<double>();
        public ChartValues<double> MiddleBandValues { get; set; } = new ChartValues<double>();
        public ChartValues<double> LowerBandValues { get; set; } = new ChartValues<double>();
        public ChartValues<double> RSIValues { get; set; } = new ChartValues<double>();

        // Candle pattern data (for CandleSeries)
        public ChartValues<OhlcPoint> PatternCandles { get; set; } = new ChartValues<OhlcPoint>();

        // Add other properties as needed for your UI/data binding

        // Add this method to help populate chart values in correct order
        public async Task PopulateChartValuesFromHistorical(List<HistoricalData> historical)
        {
            if (historical == null || historical.Count == 0)
                return;

            if (_disposed) return;

            // Perform calculations on background thread
            var (priceValues, candleValues) = await Task.Run(() =>
            {
                var prices = new List<double>();
                var candles = new List<OhlcPoint>();

                foreach (var data in historical)
                {
                    prices.Add(data.Close);
                    candles.Add(new OhlcPoint(data.Open, data.High, data.Low, data.Close));
                }

                return (prices, candles);
            });

            if (_disposed) return;

            // Update UI collections on UI thread using batched approach
            if (UiThreadHelper.HasDispatcher)
            {
                await UiThreadHelper.InvokeAsync(() =>
                {
                    if (_disposed) return;

                    // Clear existing data
                    StockPriceValues?.Clear();
                    UpperBandValues?.Clear();
                    MiddleBandValues?.Clear();
                    LowerBandValues?.Clear();
                    RSIValues?.Clear();
                    PatternCandles?.Clear();

                    // Add all values in batch to reduce UI notifications
                    if (StockPriceValues != null)
                        StockPriceValues.AddRange(priceValues);

                    if (PatternCandles != null)
                        PatternCandles.AddRange(candleValues);

                });
            }
            else
            {
                // Fallback for testing scenarios where no Application.Current exists
                if (_disposed) return;

                StockPriceValues?.Clear();
                UpperBandValues?.Clear();
                MiddleBandValues?.Clear();
                LowerBandValues?.Clear();
                RSIValues?.Clear();
                PatternCandles?.Clear();

                if (StockPriceValues != null)
                {
                    foreach (var price in priceValues)
                        StockPriceValues.Add(price);
                }

                if (PatternCandles != null)
                {
                    foreach (var candle in candleValues)
                        PatternCandles.Add(candle);
                }
            }
        }

        // Add this method if not already present
        public async Task PopulateChartValuesFromHistorical(List<HistoricalPrice> historical)
        {
            if (historical == null || historical.Count == 0)
                return;

            if (_disposed) return;

            // Perform all calculations on background thread
            var calculationResults = await Task.Run(() =>
            {
                var priceValues = new List<double>();
                var candleValues = new List<OhlcPoint>();
                var upperBandValues = new List<double>();
                var middleBandValues = new List<double>();
                var lowerBandValues = new List<double>();
                var rsiValues = new List<double>();

                // Populate price and candle data
                foreach (var h in historical)
                {
                    priceValues.Add(h.Close);
                    candleValues.Add(new OhlcPoint(h.Open, h.High, h.Low, h.Close));
                }

                // Calculate Bollinger Bands and RSI
                int period = Math.Min(20, historical.Count);
                var prices = historical.Select(h => h.Close).ToList();
                if (prices.Count >= period)
                {
                    var (upperBand, middleBand, lowerBand) = CalculateBollingerBands(prices, period, 2.0);
                    upperBandValues.AddRange(upperBand);
                    middleBandValues.AddRange(middleBand);
                    lowerBandValues.AddRange(lowerBand);

                    var rsi = CalculateRSI(prices, 14);
                    rsiValues.AddRange(rsi);
                }

                return new
                {
                    PriceValues = priceValues,
                    CandleValues = candleValues,
                    UpperBandValues = upperBandValues,
                    MiddleBandValues = middleBandValues,
                    LowerBandValues = lowerBandValues,
                    RSIValues = rsiValues
                };
            });

            if (_disposed) return;

            // Update UI collections on UI thread using batched updates
            if (UiThreadHelper.HasDispatcher)
            {
                await UiThreadHelper.InvokeAsync(() =>
                {
                    if (_disposed) return;

                    // Clear all collections first
                    StockPriceValues?.Clear();
                    UpperBandValues?.Clear();
                    MiddleBandValues?.Clear();
                    LowerBandValues?.Clear();
                    RSIValues?.Clear();
                    PatternCandles?.Clear();

                    // Add all values in batches to reduce UI notifications
                    if (StockPriceValues != null && calculationResults.PriceValues.Any())
                        StockPriceValues.AddRange(calculationResults.PriceValues);

                    if (PatternCandles != null && calculationResults.CandleValues.Any())
                        PatternCandles.AddRange(calculationResults.CandleValues);

                    if (UpperBandValues != null && calculationResults.UpperBandValues.Any())
                        UpperBandValues.AddRange(calculationResults.UpperBandValues);

                    if (MiddleBandValues != null && calculationResults.MiddleBandValues.Any())
                        MiddleBandValues.AddRange(calculationResults.MiddleBandValues);

                    if (LowerBandValues != null && calculationResults.LowerBandValues.Any())
                        LowerBandValues.AddRange(calculationResults.LowerBandValues);

                    if (RSIValues != null && calculationResults.RSIValues.Any())
                        RSIValues.AddRange(calculationResults.RSIValues);

                });
            }
            else
            {
                // Fallback for testing scenarios where no Application.Current exists
                if (_disposed) return;

                StockPriceValues?.Clear();
                UpperBandValues?.Clear();
                MiddleBandValues?.Clear();
                LowerBandValues?.Clear();
                RSIValues?.Clear();
                PatternCandles?.Clear();

                if (StockPriceValues != null)
                {
                    foreach (var price in calculationResults.PriceValues)
                        StockPriceValues.Add(price);
                }

                if (PatternCandles != null)
                {
                    foreach (var candle in calculationResults.CandleValues)
                        PatternCandles.Add(candle);
                }

                if (UpperBandValues != null)
                {
                    foreach (var value in calculationResults.UpperBandValues)
                        UpperBandValues.Add(value);
                }

                if (MiddleBandValues != null)
                {
                    foreach (var value in calculationResults.MiddleBandValues)
                        MiddleBandValues.Add(value);
                }

                if (LowerBandValues != null)
                {
                    foreach (var value in calculationResults.LowerBandValues)
                        LowerBandValues.Add(value);
                }

                if (RSIValues != null)
                {
                    foreach (var value in calculationResults.RSIValues)
                        RSIValues.Add(value);
                }
            }
        }

        // Helper for Bollinger Bands
        private (List<double> upperBand, List<double> middleBand, List<double> lowerBand) CalculateBollingerBands(List<double> prices, int period, double stdDevMultiplier)
        {
            var middleBand = new List<double>();
            var upperBand = new List<double>();
            var lowerBand = new List<double>();
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

        // Helper for RSI
        private List<double> CalculateRSI(List<double> prices, int period)
        {
            var rsiValues = new List<double>();
            for (int i = 0; i < period; i++) rsiValues.Add(double.NaN);
            if (prices.Count <= period) return rsiValues;
            List<double> gains = new List<double>();
            List<double> losses = new List<double>();
            for (int i = 1; i < prices.Count; i++)
            {
                double change = prices[i] - prices[i - 1];
                gains.Add(change > 0 ? change : 0);
                losses.Add(change < 0 ? -change : 0);
            }
            double avgGain = gains.Take(period).Average();
            double avgLoss = losses.Take(period).Average();
            double rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
            double rsi = 100 - (100 / (1 + rs));
            rsiValues.Add(rsi);
            for (int i = period + 1; i < prices.Count; i++)
            {
                avgGain = ((avgGain * (period - 1)) + gains[i - 1]) / period;
                avgLoss = ((avgLoss * (period - 1)) + losses[i - 1]) / period;
                rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                rsi = 100 - (100 / (1 + rs));
                rsiValues.Add(rsi);
            }
            return rsiValues;
        }

        /// <summary>
        /// Gets option contracts for a specific strike price and expiration date
        /// </summary>
        /// <param name="strikePrice">The strike price to filter by</param>
        /// <param name="expirationDate">The expiration date to filter by</param>
        /// <returns>List of option contracts matching the criteria</returns>
        public List<OptionData> GetOptionsByStrikeAndExpiration(double strikePrice, DateTime expirationDate)
        {
            if (OptionChain == null || OptionChain.Count == 0)
                return new List<OptionData>();

            return OptionChain.Where(o =>
                Math.Abs(o.StrikePrice - strikePrice) < 0.01 &&
                o.ExpirationDate.Date == expirationDate.Date)
                .ToList();
        }

        /// <summary>
        /// Gets all available strike prices for a given expiration date
        /// </summary>
        /// <param name="expirationDate">The expiration date to filter by</param>
        /// <returns>List of available strike prices</returns>
        public List<double> GetAvailableStrikes(DateTime expirationDate)
        {
            if (OptionChain == null || OptionChain.Count == 0)
                return new List<double>();

            return OptionChain.Where(o => o.ExpirationDate.Date == expirationDate.Date)
                             .Select(o => o.StrikePrice)
                             .Distinct()
                             .OrderBy(s => s)
                             .ToList();
        }

        /// <summary>
        /// Gets all available expiration dates in the option chain
        /// </summary>
        /// <returns>List of available expiration dates</returns>
        public List<DateTime> GetAvailableExpirations()
        {
            if (OptionChain == null || OptionChain.Count == 0)
                return new List<DateTime>();

            return OptionChain.Select(o => o.ExpirationDate)
                             .Distinct()
                             .OrderBy(d => d)
                             .ToList();
        }

        /// <summary>
        /// Checks if option data is within the cache window (fresh enough to use)
        /// </summary>
        /// <param name="currentTime">Current time to compare against</param>
        /// <returns>True if option data is fresh, false if expired or no cache window set</returns>
        public bool IsOptionDataFresh(DateTime currentTime)
        {
            if (!OptionDataFetchTimestamp.HasValue || !OptionDataCacheWindow.HasValue)
                return false;

            var dataAge = currentTime - OptionDataFetchTimestamp.Value;
            return dataAge <= OptionDataCacheWindow.Value;
        }

        /// <summary>
        /// Clears all chart data collections to free memory
        /// </summary>
        public void ClearChartData()
        {
            if (_disposed) return;

            StockPriceValues?.Clear();
            UpperBandValues?.Clear();
            MiddleBandValues?.Clear();
            LowerBandValues?.Clear();
            RSIValues?.Clear();
            PatternCandles?.Clear();
            OptionChain?.Clear();
        }

        /// <summary>
        /// Dispose pattern implementation for proper memory cleanup
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected dispose method
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                // Clear chart data collections
                ClearChartData();

                // Clear collections to allow reuse and prevent NullReferenceExceptions
                StockPriceValues?.Clear();
                UpperBandValues?.Clear();
                MiddleBandValues?.Clear();
                LowerBandValues?.Clear();
                RSIValues?.Clear();
                PatternCandles?.Clear();
                OptionChain?.Clear();
            }

            _disposed = true;
        }
    }

    // Class to hold historical chart data
    public class HistoricalDataResponse
    {
        public string Symbol { get; set; }
        public List<HistoricalData> historical { get; set; }
    }

    public class HistoricalData
    {
        public DateTime Date { get; set; }
        public double Open { get; set; }
        public double High { get; set; }
        public double Low { get; set; }
        public double Close { get; set; }
        public long Volume { get; set; }
    }
}
