using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a mean reversion strategy using Bollinger Bands
    /// </summary>
    public class BollingerBandsMeanReversionStrategy : TradingStrategyProfile
    {
        private int _period = 20;
        private double _stdDevMultiplier = 2.0;
        private double _oversoldPercentage = 95.0;
        private double _overboughtPercentage = 5.0;
        private bool _requireRsiConfirmation = true;

        public BollingerBandsMeanReversionStrategy()
        {
            Name = "Bollinger Bands";
            Description = "Generates signals when price moves outside of Bollinger Bands and then reverts. " +
                          "Buy signal when price moves below lower band and then back inside. " +
                          "Sell signal when price moves above upper band and then back inside. " +
                          "Can be combined with RSI confirmation for better accuracy.";
            RiskLevel = 0.7;
            MinConfidence = 0.6;
        }

        /// <summary>
        /// Period for the Bollinger Bands calculation (typically 20)
        /// </summary>
        public int Period
        {
            get => _period;
            set
            {
                if (value > 0 && _period != value)
                {
                    _period = value;
                    OnPropertyChanged(nameof(Period));
                }
            }
        }

        /// <summary>
        /// Standard deviation multiplier for band width (typically 2.0)
        /// </summary>
        public double StdDevMultiplier
        {
            get => _stdDevMultiplier;
            set
            {
                if (value > 0 && _stdDevMultiplier != value)
                {
                    _stdDevMultiplier = value;
                    OnPropertyChanged(nameof(StdDevMultiplier));
                }
            }
        }

        /// <summary>
        /// Percentage for oversold condition (typically 95%)
        /// </summary>
        public double OversoldPercentage
        {
            get => _oversoldPercentage;
            set
            {
                if (value >= 50 && value <= 100 && _oversoldPercentage != value)
                {
                    _oversoldPercentage = value;
                    OnPropertyChanged(nameof(OversoldPercentage));
                }
            }
        }

        /// <summary>
        /// Percentage for overbought condition (typically 5%)
        /// </summary>
        public double OverboughtPercentage
        {
            get => _overboughtPercentage;
            set
            {
                if (value >= 0 && value <= 50 && _overboughtPercentage != value)
                {
                    _overboughtPercentage = value;
                    OnPropertyChanged(nameof(OverboughtPercentage));
                }
            }
        }

        /// <summary>
        /// Whether to require RSI confirmation for signals
        /// </summary>
        public bool RequireRsiConfirmation
        {
            get => _requireRsiConfirmation;
            set
            {
                if (_requireRsiConfirmation != value)
                {
                    _requireRsiConfirmation = value;
                    OnPropertyChanged(nameof(RequireRsiConfirmation));
                }
            }
        }

        public override IEnumerable<string> RequiredIndicators => new[]
        {
            "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width", "%B", "RSI"
        };

        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || prices.Count < Period + 1)
                return null;

            int currentIndex = index ?? prices.Count - 1;
            if (currentIndex < Period || currentIndex >= prices.Count)
                return null;

            // Calculate Bollinger Bands
            var bands = CalculateBollingerBands(prices, Period, StdDevMultiplier);
            // Fix the null check for the tuple
            if (bands.upperBand == null || bands.middleBand == null || bands.lowerBand == null || bands.percentB == null)
                return null;

            var upperBand = bands.upperBand;
            var middleBand = bands.middleBand;
            var lowerBand = bands.lowerBand;
            var percentBs = bands.percentB;

            // Calculate %B for current and previous candle
            int bbIndex = currentIndex - Period + 1;
            if (bbIndex < 0 || bbIndex >= percentBs.Count || bbIndex - 1 < 0)
                return null;

            double currentPercentB = percentBs[bbIndex];
            double previousPercentB = percentBs[bbIndex - 1];

            // Calculate RSI for confirmation if required
            List<double> rsi = null;
            if (RequireRsiConfirmation)
            {
                rsi = CalculateRSI(prices, 14);
                if (rsi == null || currentIndex - 14 >= rsi.Count)
                    return null;
            }

            // Check for oversold mean reversion (buy signal)
            if (previousPercentB <= 0 && currentPercentB > 0)
            {
                // If RSI confirmation required, check RSI < 30 (oversold)
                if (!RequireRsiConfirmation || (rsi != null && rsi[currentIndex - 14] < 30))
                {
                    // Check volume if available (prefer higher volume for confirmation)
                    bool volumeConfirms = true;
                    if (prices[0].Volume > 0) // Volume data is available
                    {
                        int lookback = Math.Min(10, currentIndex);
                        double avgVolume = prices.Skip(currentIndex - lookback).Take(lookback).Average(p => p.Volume);
                        volumeConfirms = prices[currentIndex].Volume > avgVolume;
                    }

                    if (volumeConfirms)
                    {
                        return "BUY";
                    }
                }
            }

            // Check for overbought mean reversion (sell signal)
            if (previousPercentB >= 1 && currentPercentB < 1)
            {
                // If RSI confirmation required, check RSI > 70 (overbought)
                if (!RequireRsiConfirmation || (rsi != null && rsi[currentIndex - 14] > 70))
                {
                    // Check volume if available (prefer higher volume for confirmation)
                    bool volumeConfirms = true;
                    if (prices[0].Volume > 0) // Volume data is available
                    {
                        int lookback = Math.Min(10, currentIndex);
                        double avgVolume = prices.Skip(currentIndex - lookback).Take(lookback).Average(p => p.Volume);
                        volumeConfirms = prices[currentIndex].Volume > avgVolume;
                    }

                    if (volumeConfirms)
                    {
                        return "SELL";
                    }
                }
            }

            return null;
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            if (indicators == null)
                return false;

            // Check if we have the required Bollinger Band indicators
            if (!indicators.TryGetValue("BB_Upper", out double upper) ||
                !indicators.TryGetValue("BB_Lower", out double lower) ||
                !indicators.TryGetValue("BB_Middle", out double middle))
            {
                return false;
            }

            // Check if price is available
            if (!indicators.TryGetValue("Price", out double price) &&
                !indicators.TryGetValue("Close", out price))
            {
                return false;
            }

            // Calculate %B if not available
            double percentB;
            if (!indicators.TryGetValue("%B", out percentB))
            {
                percentB = (price - lower) / (upper - lower);
            }

            // Check if RSI confirmation is required and available
            if (RequireRsiConfirmation)
            {
                if (!indicators.TryGetValue("RSI", out double rsi))
                {
                    return false;
                }

                // Check for potentially valid conditions based on %B and RSI
                bool oversold = percentB < 0.05 && rsi < 30;
                bool overbought = percentB > 0.95 && rsi > 70;
                return oversold || overbought;
            }
            else
            {
                // Check for potentially valid conditions based on %B only
                bool oversold = percentB < 0.05;
                bool overbought = percentB > 0.95;
                return oversold || overbought;
            }
        }

        /// <summary>
        /// Calculate Bollinger Bands and %B for a price series
        /// </summary>
        private (List<double> upperBand, List<double> middleBand, List<double> lowerBand, List<double> percentB)
            CalculateBollingerBands(List<HistoricalPrice> prices, int period, double stdDevMultiplier)
        {
            if (prices.Count < period)
                return (null, null, null, null);

            var upperBand = new List<double>();
            var middleBand = new List<double>();
            var lowerBand = new List<double>();
            var percentB = new List<double>();

            // Add NaN for periods where Bollinger Bands can't be calculated
            for (int i = 0; i < period - 1; i++)
            {
                upperBand.Add(double.NaN);
                middleBand.Add(double.NaN);
                lowerBand.Add(double.NaN);
                percentB.Add(double.NaN);
            }

            // Calculate Bollinger Bands for each subsequent period
            for (int i = period - 1; i < prices.Count; i++)
            {
                double sum = 0;
                double sumOfSquares = 0;

                for (int j = i - period + 1; j <= i; j++)
                {
                    sum += prices[j].Close;
                    sumOfSquares += prices[j].Close * prices[j].Close;
                }

                double mean = sum / period;
                double variance = sumOfSquares / period - mean * mean;
                double stdDev = Math.Sqrt(variance);

                double upper = mean + stdDevMultiplier * stdDev;
                double lower = mean - stdDevMultiplier * stdDev;

                upperBand.Add(upper);
                middleBand.Add(mean);
                lowerBand.Add(lower);

                // Calculate %B
                double pctB = (prices[i].Close - lower) / (upper - lower);
                percentB.Add(pctB);
            }

            return (upperBand, middleBand, lowerBand, percentB);
        }

        /// <summary>
        /// Calculate RSI for a price series
        /// </summary>
        private List<double> CalculateRSI(List<HistoricalPrice> prices, int period)
        {
            if (prices.Count <= period)
                return null;

            var rsiValues = new List<double>();

            // Add empty values for initial periods
            for (int i = 0; i < period; i++)
                rsiValues.Add(double.NaN);

            var gains = new List<double>();
            var losses = new List<double>();

            // Calculate initial gains and losses
            for (int i = 1; i <= period; i++)
            {
                var change = prices[i].Close - prices[i - 1].Close;
                gains.Add(Math.Max(0, change));
                losses.Add(Math.Max(0, -change));
            }

            // Calculate initial average gain and loss
            var avgGain = gains.Average();
            var avgLoss = losses.Average();

            // Calculate first RSI
            var rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
            var rsi = 100 - (100 / (1 + rs));
            rsiValues.Add(rsi);

            // Calculate remaining RSI values
            for (int i = period + 1; i < prices.Count; i++)
            {
                var change = prices[i].Close - prices[i - 1].Close;
                var gain = Math.Max(0, change);
                var loss = Math.Max(0, -change);

                avgGain = ((avgGain * (period - 1)) + gain) / period;
                avgLoss = ((avgLoss * (period - 1)) + loss) / period;

                rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                rsi = 100 - (100 / (1 + rs));
                rsiValues.Add(rsi);
            }

            return rsiValues;
        }
    }
}