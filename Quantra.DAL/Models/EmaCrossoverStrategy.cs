using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Utilities;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a strategy based on Exponential Moving Average crossovers
    /// </summary>
    public class EmaCrossoverStrategy : TradingStrategyProfile
    {
        private int _fastPeriod = 12;
        private int _slowPeriod = 26;
        private bool _useVolumeFilter = true;
        private double _volumeThreshold = 1.5;

        public EmaCrossoverStrategy()
        {
            Name = "EMA Crossover";
            Description = "Generates signals based on crossovers of fast and slow exponential moving averages. " +
                          "Buy signal when fast EMA crosses above slow EMA, sell signal when fast EMA crosses below slow EMA. " +
                          "EMA is more responsive to recent price changes than SMA, making it more suitable for trending markets.";
            RiskLevel = 0.6; // Slightly higher than SMA due to faster response to price changes
            MinConfidence = 0.65;
        }

        /// <summary>
        /// Period for the faster moving average
        /// </summary>
        public int FastPeriod
        {
            get => _fastPeriod;
            set
            {
                if (value > 0 && value < _slowPeriod && _fastPeriod != value)
                {
                    _fastPeriod = value;
                    OnPropertyChanged(nameof(FastPeriod));
                }
            }
        }

        /// <summary>
        /// Period for the slower moving average
        /// </summary>
        public int SlowPeriod
        {
            get => _slowPeriod;
            set
            {
                if (value > _fastPeriod && _slowPeriod != value)
                {
                    _slowPeriod = value;
                    OnPropertyChanged(nameof(SlowPeriod));
                }
            }
        }

        /// <summary>
        /// Whether to filter signals based on volume
        /// </summary>
        public bool UseVolumeFilter
        {
            get => _useVolumeFilter;
            set
            {
                if (_useVolumeFilter != value)
                {
                    _useVolumeFilter = value;
                    OnPropertyChanged(nameof(UseVolumeFilter));
                }
            }
        }

        /// <summary>
        /// Volume should be this multiple of the average volume
        /// </summary>
        public double VolumeThreshold
        {
            get => _volumeThreshold;
            set
            {
                if (value > 0 && _volumeThreshold != value)
                {
                    _volumeThreshold = value;
                    OnPropertyChanged(nameof(VolumeThreshold));
                }
            }
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "EMA12", "EMA26", "Volume" };

        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || prices.Count < SlowPeriod + 1)
                return null;

            int currentIndex = index ?? prices.Count - 1;
            if (currentIndex < SlowPeriod || currentIndex >= prices.Count)
                return null;

            // Calculate EMA values
            List<double> fastEMA = MovingAverageUtils.CalculateEMA(prices, FastPeriod);
            List<double> slowEMA = MovingAverageUtils.CalculateEMA(prices, SlowPeriod);

            if (fastEMA == null || slowEMA == null ||
                currentIndex < FastPeriod ||
                currentIndex < SlowPeriod)
                return null;

            // Get current and previous values for comparison
            double currentFastEma = fastEMA[currentIndex - FastPeriod + 1];
            double currentSlowEma = slowEMA[currentIndex - SlowPeriod + 1];

            // Need at least 2 periods to detect a crossover
            if (currentIndex <= FastPeriod || currentIndex <= SlowPeriod)
                return null;

            double previousFastEma = fastEMA[currentIndex - FastPeriod];
            double previousSlowEma = slowEMA[currentIndex - SlowPeriod];

            // Check for crossover
            bool currentFastAboveSlow = currentFastEma > currentSlowEma;
            bool previousFastAboveSlow = previousFastEma > previousSlowEma;

            // Volume filter (if enabled)
            bool volumeConfirms = true;
            if (UseVolumeFilter)
            {
                int volumeLookback = 20;
                int startVolumeIndex = Math.Max(0, currentIndex - volumeLookback);
                double avgVolume = prices.Skip(startVolumeIndex).Take(Math.Min(volumeLookback, currentIndex - startVolumeIndex)).Average(p => p.Volume);
                volumeConfirms = prices[currentIndex].Volume >= avgVolume * VolumeThreshold;
            }

            // Buy signal: Fast EMA crosses above slow EMA
            if (currentFastAboveSlow && !previousFastAboveSlow && volumeConfirms)
            {
                return "BUY";
            }

            // Sell signal: Fast EMA crosses below slow EMA
            if (!currentFastAboveSlow && previousFastAboveSlow && volumeConfirms)
            {
                return "SELL";
            }

            return null;
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            if (indicators == null)
                return false;

            // Check if we have all required EMA indicators
            if (!indicators.TryGetValue("EMA12", out double fastEma) ||
                !indicators.TryGetValue("EMA26", out double slowEma))
            {
                // Try alternative indicators
                if (!indicators.TryGetValue("EMA_FAST", out fastEma) ||
                    !indicators.TryGetValue("EMA_SLOW", out slowEma))
                {
                    return false;
                }
            }

            // Check volume condition if required
            if (UseVolumeFilter)
            {
                if (!indicators.TryGetValue("Volume", out double volume) ||
                    !indicators.TryGetValue("VolumeAvg", out double volumeAvg))
                {
                    // If we can't validate volume, we'll still continue but with lower confidence
                    return Math.Abs(fastEma - slowEma) / slowEma > 0.01; // At least 1% difference
                }

                bool volumeConfirms = volume >= volumeAvg * VolumeThreshold;
                if (!volumeConfirms)
                    return false;
            }

            // Get trend direction
            bool uptrend = fastEma > slowEma;
            double crossoverStrength = Math.Abs(fastEma - slowEma) / slowEma;

            // Check if we have trend confirmation from other indicators
            bool trendConfirmed = true;

            if (indicators.TryGetValue("ADX", out double adx))
            {
                // ADX > 25 indicates strong trend
                trendConfirmed = trendConfirmed && adx > 25;
            }

            if (indicators.TryGetValue("RSI", out double rsi))
            {
                // RSI should align with trend direction
                trendConfirmed = trendConfirmed && ((uptrend && rsi > 50) || (!uptrend && rsi < 50));
            }

            return trendConfirmed && crossoverStrength > 0.005; // At least 0.5% difference
        }


    }
}