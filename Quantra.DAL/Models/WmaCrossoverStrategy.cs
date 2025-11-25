using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Utilities;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a strategy based on Weighted Moving Average crossovers
    /// </summary>
    public class WmaCrossoverStrategy : TradingStrategyProfile
    {
        private int _fastPeriod = 10;
        private int _slowPeriod = 30;
        private bool _useVolumeFilter = true;
        private double _volumeThreshold = 1.5;

        public WmaCrossoverStrategy()
        {
            Name = "WMA Crossover";
            Description = "Generates signals based on crossovers of fast and slow weighted moving averages. " +
                          "Buy signal when fast WMA crosses above slow WMA, sell signal when fast WMA crosses below slow WMA. " +
                          "WMA assigns more weight to recent prices, making it more responsive to recent changes while reducing noise.";
            RiskLevel = 0.55; // Between SMA and EMA in terms of responsiveness
            MinConfidence = 0.65;
        }

        /// <summary>
        /// Period for the faster weighted moving average
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
        /// Period for the slower weighted moving average
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

        public override IEnumerable<string> RequiredIndicators => new[] { "WMA10", "WMA30", "Volume" };

        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || prices.Count < SlowPeriod + 1)
                return null;

            int currentIndex = index ?? prices.Count - 1;
            if (currentIndex < SlowPeriod || currentIndex >= prices.Count)
                return null;

            // Calculate WMA values
            List<double> fastWMA = MovingAverageUtils.CalculateWMA(prices, FastPeriod);
            List<double> slowWMA = MovingAverageUtils.CalculateWMA(prices, SlowPeriod);

            if (fastWMA == null || slowWMA == null ||
                currentIndex < FastPeriod ||
                currentIndex < SlowPeriod)
                return null;

            // Current position for WMAs
            int fastWmaIndex = currentIndex - (FastPeriod - 1);
            int slowWmaIndex = currentIndex - (SlowPeriod - 1);

            if (fastWmaIndex < 0 || fastWmaIndex >= fastWMA.Count ||
                slowWmaIndex < 0 || slowWmaIndex >= slowWMA.Count)
                return null;

            // Check for crossover
            bool currentFastAboveSlow = fastWMA[fastWmaIndex] > slowWMA[slowWmaIndex];
            bool previousFastAboveSlow = false;

            // Ensure we have a previous data point to compare
            if (fastWmaIndex > 0 && slowWmaIndex > 0)
            {
                previousFastAboveSlow = fastWMA[fastWmaIndex - 1] > slowWMA[slowWmaIndex - 1];

                // Volume filter (if enabled)
                bool volumeConfirms = true;
                if (UseVolumeFilter)
                {
                    int volumeLookback = 20;
                    int startVolumeIndex = Math.Max(0, currentIndex - volumeLookback);
                    double avgVolume = prices.Skip(startVolumeIndex).Take(Math.Min(volumeLookback, currentIndex - startVolumeIndex)).Average(p => p.Volume);
                    volumeConfirms = prices[currentIndex].Volume >= avgVolume * VolumeThreshold;
                }

                // Buy signal: Fast WMA crosses above slow WMA
                if (currentFastAboveSlow && !previousFastAboveSlow && volumeConfirms)
                {
                    return "BUY";
                }

                // Sell signal: Fast WMA crosses below slow WMA
                if (!currentFastAboveSlow && previousFastAboveSlow && volumeConfirms)
                {
                    return "SELL";
                }
            }

            return null;
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            if (indicators == null)
                return false;

            // Check if we have all required WMA indicators
            if (!indicators.TryGetValue("WMA10", out double fastWma) ||
                !indicators.TryGetValue("WMA30", out double slowWma))
            {
                // Try alternative indicators
                if (!indicators.TryGetValue("WMA_FAST", out fastWma) ||
                    !indicators.TryGetValue("WMA_SLOW", out slowWma))
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
                    return Math.Abs(fastWma - slowWma) / slowWma > 0.01; // At least 1% difference
                }

                bool volumeConfirms = volume >= volumeAvg * VolumeThreshold;
                if (!volumeConfirms)
                    return false;
            }

            // Get trend direction
            bool uptrend = fastWma > slowWma;
            double crossoverStrength = Math.Abs(fastWma - slowWma) / slowWma;

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