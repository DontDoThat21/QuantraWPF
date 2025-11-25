using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Utilities;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a strategy based on Simple Moving Average crossovers
    /// </summary>
    public class SmaCrossoverStrategy : TradingStrategyProfile
    {
        private int _fastPeriod = 20;
        private int _slowPeriod = 50;
        private bool _useVolumeFilter = true;
        private double _volumeThreshold = 1.5;

        public SmaCrossoverStrategy()
        {
            Name = "SMA Crossover";
            Description = "Generates signals based on crossovers of fast and slow moving averages. " +
                          "Buy signal when fast MA crosses above slow MA, sell signal when fast MA crosses below slow MA. " +
                          "Can filter signals based on volume confirmation to reduce false signals.";
            RiskLevel = 0.5;
            MinConfidence = 0.6;
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

        public override IEnumerable<string> RequiredIndicators => new[] { "SMA20", "SMA50", "Volume" };

        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || prices.Count < SlowPeriod + 1)
                return null;

            int currentIndex = index ?? prices.Count - 1;
            if (currentIndex < SlowPeriod || currentIndex >= prices.Count)
                return null;

            // Calculate SMA values
            List<double> fastSMA = MovingAverageUtils.CalculateSMA(prices, FastPeriod);
            List<double> slowSMA = MovingAverageUtils.CalculateSMA(prices, SlowPeriod);

            if (fastSMA == null || slowSMA == null ||
                currentIndex - SlowPeriod + 1 >= fastSMA.Count ||
                currentIndex - SlowPeriod + 1 >= slowSMA.Count)
                return null;

            // Current position for SMAs
            int smaIndex = currentIndex - SlowPeriod + 1;

            // Check for crossover
            bool currentFastAboveSlow = fastSMA[smaIndex] > slowSMA[smaIndex];
            bool previousFastAboveSlow = false;

            // Ensure we have a previous data point to compare
            if (smaIndex > 0)
            {
                previousFastAboveSlow = fastSMA[smaIndex - 1] > slowSMA[smaIndex - 1];

                // Volume filter (if enabled)
                bool volumeConfirms = true;
                if (UseVolumeFilter)
                {
                    int volumeLookback = 20;
                    int startVolumeIndex = Math.Max(0, currentIndex - volumeLookback);
                    double avgVolume = prices.Skip(startVolumeIndex).Take(Math.Min(volumeLookback, currentIndex - startVolumeIndex)).Average(p => p.Volume);
                    volumeConfirms = prices[currentIndex].Volume >= avgVolume * VolumeThreshold;
                }

                // Buy signal: Fast SMA crosses above slow SMA
                if (currentFastAboveSlow && !previousFastAboveSlow && volumeConfirms)
                {
                    return "BUY";
                }

                // Sell signal: Fast SMA crosses below slow SMA
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

            // Check if we have all required SMA indicators
            if (!indicators.TryGetValue("SMA20", out double fastSma) ||
                !indicators.TryGetValue("SMA50", out double slowSma))
            {
                // Try alternative indicators
                if (!indicators.TryGetValue("SMA", out fastSma) ||
                    !indicators.TryGetValue("SMA_SLOW", out slowSma))
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
                    return Math.Abs(fastSma - slowSma) / slowSma > 0.01; // At least 1% difference
                }

                bool volumeConfirms = volume >= volumeAvg * VolumeThreshold;
                if (!volumeConfirms)
                    return false;
            }

            // Get trend direction
            bool uptrend = fastSma > slowSma;
            double crossoverStrength = Math.Abs(fastSma - slowSma) / slowSma;

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