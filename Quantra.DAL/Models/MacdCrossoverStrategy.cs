using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a strategy based on MACD crossovers and divergences
    /// </summary>
    public class MacdCrossoverStrategy : TradingStrategyProfile
    {
        private int _fastEma = 12;
        private int _slowEma = 26;
        private int _signalPeriod = 9;
        private bool _includeDivergence = true;
        private bool _useConfirmationCandle = true;

        public MacdCrossoverStrategy()
        {
            Name = "MACD Crossover";
            Description = "Generates signals based on MACD line crossing above/below signal line. " +
                          "Buy signal when MACD crosses above signal line, sell signal when MACD crosses below signal line. " +
                          "Can also identify bullish/bearish divergences between price and MACD for stronger signals.";
            RiskLevel = 0.6;
            MinConfidence = 0.65;
        }

        /// <summary>
        /// Fast EMA period for MACD calculation
        /// </summary>
        public int FastEma
        {
            get => _fastEma;
            set
            {
                if (value > 0 && value < _slowEma && _fastEma != value)
                {
                    _fastEma = value;
                    OnPropertyChanged(nameof(FastEma));
                }
            }
        }

        /// <summary>
        /// Slow EMA period for MACD calculation
        /// </summary>
        public int SlowEma
        {
            get => _slowEma;
            set
            {
                if (value > _fastEma && _slowEma != value)
                {
                    _slowEma = value;
                    OnPropertyChanged(nameof(SlowEma));
                }
            }
        }

        /// <summary>
        /// Signal line period (EMA of MACD)
        /// </summary>
        public int SignalPeriod
        {
            get => _signalPeriod;
            set
            {
                if (value > 0 && _signalPeriod != value)
                {
                    _signalPeriod = value;
                    OnPropertyChanged(nameof(SignalPeriod));
                }
            }
        }

        /// <summary>
        /// Whether to check for divergence between price and MACD
        /// </summary>
        public bool IncludeDivergence
        {
            get => _includeDivergence;
            set
            {
                if (_includeDivergence != value)
                {
                    _includeDivergence = value;
                    OnPropertyChanged(nameof(IncludeDivergence));
                }
            }
        }

        /// <summary>
        /// Whether to require a confirmation candle after crossover
        /// </summary>
        public bool UseConfirmationCandle
        {
            get => _useConfirmationCandle;
            set
            {
                if (_useConfirmationCandle != value)
                {
                    _useConfirmationCandle = value;
                    OnPropertyChanged(nameof(UseConfirmationCandle));
                }
            }
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "MACD", "Signal", "Histogram" };

        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || prices.Count < SlowEma + SignalPeriod + 1)
                return null;

            int currentIndex = index ?? prices.Count - 1;
            if (currentIndex < SlowEma + SignalPeriod || currentIndex >= prices.Count)
                return null;

            // Calculate MACD values
            var result = CalculateMACD(prices, FastEma, SlowEma, SignalPeriod);
            if (result.macdLine == null || result.signalLine == null || result.histogram == null)
                return null;
                
            var macdLine = result.macdLine;
            var signalLine = result.signalLine;
            var histogram = result.histogram;

            // MACD index is adjusted for calculation periods
            int macdIndex = currentIndex - FastEma - SlowEma - SignalPeriod + 3;
            if (macdIndex < 0 || macdIndex >= macdLine.Count || macdIndex - 1 < 0)
                return null;

            // Get current and previous MACD values
            double currentMacd = macdLine[macdIndex];
            double currentSignal = signalLine[macdIndex];
            double previousMacd = macdLine[macdIndex - 1];
            double previousSignal = signalLine[macdIndex - 1];

            // Check for crossover
            bool bullishCrossover = previousMacd < previousSignal && currentMacd > currentSignal;
            bool bearishCrossover = previousMacd > previousSignal && currentMacd < currentSignal;

            // Check for confirmation candle if required
            bool confirmed = true;
            if (UseConfirmationCandle)
            {
                // For bullish crossover: close above open and above previous close
                if (bullishCrossover)
                {
                    confirmed = prices[currentIndex].Close > prices[currentIndex].Open &&
                                prices[currentIndex].Close > prices[currentIndex - 1].Close;
                }
                // For bearish crossover: close below open and below previous close
                else if (bearishCrossover)
                {
                    confirmed = prices[currentIndex].Close < prices[currentIndex].Open &&
                                prices[currentIndex].Close < prices[currentIndex - 1].Close;
                }
            }

            // Check for divergence if enabled
            bool hasDivergence = false;
            if (IncludeDivergence)
            {
                int lookbackPeriods = 14;
                int startIdx = Math.Max(0, macdIndex - lookbackPeriods);

                // For bullish crossover: Check for bullish divergence (price making lower lows, MACD making higher lows)
                if (bullishCrossover)
                {
                    hasDivergence = CheckBullishDivergence(prices, macdLine, currentIndex, macdIndex, startIdx);
                }
                // For bearish crossover: Check for bearish divergence (price making higher highs, MACD making lower highs)
                else if (bearishCrossover)
                {
                    hasDivergence = CheckBearishDivergence(prices, macdLine, currentIndex, macdIndex, startIdx);
                }
            }

            // Generate signals
            if (bullishCrossover && confirmed)
            {
                // Higher confidence if divergence is present
                return "BUY";
            }
            else if (bearishCrossover && confirmed)
            {
                // Higher confidence if divergence is present
                return "SELL";
            }

            return null;
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            return indicators.ContainsKey("MACD") && indicators.ContainsKey("Signal") && indicators.ContainsKey("Histogram");
        }

        /// <summary>
        /// Check for bullish divergence: price makes lower lows, but MACD makes higher lows
        /// </summary>
        private bool CheckBullishDivergence(List<HistoricalPrice> prices, List<double> macdLine, int priceIndex, int macdIndex, int startIdx)
        {
            if (startIdx >= macdIndex || priceIndex <= 0)
                return false;

            // Find the lowest price in the lookback period
            int lowestPriceIdx = startIdx;
            for (int i = startIdx + 1; i < priceIndex; i++)
            {
                if (prices[i].Low < prices[lowestPriceIdx].Low)
                    lowestPriceIdx = i;
            }

            // Find the corresponding MACD value
            int macdLowIdx = lowestPriceIdx - (priceIndex - macdIndex);
            if (macdLowIdx < 0 || macdLowIdx >= macdLine.Count)
                return false;

            // Find the most recent lower low in price (if it exists)
            if (prices[priceIndex].Low < prices[lowestPriceIdx].Low)
            {
                // If current price made a lower low, check if MACD made a higher low
                return macdLine[macdIndex] > macdLine[macdLowIdx];
            }
            return false;
        }

        /// <summary>
        /// Check for bearish divergence: price makes higher highs, but MACD makes lower highs
        /// </summary>
        private bool CheckBearishDivergence(List<HistoricalPrice> prices, List<double> macdLine, int priceIndex, int macdIndex, int startIdx)
        {
            if (startIdx >= macdIndex || priceIndex <= 0)
                return false;

            // Find the highest price in the lookback period
            int highestPriceIdx = startIdx;
            for (int i = startIdx + 1; i < priceIndex; i++)
            {
                if (prices[i].High > prices[highestPriceIdx].High)
                    highestPriceIdx = i;
            }

            // Find the corresponding MACD value
            int macdHighIdx = highestPriceIdx - (priceIndex - macdIndex);
            if (macdHighIdx < 0 || macdHighIdx >= macdLine.Count)
                return false;

            // Find the most recent higher high in price (if it exists)
            if (prices[priceIndex].High > prices[highestPriceIdx].High)
            {
                // If current price made a higher high, check if MACD made a lower high
                return macdLine[macdIndex] < macdLine[macdHighIdx];
            }
            return false;
        }

        /// <summary>
        /// Calculate MACD, Signal line, and Histogram
        /// </summary>
        private (List<double> macdLine, List<double> signalLine, List<double> histogram) 
            CalculateMACD(List<HistoricalPrice> prices, int fastPeriod, int slowPeriod, int signalPeriod)
        {
            if (prices.Count < Math.Max(fastPeriod, slowPeriod))
                return (null, null, null);

            // Calculate EMAs
            List<double> fastEma = CalculateEMA(prices.Select(p => p.Close).ToList(), fastPeriod);
            List<double> slowEma = CalculateEMA(prices.Select(p => p.Close).ToList(), slowPeriod);
            if (fastEma == null || slowEma == null)
                return (null, null, null);

            // Calculate MACD line (fastEMA - slowEMA)
            var macdLine = new List<double>();
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < slowPeriod - 1)
                {
                    macdLine.Add(double.NaN);
                }
                else
                {
                    int fastIdx = i - (slowPeriod - fastPeriod);
                    if (fastIdx >= 0 && fastIdx < fastEma.Count && i < slowEma.Count)
                    {
                        macdLine.Add(fastEma[fastIdx] - slowEma[i]);
                    }
                    else
                    {
                        macdLine.Add(double.NaN);
                    }
                }
            }

            // Calculate Signal line (EMA of MACD)
            var signalLine = CalculateEMA(macdLine, signalPeriod);
            if (signalLine == null)
                return (macdLine, null, null);

            // Calculate histogram (MACD - Signal)
            var histogram = new List<double>();
            for (int i = 0; i < macdLine.Count; i++)
            {
                if (i < slowPeriod + signalPeriod - 2)
                {
                    histogram.Add(double.NaN);
                }
                else
                {
                    int signalIdx = i - (slowPeriod - 1);
                    if (signalIdx >= 0 && signalIdx < signalLine.Count)
                    {
                        histogram.Add(macdLine[i] - signalLine[signalIdx]);
                    }
                    else
                    {
                        histogram.Add(double.NaN);
                    }
                }
            }

            return (macdLine, signalLine, histogram);
        }

        /// <summary>
        /// Calculate Exponential Moving Average
        /// </summary>
        private List<double> CalculateEMA(List<double> values, int period)
        {
            if (values.Count < period)
                return null;

            var result = new List<double>();

            // Add NaN for periods where EMA can't be calculated
            for (int i = 0; i < period - 1; i++)
            {
                result.Add(double.NaN);
            }

            // Calculate first EMA as SMA
            double sum = 0;
            for (int i = 0; i < period; i++)
            {
                sum += values[i];
            }
            double ema = sum / period;
            result.Add(ema);

            // Calculate multiplier
            double multiplier = 2.0 / (period + 1);

            // Calculate EMA for remaining values
            for (int i = period; i < values.Count; i++)
            {
                // EMA = (Close - Previous EMA) * multiplier + Previous EMA
                ema = (values[i] - ema) * multiplier + ema;
                result.Add(ema);
            }

            return result;
        }
    }
}