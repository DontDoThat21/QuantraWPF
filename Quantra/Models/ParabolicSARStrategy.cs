using System;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a trading strategy based on Parabolic SAR (Stop and Reverse)
    /// </summary>
    public class ParabolicSARStrategy : TradingStrategyProfile
    {
        private double _accelerationFactor = 0.02;
        private double _maxAccelerationFactor = 0.2;
        private bool _useConfirmationCandle = true;
        private bool _requireVolumeConfirmation = false;

        public ParabolicSARStrategy()
        {
            Name = "Parabolic SAR";
            Description = "Generates signals based on the Parabolic SAR (Stop and Reverse) indicator. " +
                          "Buy signal when price crosses above the SAR value. " +
                          "Sell signal when price crosses below the SAR value. " +
                          "Uses trend following with adaptable acceleration factor.";
            RiskLevel = 0.6;
            MinConfidence = 0.65;
        }

        /// <summary>
        /// Initial acceleration factor for SAR calculation (typically 0.02)
        /// </summary>
        public double AccelerationFactor
        {
            get => _accelerationFactor;
            set
            {
                if (value > 0 && value < _maxAccelerationFactor && _accelerationFactor != value)
                {
                    _accelerationFactor = value;
                    OnPropertyChanged(nameof(AccelerationFactor));
                }
            }
        }

        /// <summary>
        /// Maximum acceleration factor for SAR calculation (typically 0.2)
        /// </summary>
        public double MaxAccelerationFactor
        {
            get => _maxAccelerationFactor;
            set
            {
                if (value > _accelerationFactor && _maxAccelerationFactor != value)
                {
                    _maxAccelerationFactor = value;
                    OnPropertyChanged(nameof(MaxAccelerationFactor));
                }
            }
        }

        /// <summary>
        /// Whether to wait for a confirmation candle after a signal
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

        /// <summary>
        /// Whether to require increased volume for signal confirmation
        /// </summary>
        public bool RequireVolumeConfirmation
        {
            get => _requireVolumeConfirmation;
            set
            {
                if (_requireVolumeConfirmation != value)
                {
                    _requireVolumeConfirmation = value;
                    OnPropertyChanged(nameof(RequireVolumeConfirmation));
                }
            }
        }

        public override IEnumerable<string> RequiredIndicators => new[] 
        { 
            "SAR", "SAR_EP", "SAR_AF", "SAR_Trend" 
        };

        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || prices.Count < 3)
                return null;

            int currentIndex = index ?? prices.Count - 1;
            if (currentIndex < 2 || currentIndex >= prices.Count)
                return null;

            // Calculate Parabolic SAR values
            var (sarValues, epValues, isUptrend) = CalculateParabolicSAR(prices, AccelerationFactor, MaxAccelerationFactor);
            
            // Check if we have valid SAR values
            if (sarValues == null || sarValues.Count == 0)
                return null;

            // Adjust index for SAR values (which start with 2 NaN entries)
            int sarIndex = Math.Min(currentIndex, sarValues.Count - 1);
            if (sarIndex < 1 || double.IsNaN(sarValues[sarIndex]) || double.IsNaN(sarValues[sarIndex - 1]))
                return null;

            bool isBullishCrossover = false;
            bool isBearishCrossover = false;

            // Current values
            double currentSAR = sarValues[sarIndex];
            double currentClose = prices[currentIndex].Close;
            double previousSAR = sarValues[sarIndex - 1];
            double previousClose = prices[currentIndex - 1].Close;

            // Check for bullish crossover (SAR moves from above to below price)
            if (previousSAR > previousClose && currentSAR < currentClose)
                isBullishCrossover = true;

            // Check for bearish crossover (SAR moves from below to above price)
            if (previousSAR < previousClose && currentSAR > currentClose)
                isBearishCrossover = true;

            // If confirmation candle is required, check the next candle trend
            if (UseConfirmationCandle && currentIndex > 2)
            {
                // Check if the trend is continuing after crossover
                if (isBullishCrossover && prices[currentIndex].Close < prices[currentIndex - 1].Close)
                    isBullishCrossover = false;

                if (isBearishCrossover && prices[currentIndex].Close > prices[currentIndex - 1].Close)
                    isBearishCrossover = false;
            }

            // If volume confirmation is required, check for increased volume
            if (RequireVolumeConfirmation && prices[0].Volume > 0)
            {
                int lookback = Math.Min(5, currentIndex);
                double avgVolume = prices.Skip(currentIndex - lookback).Take(lookback).Average(p => p.Volume);
                bool volumeConfirms = prices[currentIndex].Volume > avgVolume * 1.1; // At least 10% higher than average
                
                if (!volumeConfirms)
                {
                    isBullishCrossover = false;
                    isBearishCrossover = false;
                }
            }

            // Generate signals based on crossovers
            if (isBullishCrossover)
                return "BUY";
            else if (isBearishCrossover)
                return "SELL";

            return null;
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            if (indicators == null)
                return false;

            // Check if we have the required SAR indicator and price
            if (!indicators.TryGetValue("SAR", out double sar) ||
                !indicators.TryGetValue("Price", out double price) && 
                !indicators.TryGetValue("Close", out price))
            {
                return false;
            }

            // Check if we have trend information
            bool isUptrend = false;
            if (indicators.TryGetValue("SAR_Trend", out double trend))
            {
                isUptrend = trend > 0;
            }
            else
            {
                // If trend not provided, infer from SAR position relative to price
                isUptrend = price > sar;
            }

            // Check for potentially valid conditions based on SAR position
            bool nearSignal = Math.Abs((price - sar) / price) < 0.01; // Within 1% of crossover
            return nearSignal;
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
                bool potentialReversal = (currentUptrend && prices[i].Low < sar) ||
                                        (!currentUptrend && prices[i].High > sar);

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
    }
}