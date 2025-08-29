using System;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.Models
{
    /// <summary>
    /// Implements a strategy based on Ichimoku Cloud (Ichimoku Kinko Hyo) indicators
    /// </summary>
    public class IchimokuCloudStrategy : TradingStrategyProfile
    {
        private int _tenkanPeriod = 9;
        private int _kijunPeriod = 26;
        private int _senkouBPeriod = 52;
        private int _displacement = 26;
        private bool _requireConfirmation = true;

        public IchimokuCloudStrategy()
        {
            Name = "Ichimoku Cloud";
            Description = "Generates signals based on Ichimoku Cloud components. " +
                          "Buy signal when price moves above the cloud with bullish TK cross. " +
                          "Sell signal when price moves below the cloud with bearish TK cross. " +
                          "Can also consider Chikou span for confirmation.";
            RiskLevel = 0.65;
            MinConfidence = 0.7;
        }

        /// <summary>
        /// Tenkan-sen (Conversion Line) period
        /// </summary>
        public int TenkanPeriod
        {
            get => _tenkanPeriod;
            set
            {
                if (value > 0 && _tenkanPeriod != value)
                {
                    _tenkanPeriod = value;
                    OnPropertyChanged(nameof(TenkanPeriod));
                }
            }
        }

        /// <summary>
        /// Kijun-sen (Base Line) period
        /// </summary>
        public int KijunPeriod
        {
            get => _kijunPeriod;
            set
            {
                if (value > 0 && _kijunPeriod != value)
                {
                    _kijunPeriod = value;
                    OnPropertyChanged(nameof(KijunPeriod));
                }
            }
        }

        /// <summary>
        /// Senkou Span B period
        /// </summary>
        public int SenkouBPeriod
        {
            get => _senkouBPeriod;
            set
            {
                if (value > 0 && _senkouBPeriod != value)
                {
                    _senkouBPeriod = value;
                    OnPropertyChanged(nameof(SenkouBPeriod));
                }
            }
        }

        /// <summary>
        /// Displacement period (for Senkou spans and Chikou span)
        /// </summary>
        public int Displacement
        {
            get => _displacement;
            set
            {
                if (value > 0 && _displacement != value)
                {
                    _displacement = value;
                    OnPropertyChanged(nameof(Displacement));
                }
            }
        }

        /// <summary>
        /// Whether to require Chikou span confirmation
        /// </summary>
        public bool RequireConfirmation
        {
            get => _requireConfirmation;
            set
            {
                if (_requireConfirmation != value)
                {
                    _requireConfirmation = value;
                    OnPropertyChanged(nameof(RequireConfirmation));
                }
            }
        }

        public override IEnumerable<string> RequiredIndicators => new[] { "Tenkan", "Kijun", "SenkouA", "SenkouB", "Chikou" };

        public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
        {
            if (prices == null || prices.Count < SenkouBPeriod + Displacement)
                return null;

            int currentIndex = index ?? prices.Count - 1;
            if (currentIndex < SenkouBPeriod + Displacement || currentIndex >= prices.Count)
                return null;

            // Calculate Ichimoku components
            var components = CalculateIchimoku(prices, TenkanPeriod, KijunPeriod, SenkouBPeriod, Displacement);
            
            // Get values for the current candle
            double currentTenkan = components.tenkan[currentIndex];
            double currentKijun = components.kijun[currentIndex];
            double currentSenkouA = components.senkouA[currentIndex - Displacement];
            double currentSenkouB = components.senkouB[currentIndex - Displacement];
            double currentPrice = prices[currentIndex].Close;
            
            // Get values for the previous candle
            double prevTenkan = components.tenkan[currentIndex - 1];
            double prevKijun = components.kijun[currentIndex - 1];

            // Calculate the cloud top and bottom
            double cloudTop = Math.Max(currentSenkouA, currentSenkouB);
            double cloudBottom = Math.Min(currentSenkouA, currentSenkouB);

            // Check if Tenkan crossed above Kijun
            bool tenkanCrossedAboveKijun = prevTenkan <= prevKijun && currentTenkan > currentKijun;
            
            // Check if Tenkan crossed below Kijun
            bool tenkanCrossedBelowKijun = prevTenkan >= prevKijun && currentTenkan < currentKijun;

            // Check if price is above/below cloud
            bool priceAboveCloud = currentPrice > cloudTop;
            bool priceBelowCloud = currentPrice < cloudBottom;

            // Check Chikou span confirmation if required
            bool chikouConfirmation = true;
            if (RequireConfirmation && currentIndex - Displacement >= 0)
            {
                double chikouValue = components.chikou[currentIndex];
                double pastPrice = prices[currentIndex - Displacement].Close;
                chikouConfirmation = (chikouValue > pastPrice);
            }

            // Generate signals
            if (tenkanCrossedAboveKijun && priceAboveCloud && (!RequireConfirmation || chikouConfirmation))
            {
                return "BUY";
            }
            else if (tenkanCrossedBelowKijun && priceBelowCloud && (!RequireConfirmation || !chikouConfirmation))
            {
                return "SELL";
            }

            return null;
        }

        public override bool ValidateConditions(Dictionary<string, double> indicators)
        {
            return indicators.ContainsKey("Tenkan") && 
                   indicators.ContainsKey("Kijun") && 
                   indicators.ContainsKey("SenkouA") && 
                   indicators.ContainsKey("SenkouB");
        }

        /// <summary>
        /// Calculate Ichimoku Cloud components
        /// </summary>
        private (List<double> tenkan, List<double> kijun, List<double> senkouA, List<double> senkouB, List<double> chikou) 
            CalculateIchimoku(List<HistoricalPrice> prices, int tenkanPeriod, int kijunPeriod, int senkouBPeriod, int displacement)
        {
            var tenkan = new List<double>();
            var kijun = new List<double>();
            var senkouA = new List<double>();
            var senkouB = new List<double>();
            var chikou = new List<double>();

            // Calculate Tenkan-sen (Conversion Line) and Kijun-sen (Base Line)
            for (int i = 0; i < prices.Count; i++)
            {
                // Tenkan-sen = (highest high + lowest low) / 2 for tenkanPeriod
                if (i < tenkanPeriod - 1)
                {
                    tenkan.Add(double.NaN);
                }
                else
                {
                    double highestHigh = double.MinValue;
                    double lowestLow = double.MaxValue;

                    for (int j = i - tenkanPeriod + 1; j <= i; j++)
                    {
                        if (prices[j].High > highestHigh) highestHigh = prices[j].High;
                        if (prices[j].Low < lowestLow) lowestLow = prices[j].Low;
                    }

                    tenkan.Add((highestHigh + lowestLow) / 2);
                }

                // Kijun-sen = (highest high + lowest low) / 2 for kijunPeriod
                if (i < kijunPeriod - 1)
                {
                    kijun.Add(double.NaN);
                }
                else
                {
                    double highestHigh = double.MinValue;
                    double lowestLow = double.MaxValue;

                    for (int j = i - kijunPeriod + 1; j <= i; j++)
                    {
                        if (prices[j].High > highestHigh) highestHigh = prices[j].High;
                        if (prices[j].Low < lowestLow) lowestLow = prices[j].Low;
                    }

                    kijun.Add((highestHigh + lowestLow) / 2);
                }

                // Chikou Span = Current closing price plotted displacement periods back
                chikou.Add(prices[i].Close);
            }

            // Calculate Senkou Span A = (Tenkan-sen + Kijun-sen) / 2 projected displacement periods ahead
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < kijunPeriod - 1)
                {
                    senkouA.Add(double.NaN);
                }
                else
                {
                    double currentSenkouA = (tenkan[i] + kijun[i]) / 2;
                    senkouA.Add(currentSenkouA);
                }
            }

            // Calculate Senkou Span B = (highest high + lowest low) / 2 for senkouBPeriod projected displacement periods ahead
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < senkouBPeriod - 1)
                {
                    senkouB.Add(double.NaN);
                }
                else
                {
                    double highestHigh = double.MinValue;
                    double lowestLow = double.MaxValue;

                    for (int j = i - senkouBPeriod + 1; j <= i; j++)
                    {
                        if (prices[j].High > highestHigh) highestHigh = prices[j].High;
                        if (prices[j].Low < lowestLow) lowestLow = prices[j].Low;
                    }

                    senkouB.Add((highestHigh + lowestLow) / 2);
                }
            }

            return (tenkan, kijun, senkouA, senkouB, chikou);
        }
    }
}