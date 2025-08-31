using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Represents historical data for technical indicators
    /// </summary>
    public class HistoricalIndicatorData
    {
        /// <summary>
        /// The symbol this data is for
        /// </summary>
        public string Symbol { get; set; }
        
        /// <summary>
        /// The type of indicator (RSI, MACD, etc.)
        /// </summary>
        public string IndicatorType { get; set; }
        
        /// <summary>
        /// List of dates for the data points
        /// </summary>
        public List<DateTime> Dates { get; set; } = new List<DateTime>();
        
        /// <summary>
        /// Primary indicator values
        /// </summary>
        public List<double> Values { get; set; } = new List<double>();
        
        /// <summary>
        /// Secondary indicator values (e.g., signal line for MACD)
        /// </summary>
        public List<double> SecondaryValues { get; set; } = new List<double>();
        
        /// <summary>
        /// Tertiary indicator values (e.g., histogram for MACD)
        /// </summary>
        public List<double> TertiaryValues { get; set; } = new List<double>();
        
        /// <summary>
        /// Additional properties or context for the indicator
        /// </summary>
        public Dictionary<string, object> Properties { get; set; } = new Dictionary<string, object>();
        
        /// <summary>
        /// The most recent value of the primary indicator
        /// </summary>
        public double CurrentValue => Values.Count > 0 ? Values[Values.Count - 1] : 0;
        
        /// <summary>
        /// The most recent value of the secondary indicator
        /// </summary>
        public double CurrentSecondaryValue => SecondaryValues.Count > 0 ? SecondaryValues[SecondaryValues.Count - 1] : 0;
        
        /// <summary>
        /// Returns a simple analysis of the indicator trend
        /// </summary>
        public string GetTrendAnalysis()
        {
            if (Values.Count < 2)
                return "Insufficient data";
            
            // Compare last few values to determine trend
            int lookback = Math.Min(5, Values.Count - 1);
            double current = Values[Values.Count - 1];
            double previous = Values[Values.Count - 1 - lookback];
            
            double percentChange = (current - previous) / Math.Abs(previous) * 100;
            
            if (percentChange > 5)
                return "Strong Uptrend";
            else if (percentChange > 1)
                return "Uptrend";
            else if (percentChange < -5)
                return "Strong Downtrend";
            else if (percentChange < -1)
                return "Downtrend";
            else
                return "Sideways/Neutral";
        }
        
        /// <summary>
        /// Returns a trading signal based on indicator values
        /// </summary>
        public string GetTradingSignal()
        {
            switch (IndicatorType.ToUpperInvariant())
            {
                case "RSI":
                    return GetRsiSignal();
                case "MACD":
                    return GetMacdSignal();
                case "VOLUME":
                    return GetVolumeSignal();
                // Add more indicator-specific signal methods as needed
                default:
                    return "No signal available";
            }
        }
        
        private string GetRsiSignal()
        {
            if (Values.Count == 0)
                return "No data";
            
            double current = Values[Values.Count - 1];
            
            if (current > 70)
                return "Overbought - Sell/Short";
            else if (current < 30)
                return "Oversold - Buy/Long";
            else
                return "Neutral";
        }
        
        private string GetMacdSignal()
        {
            if (Values.Count < 2 || SecondaryValues.Count < 2)
                return "Insufficient data";
            
            double currentMacd = Values[Values.Count - 1];
            double previousMacd = Values[Values.Count - 2];
            double currentSignal = SecondaryValues[SecondaryValues.Count - 1];
            double previousSignal = SecondaryValues[SecondaryValues.Count - 2];
            
            // MACD crossing above signal line is bullish
            if (previousMacd < previousSignal && currentMacd > currentSignal)
                return "Bullish Crossover - Buy";
            
            // MACD crossing below signal line is bearish
            if (previousMacd > previousSignal && currentMacd < currentSignal)
                return "Bearish Crossover - Sell";
            
            // MACD above signal line is generally bullish
            if (currentMacd > currentSignal)
                return "Bullish Trend";
            
            // MACD below signal line is generally bearish
            if (currentMacd < currentSignal)
                return "Bearish Trend";
            
            return "Neutral";
        }
        
        private string GetVolumeSignal()
        {
            if (Values.Count < 5)
                return "Insufficient data";
            
            // Calculate average volume
            double sum = 0;
            for (int i = Values.Count - 5; i < Values.Count; i++)
            {
                sum += Values[i];
            }
            double avgVolume = sum / 5;
            
            double currentVolume = Values[Values.Count - 1];
            
            if (currentVolume > avgVolume * 1.5)
                return "High Volume - Strong Signal";
            else if (currentVolume > avgVolume * 1.2)
                return "Above Average - Confirming";
            else if (currentVolume < avgVolume * 0.8)
                return "Low Volume - Weak Confirmation";
            else
                return "Average Volume";
        }
    }
}
