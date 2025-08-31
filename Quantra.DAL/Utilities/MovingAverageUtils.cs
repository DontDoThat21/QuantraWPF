using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Models;

namespace Quantra.Utilities
{
    /// <summary>
    /// Utility class for calculating various types of moving averages
    /// </summary>
    public static class MovingAverageUtils
    {
        /// <summary>
        /// Calculate Simple Moving Average (SMA)
        /// </summary>
        /// <param name="values">Input data points</param>
        /// <param name="period">Moving average period</param>
        /// <returns>List of SMA values</returns>
        public static List<double> CalculateSMA(List<double> values, int period)
        {
            if (values == null || values.Count < period)
                return null;
                
            var result = new List<double>();
            
            // Add NaN for periods where SMA can't be calculated
            for (int i = 0; i < period - 1; i++)
            {
                result.Add(double.NaN);
            }
            
            // Calculate SMA for each subsequent period
            for (int i = period - 1; i < values.Count; i++)
            {
                double sum = 0;
                for (int j = i - period + 1; j <= i; j++)
                {
                    sum += values[j];
                }
                result.Add(sum / period);
            }
            
            return result;
        }
        
        /// <summary>
        /// Calculate Simple Moving Average from historical prices using close values
        /// </summary>
        /// <param name="prices">Historical price data</param>
        /// <param name="period">Moving average period</param>
        /// <returns>List of SMA values</returns>
        public static List<double> CalculateSMA(List<HistoricalPrice> prices, int period)
        {
            if (prices == null || prices.Count < period)
                return null;
                
            var values = prices.Select(p => p.Close).ToList();
            return CalculateSMA(values, period);
        }
        
        /// <summary>
        /// Calculate Exponential Moving Average (EMA)
        /// </summary>
        /// <param name="values">Input data points</param>
        /// <param name="period">Moving average period</param>
        /// <returns>List of EMA values</returns>
        public static List<double> CalculateEMA(List<double> values, int period)
        {
            if (values == null || values.Count < period)
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
                // EMA = (Current value - Previous EMA) * multiplier + Previous EMA
                ema = (values[i] - ema) * multiplier + ema;
                result.Add(ema);
            }

            return result;
        }
        
        /// <summary>
        /// Calculate Exponential Moving Average from historical prices using close values
        /// </summary>
        /// <param name="prices">Historical price data</param>
        /// <param name="period">Moving average period</param>
        /// <returns>List of EMA values</returns>
        public static List<double> CalculateEMA(List<HistoricalPrice> prices, int period)
        {
            if (prices == null || prices.Count < period)
                return null;
                
            var values = prices.Select(p => p.Close).ToList();
            return CalculateEMA(values, period);
        }
        
        /// <summary>
        /// Calculate Weighted Moving Average (WMA)
        /// </summary>
        /// <param name="values">Input data points</param>
        /// <param name="period">Moving average period</param>
        /// <returns>List of WMA values</returns>
        public static List<double> CalculateWMA(List<double> values, int period)
        {
            if (values == null || values.Count < period)
                return null;
                
            var result = new List<double>();
            
            // Add NaN for periods where WMA can't be calculated
            for (int i = 0; i < period - 1; i++)
            {
                result.Add(double.NaN);
            }
            
            // Calculate denominator (sum of weights)
            int denominator = (period * (period + 1)) / 2;
            
            // Calculate WMA for each subsequent period
            for (int i = period - 1; i < values.Count; i++)
            {
                double weightedSum = 0;
                
                for (int j = 0; j < period; j++)
                {
                    // Weight = position+1 where position is 0-based from oldest to newest
                    // This gives higher weight to more recent prices
                    int weight = j + 1; // 1 to period
                    weightedSum += values[i - (period - 1) + j] * weight;
                }
                
                result.Add(weightedSum / denominator);
            }
            
            return result;
        }
        
        /// <summary>
        /// Calculate Weighted Moving Average from historical prices using close values
        /// </summary>
        /// <param name="prices">Historical price data</param>
        /// <param name="period">Moving average period</param>
        /// <returns>List of WMA values</returns>
        public static List<double> CalculateWMA(List<HistoricalPrice> prices, int period)
        {
            if (prices == null || prices.Count < period)
                return null;
                
            var values = prices.Select(p => p.Close).ToList();
            return CalculateWMA(values, period);
        }
    }
}