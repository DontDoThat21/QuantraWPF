using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Modules;
using Quantra.Services;
using Quantra.Services.Interfaces;

namespace Quantra.Documentation
{
    /// <summary>
    /// Example showcasing how to use the Indicator Correlation Analysis module
    /// </summary>
    public class IndicatorCorrelationExample
    {
        private readonly ITechnicalIndicatorService _indicatorService;
        
        public IndicatorCorrelationExample()
        {
            _indicatorService = ServiceLocator.Resolve<ITechnicalIndicatorService>();
        }
        
        /// <summary>
        /// Example 1: Find correlations between RSI and MACD
        /// </summary>
        public async Task Example1_IndicatorCorrelations()
        {
            string symbol = "AAPL";
            string timeframe = "1day";
            
            // Get the correlation between RSI and MACD
            double correlation = await _indicatorService.GetIndicatorCorrelation(
                symbol, timeframe, "RSI", "MACD", 30);
                
            Console.WriteLine($"Correlation between RSI and MACD for {symbol}: {correlation:F4}");
            
            // Interpret the correlation value
            if (correlation > 0.7)
            {
                Console.WriteLine("Strong positive correlation: These indicators tend to move together");
            }
            else if (correlation < -0.7)
            {
                Console.WriteLine("Strong negative correlation: These indicators tend to move in opposite directions");
            }
            else if (Math.Abs(correlation) < 0.3)
            {
                Console.WriteLine("Weak correlation: These indicators have little relationship");
            }
            else
            {
                Console.WriteLine("Moderate correlation");
            }
        }
        
        /// <summary>
        /// Example 2: Find confirmation patterns across multiple indicators
        /// </summary>
        public async Task Example2_FindConfirmationPatterns()
        {
            string symbol = "MSFT";
            string timeframe = "1day";
            
            // Get confirmation patterns over the last 14 periods
            var patterns = await _indicatorService.FindIndicatorConfirmationPatterns(symbol, timeframe, 14);
            
            Console.WriteLine($"Found {patterns.Count} confirmation patterns for {symbol}");
            
            // Print details of each pattern
            foreach (var pattern in patterns)
            {
                Console.WriteLine($"Date: {pattern.Date}, Type: {pattern.SignalType}, Strength: {pattern.ConfirmationStrength:F2}");
                Console.WriteLine($"Confirming indicators: {string.Join(", ", pattern.ConfirmingIndicators)}");
                
                if (pattern.ConflictingIndicators.Count > 0)
                {
                    Console.WriteLine($"Conflicting indicators: {string.Join(", ", pattern.ConflictingIndicators)}");
                }
                
                Console.WriteLine();
            }
        }
        
        /// <summary>
        /// Example 3: Get comprehensive correlation data for visualization
        /// </summary>
        public async Task Example3_VisualizationData()
        {
            string symbol = "AMZN";
            string timeframe = "1day";
            
            // Define indicators to analyze
            var indicators = new List<string> { "RSI", "MACD", "BollingerBands", "VWAP", "ADX", "StochRSI" };
            
            // Get visualization data
            var visualData = await _indicatorService.GetIndicatorCorrelationVisualData(
                symbol, timeframe, indicators, 30);
                
            Console.WriteLine($"Correlation matrix for {symbol}:");
            
            // Print correlation matrix
            foreach (var indicator1 in indicators)
            {
                foreach (var indicator2 in indicators)
                {
                    if (visualData.CorrelationMatrix.ContainsKey(indicator1) &&
                        visualData.CorrelationMatrix[indicator1].ContainsKey(indicator2))
                    {
                        double corr = visualData.CorrelationMatrix[indicator1][indicator2];
                        Console.WriteLine($"{indicator1} vs {indicator2}: {corr:F4}");
                    }
                }
            }
            
            // Confirmation patterns are also included in the visualization data
            Console.WriteLine($"Found {visualData.ConfirmationPatterns.Count} confirmation patterns");
        }
        
        /// <summary>
        /// Example 4: Using confirmation patterns for trading decisions
        /// </summary>
        public async Task Example4_TradingDecisions()
        {
            string symbol = "TSLA";
            string timeframe = "1day";
            
            // Get confirmation patterns
            var patterns = await _indicatorService.FindIndicatorConfirmationPatterns(symbol, timeframe, 3);
            
            // Check for recent strong confirmation patterns
            var strongPatterns = patterns.Where(p => p.ConfirmationStrength > 0.7).ToList();
            
            if (strongPatterns.Count > 0)
            {
                var latestPattern = strongPatterns.OrderByDescending(p => p.Date).First();
                
                Console.WriteLine($"Found a strong {latestPattern.SignalType} pattern on {latestPattern.Date}");
                Console.WriteLine($"Confirming indicators: {string.Join(", ", latestPattern.ConfirmingIndicators)}");
                
                // Simple trading decision based on the pattern
                if (latestPattern.SignalType == SignalType.Bullish && latestPattern.ConfirmationStrength > 0.8)
                {
                    Console.WriteLine("TRADING SIGNAL: Consider buying based on strong bullish confirmation");
                }
                else if (latestPattern.SignalType == SignalType.Bearish && latestPattern.ConfirmationStrength > 0.8)
                {
                    Console.WriteLine("TRADING SIGNAL: Consider selling based on strong bearish confirmation");
                }
                else
                {
                    Console.WriteLine("No strong trading signal at this time");
                }
            }
            else
            {
                Console.WriteLine("No strong confirmation patterns found recently");
            }
        }
    }
}