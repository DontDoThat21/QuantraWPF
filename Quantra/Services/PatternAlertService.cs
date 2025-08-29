using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;
using Quantra.Utilities;
using static Quantra.Services.PricePatternRecognitionService;

namespace Quantra.Services
{
    /// <summary>
    /// Service for detecting and alerting on price patterns
    /// </summary>
    public class PatternAlertService
    {
        private readonly PricePatternRecognitionService _patternService;
        private readonly Dictionary<string, DateTime> _lastAlertTimes = new Dictionary<string, DateTime>();
        private readonly TimeSpan _minimumAlertInterval = TimeSpan.FromHours(24); // Minimum time between alerts for the same pattern/symbol

        public PatternAlertService(PricePatternRecognitionService patternService)
        {
            _patternService = patternService ?? throw new ArgumentNullException(nameof(patternService));
        }

        /// <summary>
        /// Detects patterns and generates alerts for a specific symbol
        /// </summary>
        /// <param name="symbol">Stock symbol to check</param>
        /// <param name="confidenceThreshold">Minimum confidence level for alert generation (0-100)</param>
        /// <returns>Number of alerts generated</returns>
        public async Task<int> DetectAndAlertPatternsAsync(string symbol, double confidenceThreshold = 70)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));
            
            try
            {
                // Detect patterns for the symbol
                var patterns = await _patternService.DetectAllPatternsAsync(symbol);
                
                // Filter by confidence level
                var highConfidencePatterns = patterns
                    .Where(p => p.Confidence >= confidenceThreshold)
                    .ToList();
                
                int alertsGenerated = 0;
                
                // Generate alerts for high confidence patterns
                foreach (var pattern in highConfidencePatterns)
                {
                    // Create a unique key for this symbol/pattern type
                    string alertKey = $"{symbol}_{pattern.Type}";
                    
                    // Check if we've sent an alert for this pattern recently
                    if (_lastAlertTimes.ContainsKey(alertKey) &&
                        (DateTime.Now - _lastAlertTimes[alertKey]) < _minimumAlertInterval)
                    {
                        continue; // Skip if we've alerted for this pattern recently
                    }
                    
                    // Create and emit alert
                    var alert = CreatePatternAlert(pattern);
                    if (alert != null)
                    {
                        AlertManager.EmitGlobalAlert(alert);
                        _lastAlertTimes[alertKey] = DateTime.Now;
                        alertsGenerated++;
                    }
                }
                
                return alertsGenerated;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to detect patterns for {symbol}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Checks for patterns across multiple symbols
        /// </summary>
        /// <param name="symbols">List of symbols to check</param>
        /// <param name="confidenceThreshold">Minimum confidence level</param>
        /// <returns>Total number of alerts generated</returns>
        public async Task<int> DetectPatternsForSymbolsAsync(List<string> symbols, double confidenceThreshold = 70)
        {
            if (symbols == null || symbols.Count == 0)
                return 0;
            
            int totalAlerts = 0;
            
            foreach (var symbol in symbols)
            {
                int alertsForSymbol = await DetectAndAlertPatternsAsync(symbol, confidenceThreshold);
                totalAlerts += alertsForSymbol;
            }
            
            return totalAlerts;
        }

        /// <summary>
        /// Creates an alert from a detected pattern
        /// </summary>
        private AlertModel CreatePatternAlert(PricePattern pattern)
        {
            if (pattern == null)
                return null;
            
            string patternName = GetPatternDisplayName(pattern.Type);
            string biasIndicator = pattern.Bias == PatternBias.Bullish ? "🔼" : 
                                  pattern.Bias == PatternBias.Bearish ? "🔽" : "◼️";
            
            // Format condition text
            string condition = $"{biasIndicator} {patternName}";
            
            // Generate notes with key levels
            string notes = $"{patternName} detected with {pattern.Confidence:F0}% confidence.\n\n";
            
            // Add key price levels
            if (pattern.KeyLevels.Count > 0)
            {
                notes += "Key price levels:\n";
                foreach (var level in pattern.KeyLevels)
                {
                    notes += $"- {level.Key}: ${level.Value:F2}\n";
                }
            }
            
            // Add description based on pattern type
            notes += $"\n{GetPatternDescription(pattern.Type)}";
            
            // Get priority based on confidence and recency
            int priority = pattern.Confidence >= 85 ? 1 : 2; // High priority for high confidence patterns
            
            var alert = new AlertModel
            {
                Name = $"{pattern.Symbol} {patternName}",
                Symbol = pattern.Symbol,
                Condition = condition,
                AlertType = "Pattern Recognition",
                Category = AlertCategory.Pattern,
                IsActive = true,
                IsTriggered = false, // Set to true when pattern completes
                CreatedDate = DateTime.Now,
                Notes = notes,
                Priority = priority
            };
            
            // Add pattern-specific properties if needed
            switch (pattern.Type)
            {
                case PatternType.DoubleTop:
                case PatternType.DoubleBottom:
                case PatternType.HeadAndShoulders:
                case PatternType.InverseHeadAndShoulders:
                    // Set target price based on pattern projection
                    if (pattern.KeyLevels.ContainsKey("Target"))
                    {
                        alert.TriggerPrice = pattern.KeyLevels["Target"];
                    }
                    break;
            }
            
            return alert;
        }

        /// <summary>
        /// Gets a display-friendly name for a pattern type
        /// </summary>
        private string GetPatternDisplayName(PatternType type)
        {
            return type switch
            {
                PatternType.DoubleTop => "Double Top",
                PatternType.DoubleBottom => "Double Bottom",
                PatternType.HeadAndShoulders => "Head & Shoulders",
                PatternType.InverseHeadAndShoulders => "Inverse Head & Shoulders",
                PatternType.AscendingTriangle => "Ascending Triangle",
                PatternType.DescendingTriangle => "Descending Triangle",
                PatternType.SymmetricalTriangle => "Symmetrical Triangle",
                PatternType.BullishEngulfing => "Bullish Engulfing",
                PatternType.BearishEngulfing => "Bearish Engulfing",
                PatternType.BullishHarami => "Bullish Harami",
                PatternType.BearishHarami => "Bearish Harami",
                _ => type.ToString()
            };
        }
        
        /// <summary>
        /// Gets a description of what the pattern indicates
        /// </summary>
        private string GetPatternDescription(PatternType type)
        {
            return type switch
            {
                PatternType.DoubleTop => "A double top is a reversal pattern that forms after an uptrend when a price tests the same resistance level twice and fails to break through. This typically signals a bearish reversal in the trend.",
                
                PatternType.DoubleBottom => "A double bottom is a reversal pattern that forms after a downtrend when a price tests the same support level twice and bounces up. This typically signals a bullish reversal in the trend.",
                
                PatternType.HeadAndShoulders => "The head and shoulders pattern is a reversal pattern characterized by a peak (the head) with lower peaks on each side (the shoulders). When the neckline breaks, it typically signals a bearish reversal.",
                
                PatternType.InverseHeadAndShoulders => "The inverse head and shoulders pattern is a bullish reversal pattern characterized by a trough (the head) with higher troughs on each side (the shoulders). When the neckline breaks, it typically signals a bullish reversal.",
                
                PatternType.AscendingTriangle => "An ascending triangle is a continuation pattern characterized by a flat upper trendline and an ascending lower trendline. It generally signals a bullish continuation or a bullish breakout.",
                
                PatternType.DescendingTriangle => "A descending triangle is a continuation pattern characterized by a flat lower trendline and a descending upper trendline. It generally signals a bearish continuation or a bearish breakdown.",
                
                PatternType.SymmetricalTriangle => "A symmetrical triangle forms when converging trendlines connect a series of lower highs and higher lows. It indicates a period of consolidation before a potential breakout in either direction.",
                
                PatternType.BullishEngulfing => "A bullish engulfing pattern occurs when a small bearish candle is followed by a larger bullish candle that completely engulfs the previous candle. This signals a potential bullish reversal.",
                
                PatternType.BearishEngulfing => "A bearish engulfing pattern occurs when a small bullish candle is followed by a larger bearish candle that completely engulfs the previous candle. This signals a potential bearish reversal.",
                
                PatternType.BullishHarami => "A bullish harami pattern forms when a small bullish candle is contained within the body of the previous larger bearish candle. It suggests a potential bullish reversal.",
                
                PatternType.BearishHarami => "A bearish harami pattern forms when a small bearish candle is contained within the body of the previous larger bullish candle. It suggests a potential bearish reversal.",
                
                _ => "Chart pattern detected in price action."
            };
        }
        
        /// <summary>
        /// Gets a list of available pattern types
        /// </summary>
        public static List<string> GetAvailablePatterns()
        {
            return Enum.GetValues(typeof(PatternType))
                .Cast<PatternType>()
                .Select(p => p.ToString())
                .ToList();
        }
    }
}