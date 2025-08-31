using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Controls;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.Utilities;

namespace Quantra.DAL.Services
{
    public class TechnicalIndicatorAlertService
    {
        private readonly ITechnicalIndicatorService _indicatorService;

        public TechnicalIndicatorAlertService(ITechnicalIndicatorService indicatorService)
        {
            _indicatorService = indicatorService;
        }

        /// <summary>
        /// Checks if a technical indicator alert condition is met
        /// </summary>
        /// <param name="alert">The alert to check</param>
        /// <returns>True if the alert condition is met, false otherwise</returns>
        public async Task<bool> CheckAlertConditionAsync(AlertModel alert)
        {
            if (alert == null || !alert.IsActive || alert.IsTriggered || alert.Category != AlertCategory.TechnicalIndicator)
                return false;

            try
            {
                // Get the current indicator value based on the indicator name
                double indicatorValue = await GetIndicatorValueAsync(alert.Symbol, alert.IndicatorName);
                alert.CurrentIndicatorValue = indicatorValue;

                // Check if the condition is met
                bool conditionMet = EvaluateCondition(indicatorValue, alert.ComparisonOperator, alert.ThresholdValue);

                if (conditionMet && !alert.IsTriggered)
                {
                    // Trigger the alert
                    alert.IsTriggered = true;
                    alert.TriggeredDate = DateTime.Now;
                    
                    // Update the condition description for better readability
                    alert.Condition = $"{alert.IndicatorName} {GetOperatorDisplayString(alert.ComparisonOperator)} {alert.ThresholdValue:F2}";
                    
                    return true;
                }

                return false;
            }
            catch (Exception ex)
            {
                AlertsControl.EmitGlobalError($"Error checking alert condition for {alert.Name}: {ex.Message}", ex);
                return false;
            }
        }

        /// <summary>
        /// Check all active technical indicator alerts and trigger those with conditions met
        /// </summary>
        /// <param name="alerts">List of alerts to check</param>
        /// <returns>Number of alerts triggered</returns>
        public async Task<int> CheckAllAlertsAsync(List<AlertModel> alerts)
        {
            if (alerts == null || alerts.Count == 0)
                return 0;

            int triggeredCount = 0;
            List<Task<bool>> checkTasks = new List<Task<bool>>();

            foreach (var alert in alerts)
            {
                if (alert.Category == AlertCategory.TechnicalIndicator && alert.IsActive && !alert.IsTriggered)
                {
                    checkTasks.Add(CheckAlertConditionAsync(alert));
                }
            }

            // Use simple throttling for alert checks to prevent resource contention
            using var throttler = new ConcurrentTaskThrottler(Math.Min(checkTasks.Count, 4));
            var taskFactories = checkTasks.Select(task => new Func<Task<bool>>(() => task));
            bool[] results = await throttler.ExecuteThrottledAsync(taskFactories);
            foreach (bool triggered in results)
            {
                if (triggered) triggeredCount++;
            }

            return triggeredCount;
        }

        /// <summary>
        /// Evaluates if a condition is met based on the comparison operator
        /// </summary>
        private bool EvaluateCondition(double currentValue, ComparisonOperator op, double thresholdValue)
        {
            return op switch
            {
                ComparisonOperator.Equal => Math.Abs(currentValue - thresholdValue) < 0.0001,
                ComparisonOperator.NotEqual => Math.Abs(currentValue - thresholdValue) >= 0.0001,
                ComparisonOperator.GreaterThan => currentValue > thresholdValue,
                ComparisonOperator.LessThan => currentValue < thresholdValue,
                ComparisonOperator.GreaterThanOrEqual => currentValue >= thresholdValue,
                ComparisonOperator.LessThanOrEqual => currentValue <= thresholdValue,
                ComparisonOperator.CrossesAbove => currentValue > thresholdValue, // Simplified for now
                ComparisonOperator.CrossesBelow => currentValue < thresholdValue, // Simplified for now
                _ => false
            };
        }

        /// <summary>
        /// Gets the display string for a comparison operator
        /// </summary>
        public static string GetOperatorDisplayString(ComparisonOperator op)
        {
            return op switch
            {
                ComparisonOperator.Equal => "=",
                ComparisonOperator.NotEqual => "≠",
                ComparisonOperator.GreaterThan => ">",
                ComparisonOperator.LessThan => "<",
                ComparisonOperator.GreaterThanOrEqual => "≥",
                ComparisonOperator.LessThanOrEqual => "≤",
                ComparisonOperator.CrossesAbove => "crosses above",
                ComparisonOperator.CrossesBelow => "crosses below",
                _ => "?"
            };
        }

        /// <summary>
        /// Gets the indicator value for a specified symbol and indicator
        /// </summary>
        private async Task<double> GetIndicatorValueAsync(string symbol, string indicatorName)
        {
            string timeframe = "1day"; // Default timeframe, could be made configurable

            // Use the appropriate method based on the indicator name
            switch (indicatorName)
            {
                case "RSI":
                    return await _indicatorService.GetRSI(symbol, timeframe);
                case "ADX":
                    return await _indicatorService.GetADX(symbol, timeframe);
                case "ATR":
                    return await _indicatorService.GetATR(symbol, timeframe);
                case "Momentum":
                    return await _indicatorService.GetMomentum(symbol, timeframe);
                case "OBV":
                    return await _indicatorService.GetOBV(symbol, timeframe);
                case "MFI":
                    return await _indicatorService.GetMFI(symbol, timeframe);
                case "ParabolicSAR":
                    return await _indicatorService.GetParabolicSAR(symbol, timeframe);
                case "StochK":
                case "StochD":
                    var stoch = await _indicatorService.GetStochastic(symbol, timeframe);
                    return indicatorName == "StochK" ? stoch.K : stoch.D;
                case "Volume":
                    var volumeIndicators = await _indicatorService.CalculateIndicators(symbol, timeframe);
                    if (volumeIndicators.TryGetValue("Volume", out double volume))
                    {
                        return volume;
                    }
                    return 0;
                case "VolumeSpike":
                    var indicators = await _indicatorService.CalculateIndicators(symbol, timeframe);
                    if (indicators.TryGetValue("volume_ratio", out double volumeRatio))
                    {
                        return volumeRatio; // Return the volume ratio as an indicator of volume spike
                    }
                    else if (indicators.TryGetValue("volume_z_score", out double volumeZScore))
                    {
                        return volumeZScore; // Alternative: return z-score if ratio not available
                    }
                    return 1.0; // Default value (no spike)
                default:
                    // If not a built-in indicator, try to get it from general indicators
                    var generalIndicators = await _indicatorService.CalculateIndicators(symbol, timeframe);
                    if (generalIndicators.TryGetValue(indicatorName, out double value))
                    {
                        return value;
                    }
                    throw new InvalidOperationException($"Indicator '{indicatorName}' not found");
            }
        }

        /// <summary>
        /// Gets a list of available technical indicator names
        /// </summary>
        public static List<string> GetAvailableIndicators()
        {
            return new List<string>
            {
                "RSI",
                "MACD",
                "ADX",
                "ATR",
                "Momentum",
                "OBV",
                "MFI",
                "ParabolicSAR",
                "StochK",
                "StochD",
                "VWAP",
                "BullPower",
                "BearPower",
                "ROC",
                "CCI",
                "Williams%R",
                "UltimateOscillator",
                "Volume",
                "VolumeSpike"
            };
        }
    }
}