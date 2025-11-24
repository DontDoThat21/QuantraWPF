using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.CrossCutting.ErrorHandling;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to detect and monitor volume spikes in trading data.
    /// 
    /// This service analyzes trading volume patterns to detect abnormal increases
    /// that may indicate potential breakout opportunities or significant market events.
    /// Volume spikes are measured as either:
    /// - Volume Ratio: Current volume divided by the average volume over recent periods
    /// - Volume Z-Score: Number of standard deviations from the mean volume
    /// 
    /// Usage:
    /// 1. In AlertsControl, create a volume alert with `VolumeSpike` indicator
    /// 2. Set a threshold (default is 2.0x normal volume)
    /// 3. When the volume ratio exceeds the threshold, the alert triggers
    /// 
    /// This alert type helps traders identify unusual trading activity that
    /// may precede significant price movements.
    /// </summary>
    public class VolumeAlertService
    {
        private readonly IHistoricalDataService _historicalDataService;
        private readonly ITechnicalIndicatorService _technicalIndicatorService;

        // Constants for volume spike detection
        private const double DEFAULT_VOLUME_RATIO_THRESHOLD = 2.0; // 2x the average volume is considered a spike
        private const double DEFAULT_VOLUME_ZSCORE_THRESHOLD = 2.5; // 2.5 standard deviations from mean

        public VolumeAlertService(IHistoricalDataService historicalDataService, ITechnicalIndicatorService technicalIndicatorService)
        {
            _historicalDataService = historicalDataService;
            _technicalIndicatorService = technicalIndicatorService;
        }

        /// <summary>
        /// Checks if a volume spike alert condition is met
        /// </summary>
        /// <param name="alert">The alert to check</param>
        /// <returns>True if the alert condition is met, false otherwise</returns>
        public async Task<bool> CheckAlertConditionAsync(AlertModel alert)
        {
            if (alert == null || !alert.IsActive || alert.IsTriggered || alert.Category != AlertCategory.TechnicalIndicator)
                return false;

            if (alert.IndicatorName != "Volume" && alert.IndicatorName != "VolumeSpike")
                return false;

            try
            {
                // Get the current volume ratio or z-score
                var indicators = await _technicalIndicatorService.CalculateIndicators(alert.Symbol, "1day");

                double indicatorValue = 0;
                if (alert.IndicatorName == "Volume")
                {
                    if (indicators.TryGetValue("Volume", out double volume))
                    {
                        indicatorValue = volume;
                    }
                }
                else if (alert.IndicatorName == "VolumeSpike")
                {
                    // Try to get volume ratio first, fall back to z-score if not available
                    if (indicators.TryGetValue("volume_ratio", out double volumeRatio))
                    {
                        indicatorValue = volumeRatio;
                    }
                    else if (indicators.TryGetValue("volume_z_score", out double volumeZScore))
                    {
                        indicatorValue = volumeZScore;
                    }
                    else
                    {
                        // Calculate volume ratio manually if not available in indicators
                        indicatorValue = await CalculateVolumeRatioAsync(alert.Symbol);
                    }
                }

                alert.CurrentIndicatorValue = indicatorValue;

                // Check if the condition is met
                bool conditionMet = EvaluateCondition(indicatorValue, alert.ComparisonOperator, alert.ThresholdValue);

                if (conditionMet && !alert.IsTriggered)
                {
                    // Trigger the alert
                    alert.IsTriggered = true;
                    alert.TriggeredDate = DateTime.Now;

                    // Add details about the volume spike to the notes
                    if (string.IsNullOrEmpty(alert.Notes))
                    {
                        alert.Notes = $"Volume spike detected: {indicatorValue:F2}x normal volume";
                    }
                    else
                    {
                        alert.Notes += $"\nVolume spike detected: {indicatorValue:F2}x normal volume";
                    }

                    // Format the condition description for better readability
                    alert.Condition = $"{alert.IndicatorName} {TechnicalIndicatorAlertService.GetOperatorDisplayString(alert.ComparisonOperator)} {alert.ThresholdValue:F2}";

                    return true;
                }

                return false;
            }
            catch (Exception ex)
            {
                // Use centralized error handling; avoids UI project dependency
                ErrorHandlingManager.Instance.HandleException(ex, "VolumeAlertService.CheckAlertConditionAsync");
                return false;
            }
        }

        /// <summary>
        /// Check all active volume alerts and trigger those with conditions met
        /// </summary>
        /// <param name="alerts">List of alerts to check</param>
        /// <returns>Number of alerts triggered</returns>
        public async Task<int> CheckAllVolumeAlertsAsync(List<AlertModel> alerts)
        {
            if (alerts == null || alerts.Count == 0)
                return 0;

            int triggeredCount = 0;

            // Prepare eligible alerts
            var eligibleAlerts = alerts.Where(alert =>
                alert.Category == AlertCategory.TechnicalIndicator &&
                (alert.IndicatorName == "Volume" || alert.IndicatorName == "VolumeSpike") &&
                alert.IsActive && !alert.IsTriggered).ToList();

            if (eligibleAlerts.Count == 0)
                return 0;

            // Throttle concurrent checks to avoid resource contention
            using var semaphore = new SemaphoreSlim(Math.Min(eligibleAlerts.Count, 4));
            var tasks = new List<Task>();

            foreach (var alert in eligibleAlerts)
            {
                tasks.Add(ProcessAlertAsync(alert));
            }

            await Task.WhenAll(tasks);
            return triggeredCount;

            async Task ProcessAlertAsync(AlertModel alert)
            {
                await semaphore.WaitAsync();
                try
                {
                    if (await CheckAlertConditionAsync(alert))
                    {
                        Interlocked.Increment(ref triggeredCount);
                    }
                }
                finally
                {
                    semaphore.Release();
                }
            }
        }

        /// <summary>
        /// Calculates volume ratio (current volume / average volume) for a symbol
        /// </summary>
        /// <param name="symbol">The symbol to check</param>
        /// <returns>The volume ratio</returns>
        private async Task<double> CalculateVolumeRatioAsync(string symbol)
        {
            try
            {
                // Get historical data for the past 20 days
                var historicalData = await _historicalDataService.GetHistoricalPrices(symbol, "1mo", "1d");

                if (historicalData == null || historicalData.Count < 2)
                {
                    return 1.0; // Default if not enough data
                }

                // Calculate average volume (excluding today)
                double sumVolume = 0;
                int count = 0;

                // Skip the most recent day (today/yesterday depending on time)
                for (int i = 1; i < Math.Min(21, historicalData.Count); i++)
                {
                    sumVolume += historicalData[i].Volume;
                    count++;
                }

                if (count == 0 || sumVolume == 0)
                {
                    return 1.0; // Default if no volume data
                }

                double avgVolume = sumVolume / count;
                double currentVolume = historicalData.First().Volume;

                return currentVolume / avgVolume;
            }
            catch (Exception ex)
            {
                // Use centralized error handling; avoids UI project dependency
                ErrorHandlingManager.Instance.HandleException(ex, "VolumeAlertService.CalculateVolumeRatioAsync");
                return 1.0; // Default value indicating no spike
            }
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
                ComparisonOperator.CrossesAbove => currentValue > thresholdValue, // Simplified
                ComparisonOperator.CrossesBelow => currentValue < thresholdValue, // Simplified
                _ => false
            };
        }

        /// <summary>
        /// Gets the recommended threshold for volume spike detection
        /// </summary>
        public static double GetRecommendedVolumeRatioThreshold()
        {
            return DEFAULT_VOLUME_RATIO_THRESHOLD;
        }

        /// <summary>
        /// Gets the recommended threshold for volume z-score detection
        /// </summary>
        public static double GetRecommendedVolumeZScoreThreshold()
        {
            return DEFAULT_VOLUME_ZSCORE_THRESHOLD;
        }
    }
}