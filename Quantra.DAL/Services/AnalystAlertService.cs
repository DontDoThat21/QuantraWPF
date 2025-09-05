using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.Utilities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing and tracking analyst rating alerts
    /// </summary>
    public class AnalystAlertService
    {
        private readonly IAnalystRatingService _analystRatingService;
        private readonly Dictionary<string, List<AlertModel>> _recentAlerts = new Dictionary<string, List<AlertModel>>();
        private readonly int _alertsHistoryLimit = 100; // Number of alerts to keep in memory per symbol
        
        public AnalystAlertService(IAnalystRatingService analystRatingService)
        {
            _analystRatingService = analystRatingService ?? throw new ArgumentNullException(nameof(analystRatingService));
        }
        
        /// <summary>
        /// Creates and emits an analyst rating alert with enhanced context
        /// </summary>
        public void EmitAnalystRatingAlert(AlertModel alert, AnalystRatingAggregate consensusData = null)
        {
            try
            {
                // Add information about current consensus if available
                if (consensusData != null)
                {
                    string consensusInfo = $"Current consensus: {consensusData.ConsensusRating} " +
                                          $"({consensusData.BuyCount} Buy, {consensusData.HoldCount} Hold, {consensusData.SellCount} Sell)\n" +
                                          $"Consensus trend: {consensusData.ConsensusTrend}";
                                          
                    alert.Notes = string.IsNullOrEmpty(alert.Notes) ? 
                        consensusInfo : 
                        $"{alert.Notes}\n\n{consensusInfo}";
                }
                
                // Store alert in memory for tracking patterns
                StoreAlert(alert);

                // Emit the global alert
                Alerting.EmitGlobalAlert(alert);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to emit analyst rating alert for {alert.Symbol}", ex.ToString());
            }
        }
        
        /// <summary>
        /// Stores an alert in memory for pattern recognition
        /// </summary>
        private void StoreAlert(AlertModel alert)
        {
            if (string.IsNullOrEmpty(alert.Symbol))
                return;
                
            string symbol = alert.Symbol.ToUpper();
            
            // Initialize list if needed
            if (!_recentAlerts.ContainsKey(symbol))
                _recentAlerts[symbol] = new List<AlertModel>();
                
            // Add alert to the list
            _recentAlerts[symbol].Add(alert);
            
            // Trim list if it gets too large
            if (_recentAlerts[symbol].Count > _alertsHistoryLimit)
                _recentAlerts[symbol] = _recentAlerts[symbol].OrderByDescending(a => a.CreatedDate).Take(_alertsHistoryLimit).ToList();
        }
        
        /// <summary>
        /// Gets recent alerts for a symbol
        /// </summary>
        public List<AlertModel> GetRecentAlerts(string symbol, AlertCategory? category = null, int maxCount = 20)
        {
            if (string.IsNullOrEmpty(symbol))
                return new List<AlertModel>();
                
            symbol = symbol.ToUpper();
            
            if (!_recentAlerts.ContainsKey(symbol))
                return new List<AlertModel>();
                
            var alerts = _recentAlerts[symbol];
            
            // Filter by category if specified
            if (category.HasValue)
                alerts = alerts.Where(a => a.Category == category.Value).ToList();
                
            // Return most recent alerts
            return alerts.OrderByDescending(a => a.CreatedDate).Take(maxCount).ToList();
        }
        
        /// <summary>
        /// Checks for and emits alerts about rating trends for a symbol
        /// </summary>
        public async Task CheckRatingTrendsAsync(string symbol)
        {
            try
            {
                // Get historical consensus data
                var historyData = await _analystRatingService.GetConsensusHistoryAsync(symbol, 30);
                
                if (historyData.Count < 2)
                    return; // Need at least 2 data points for trend analysis
                    
                // Order by date
                var orderedHistory = historyData.OrderBy(h => h.LastUpdated).ToList();
                
                // Check for consistent trend over multiple data points
                int upgradeCount = 0;
                int downgradeCount = 0;
                
                for (int i = 1; i < orderedHistory.Count; i++)
                {
                    var current = orderedHistory[i];
                    var previous = orderedHistory[i-1];
                    
                    if (current.ConsensusScore > previous.ConsensusScore + 0.05)
                        upgradeCount++;
                    else if (current.ConsensusScore < previous.ConsensusScore - 0.05)
                        downgradeCount++;
                }
                
                // Only emit alerts for significant trends (3+ points in same direction)
                if (upgradeCount >= 3 && upgradeCount > downgradeCount * 2)
                {
                    EmitTrendAlert(symbol, true, upgradeCount, orderedHistory);
                }
                else if (downgradeCount >= 3 && downgradeCount > upgradeCount * 2)
                {
                    EmitTrendAlert(symbol, false, downgradeCount, orderedHistory);
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to check rating trends for {symbol}", ex.ToString());
            }
        }
        
        /// <summary>
        /// Emits an alert about a consistent consensus trend
        /// </summary>
        private void EmitTrendAlert(string symbol, bool isPositive, int dataPoints, List<AnalystRatingAggregate> history)
        {
            var latest = history.Last();
            var oldest = history.First();
            
            string trendName = isPositive ? "improving" : "deteriorating";
            string alertMessage = $"Analyst sentiment for {symbol} has been consistently {trendName} ({dataPoints} consecutive shifts)";
            
            var alert = new AlertModel
            {
                Name = alertMessage,
                Symbol = symbol,
                Condition = $"Consistent {trendName.ToUpper()} trend",
                AlertType = "Analyst Trend",
                IsActive = true,
                Priority = isPositive ? 1 : 2,
                Category = isPositive ? AlertCategory.Opportunity : AlertCategory.Standard,
                CreatedDate = DateTime.Now,
                Notes = $"Consensus shifted from {oldest.ConsensusRating} ({oldest.ConsensusScore:F2}) to {latest.ConsensusRating} ({latest.ConsensusScore:F2})\n" +
                        $"Current breakdown: {latest.BuyCount} Buy, {latest.HoldCount} Hold, {latest.SellCount} Sell\n" +
                        $"Trend started on {oldest.LastUpdated.ToShortDateString()}"
            };
            
            // Emit global alert
            Alerting.EmitGlobalAlert(alert);
        }
    }
}