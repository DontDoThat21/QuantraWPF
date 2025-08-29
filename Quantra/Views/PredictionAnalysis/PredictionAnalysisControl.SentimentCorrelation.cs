using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using Quantra.Models;
using Quantra.Modules;

namespace Quantra.Controls
{
    public partial class PredictionAnalysisControl : UserControl
    {
        // Sentiment correlation analyzer
        private SentimentPriceCorrelationAnalysis _sentimentCorrelationAnalysis;
        
        // Latest correlation analysis result
        private SentimentPriceCorrelationResult _lastSentimentCorrelation;
        
        // Sentiment visualization controls
        private SentimentVisualizationControl _sentimentVisualizationControl;
        private SentimentDashboardControl _sentimentDashboardControl;
        
        /// <summary>
        /// Initializes the sentiment correlation analysis component
        /// </summary>
        private void InitializeSentimentCorrelationAnalysis()
        {
            _sentimentCorrelationAnalysis = new SentimentPriceCorrelationAnalysis(_userSettings);
            
            // Create the sentiment visualization control
            _sentimentVisualizationControl = new SentimentVisualizationControl();
            
            // Create the sentiment dashboard control (new interactive dashboard)
            _sentimentDashboardControl = new SentimentDashboardControl();
            
            // Add them to the UI if sentiment visualization container exists
            var sentimentVisualizationContainer = this.FindName("sentimentVisualizationContainer") as Panel;
            if (sentimentVisualizationContainer != null)
            {
                // Clear any existing children
                sentimentVisualizationContainer.Children.Clear();
                
                // Add both visualization controls
                sentimentVisualizationContainer.Children.Add(_sentimentVisualizationControl);
                sentimentVisualizationContainer.Children.Add(_sentimentDashboardControl);
                
                // Hide visualization control by default (dashboard will be the main display)
                _sentimentVisualizationControl.Visibility = Visibility.Collapsed;
                _sentimentDashboardControl.Visibility = Visibility.Visible;
            }
        }
        
        /// <summary>
        /// Analyzes correlation between sentiment and price for the selected symbol
        /// </summary>
        private async Task AnalyzeSentimentPriceCorrelation(string symbol, int lookbackDays = 30)
        {
            try
            {
                // Show loading status
                UpdateStatus($"Analyzing sentiment-price correlation for {symbol}...");
                
                // Run correlation analysis
                _lastSentimentCorrelation = await _sentimentCorrelationAnalysis.AnalyzeSentimentPriceCorrelation(
                    symbol, lookbackDays);
                
                // Update UI with results
                UpdateSentimentCorrelationDisplay();
                
                UpdateStatus($"Sentiment-price correlation analysis for {symbol} complete.");
            }
            catch (Exception ex)
            {
                UpdateStatus($"Error analyzing sentiment-price correlation: {ex.Message}", isError: true);
                DatabaseMonolith.Log("Error", "Sentiment-price correlation analysis failed", ex.ToString());
            }
        }
        
        /// <summary>
        /// Toggles between simple sentiment visualization and advanced dashboard
        /// </summary>
        public void ToggleSentimentVisualizationMode(bool useDashboard = true)
        {
            if (_sentimentVisualizationControl != null && _sentimentDashboardControl != null)
            {
                _sentimentVisualizationControl.Visibility = useDashboard ? Visibility.Collapsed : Visibility.Visible;
                _sentimentDashboardControl.Visibility = useDashboard ? Visibility.Visible : Visibility.Collapsed;
            }
        }
        
        /// <summary>
        /// Updates the UI with sentiment correlation results
        /// </summary>
        private void UpdateSentimentCorrelationDisplay()
        {
            if (_lastSentimentCorrelation == null)
                return;
                
            var result = _lastSentimentCorrelation;
            
            try
            {
                // Get visualization data
                var visualData = _sentimentCorrelationAnalysis.GetVisualizationData(
                    result.Symbol, 
                    30, 
                    null).GetAwaiter().GetResult();
                
                // Update the original sentiment visualization (for legacy compatibility)
                _sentimentVisualizationControl?.UpdateVisualization(visualData);
                
                // Update the new interactive sentiment dashboard
                _sentimentDashboardControl?.UpdateDashboard(result.Symbol);
                
                // Add sentiment correlation info to indicators dictionary for prediction details
                if (indicators != null && result.SourceCorrelations.Count > 0)
                {
                    foreach (var source in result.SourceCorrelations.Keys)
                    {
                        indicators[$"{source}Correlation"] = result.SourceCorrelations[source];
                    }
                    
                    indicators["SentimentLeadLag"] = result.LeadLagRelationship;
                    indicators["SentimentPredictiveAccuracy"] = result.PredictiveAccuracy;
                    indicators["SentimentOverallCorrelation"] = result.OverallCorrelation;
                    
                    // If sentiment leads price (positive lead/lag) and has good predictive accuracy,
                    // we can adjust our confidence slightly
                    if (result.LeadLagRelationship > 0 && result.PredictiveAccuracy > 0.6)
                    {
                        confidence += 0.05;
                    }
                }
                
                // Show the sentiment visualization by making the container visible
                var sentimentTab = this.FindName("SentimentTabItem") as TabItem;
                if (sentimentTab != null)
                {
                    sentimentTab.Visibility = Visibility.Visible;
                }
                
                var sentimentPanel = this.FindName("sentimentVisualizationContainer") as Panel;
                if (sentimentPanel != null)
                {
                    sentimentPanel.Visibility = Visibility.Visible;
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to update sentiment visualization", ex.ToString());
            }
        }
        
        /// <summary>
        /// Integrates sentiment correlation data into the prediction model
        /// </summary>
        private void IntegrateSentimentCorrelationIntoPrediction(ref PredictionModel prediction)
        {
            if (_lastSentimentCorrelation == null || prediction == null)
                return;
                
            // Add correlation data to prediction details
            prediction.AnalysisDetails += $"\n\nSentiment Correlation Analysis:\n";
            prediction.AnalysisDetails += $"Overall Correlation: {_lastSentimentCorrelation.OverallCorrelation:F4}\n";
            prediction.AnalysisDetails += $"Sentiment {(_lastSentimentCorrelation.LeadLagRelationship > 0 ? "leads" : "lags")} price by {Math.Abs(_lastSentimentCorrelation.LeadLagRelationship):F1} days\n";
            prediction.AnalysisDetails += $"Predictive Accuracy: {_lastSentimentCorrelation.PredictiveAccuracy:P2}\n";
            
            // Add source-specific correlation info
            prediction.AnalysisDetails += "\nSource Correlations:\n";
            foreach (var source in _lastSentimentCorrelation.SourceCorrelations.Keys)
            {
                prediction.AnalysisDetails += $"- {source}: {_lastSentimentCorrelation.SourceCorrelations[source]:F4}\n";
            }
            
            // Add OpenAI sentiment if available
            if (_lastSentimentCorrelation.SourceCorrelations.ContainsKey("OpenAI"))
            {
                prediction.OpenAISentiment = _lastSentimentCorrelation.SourceCorrelations["OpenAI"];
                prediction.UsesOpenAI = true;
            }
            
            // Add recent sentiment shift events if any
            var recentEvents = _lastSentimentCorrelation.SentimentShiftEvents;
            if (recentEvents.Count > 0)
            {
                prediction.AnalysisDetails += "\nRecent Sentiment Shifts:\n";
                foreach (var evt in recentEvents.GetRange(0, Math.Min(3, recentEvents.Count)))
                {
                    string direction = evt.SentimentShift > 0 ? "positive" : "negative";
                    prediction.AnalysisDetails += $"- {evt.Date:MM/dd}: {evt.Source} shifted {direction} ({evt.SentimentShift:F2}), price changed {evt.SubsequentPriceChange:F2}%\n";
                }
            }
        }
        
        /// <summary>
        /// Updates status display with the given message
        /// </summary>
        private void UpdateStatus(string message, bool isError = false)
        {
            // Update status text if available
            var statusText = this.FindName("StatusText") as TextBlock;
            if (statusText != null)
                statusText.Text = message;
                
            // Log the message
            if (isError)
            {
                DatabaseMonolith.Log("Error", message);
            }
            else
            {
                DatabaseMonolith.Log("Info", message);
            }
        }
    }
}