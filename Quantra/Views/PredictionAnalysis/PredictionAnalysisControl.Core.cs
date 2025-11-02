using System;
using Microsoft.Extensions.Configuration;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using LiveCharts;
using LiveCharts.Defaults;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using System.Windows.Threading;
using Quantra.Adapters;
using System.Threading.Tasks;
using Quantra.DAL.Services;

namespace Quantra.Controls
{
    public partial class PredictionAnalysisControl : UserControl
    {
        private WebullTradingBot tradingBot;
        private List<PatternModel> detectedPatterns;
        private PatternModel selectedPattern;
        private Dictionary<string, double> sectorPerformance;
        private bool chartsInitialized;
        private bool isAutomatedMode;
        
        // Chart data collections for binding
        public ChartValues<double> PriceValues { get; set; }
        public ChartValues<double> VwapValues { get; set; }
        public ChartValues<double> PredictionValues { get; set; }
        public ChartValues<double> TopPerformerValues { get; set; }
        public List<string> TopPerformerLabels { get; set; }
        public ChartValues<OhlcPoint> PatternCandles { get; set; }
        public ChartValues<double> PatternHighlights { get; set; }
        public Func<double, string> DateFormatter { get; set; }

        // Additional constructor with configuration
        public PredictionAnalysisControl(IConfiguration configuration)
        {
            tradingBot = new WebullTradingBot();
            // Initialize other configuration-specific items if needed
            InitializeControl();
        }

        /// <summary>
        /// Initializes the control with proper layout settings, registers event handlers, 
        /// and sets up data binding for the prediction analysis control.
        /// </summary>
        private void InitializeControl()
        {
            // Set sizing properties programmatically to ensure they take effect
            this.HorizontalAlignment = HorizontalAlignment.Stretch;
            this.VerticalAlignment = VerticalAlignment.Stretch;
            this.MinWidth = 400;
            this.MinHeight = 300;

            // Log the control creation with dimensions
            //DatabaseMonolith.Log("Info", $"PredictionAnalysisControl created with MinSize: {this.MinWidth}x{this.MinHeight}");

            // Initialize the AutoModeToggle if it exists - remove event handler registration to avoid duplicates
            if (AutoModeToggle != null)
            {
                // Initialize with auto mode off
                isAutomatedMode = false;
                AutoModeToggle.IsChecked = false;
                //DatabaseMonolith.Log("Info", "Auto mode initialized to OFF");
            }

            // Fix: Use a dispatcher to measure after UI layout completes
            this.Loaded += (s, e) =>
            {
                // Additional initialization after loading if needed
                //DatabaseMonolith.Log("Info", "PredictionAnalysisControl layout completed");
            };

            InitializeChartData(); // Call the implementation from Charts.cs;
            
            // Initialize model collections
            detectedPatterns = new List<PatternModel>();
            sectorPerformance = new Dictionary<string, double>();
            
            // Initialize sentiment correlation analysis
            InitializeSentimentCorrelationAnalysis();

            // Set axis formatter
            DateFormatter = value => DateTime.FromOADate(value).ToString("MM/dd");

            // Register PredictionDataGrid event handler here to make it centralized
            if (PredictionDataGrid != null)
            {
                PredictionDataGrid.SelectionChanged -= PredictionDataGrid_SelectionChanged; // Remove to avoid duplicates
                PredictionDataGrid.SelectionChanged += PredictionDataGrid_SelectionChanged;
            }
        }

        // Add this method to manually measure and update the control layout
        public void ForceLayoutUpdate()
        {
            try
            {
                // Measure and arrange the control with available size
                this.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                this.Arrange(new Rect(0, 0, this.DesiredSize.Width, this.DesiredSize.Height));
                this.UpdateLayout();

                // Log the control size after manual layout
                //DatabaseMonolith.Log("Info", $"PredictionAnalysisControl after forced layout: {this.ActualWidth}x{this.ActualHeight}");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error during layout update", ex.ToString());
            }
        }

        private bool IsInTechSector(string symbol)
        {
            // Simple check for technology sector companies
            string[] techCompanies = { "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "AMD", "INTC", "ADBE", "CRM", "CSCO" };
            return techCompanies.Contains(symbol);
        }
        
        /// <summary>
        /// Programmatically selects or focuses the Top Predictions section.
        /// </summary>
        public void SelectTopPredictionsTab()
        {
            // If the PredictionDataGrid exists, set focus to it.
            if (PredictionDataGrid != null)
            {
                PredictionDataGrid.Focus();
                // Optionally, select the first row if available
                if (PredictionDataGrid.Items.Count > 0)
                {
                    PredictionDataGrid.SelectedIndex = 0;
                    var row = (DataGridRow)PredictionDataGrid.ItemContainerGenerator.ContainerFromIndex(0);
                    if (row != null)
                        row.MoveFocus(new System.Windows.Input.TraversalRequest(System.Windows.Input.FocusNavigationDirection.Next));
                }
            }
        }

        private async Task<double> GetCurrentStockPrice(string symbol)
        {
            try
            {
                // Use Alpha Vantage service to get current price using the correct method
                var alphaVantageService = new AlphaVantageService();
                // Use GetQuoteDataAsync instead of GetQuoteAsync which doesn't exist
                var quote = await alphaVantageService.GetQuoteDataAsync(symbol);
                return quote?.Price ?? 0.0;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error getting price for {symbol}", ex.ToString());
                return 0.0;
            }
        }

        private double CalculatePredictionConfidence(Dictionary<string, double> indicators)
        {
            if (indicators == null || indicators.Count == 0)
                return 0.5; // Default neutral confidence

            var confidenceFactors = new List<double>();

            // RSI confidence factor
            if (indicators.ContainsKey("RSI"))
            {
                var rsi = indicators["RSI"];
                if (rsi < 30) confidenceFactors.Add(0.8); // Oversold - bullish
                else if (rsi > 70) confidenceFactors.Add(0.8); // Overbought - bearish
                else confidenceFactors.Add(0.5); // Neutral
            }

            // MACD confidence factor
            if (indicators.ContainsKey("MACDHistogram"))
            {
                var macdHist = indicators["MACDHistogram"];
                confidenceFactors.Add(Math.Abs(macdHist) > 0.1 ? 0.7 : 0.5);
            }

            // Volume confidence factor
            if (indicators.ContainsKey("Volume"))
            {
                var volume = indicators["Volume"];
                confidenceFactors.Add(volume > 1000000 ? 0.7 : 0.6);
            }

            // Return average confidence, clamped between 0.1 and 0.9
            var avgConfidence = confidenceFactors.Count > 0 ? confidenceFactors.Average() : 0.5;
            return Math.Max(0.1, Math.Min(0.9, avgConfidence));
        }

        private string DeterminePredictedAction(Dictionary<string, double> indicators)
        {
            if (indicators == null || indicators.Count == 0)
                return "HOLD";

            var bullishSignals = 0;
            var bearishSignals = 0;

            // RSI signals
            if (indicators.ContainsKey("RSI"))
            {
                var rsi = indicators["RSI"];
                if (rsi < 30) bullishSignals++;
                else if (rsi > 70) bearishSignals++;
            }

            // MACD signals
            if (indicators.ContainsKey("MACDHistogram"))
            {
                var macdHist = indicators["MACDHistogram"];
                if (macdHist > 0) bullishSignals++;
                else if (macdHist < 0) bearishSignals++;
            }

            // Volume confirmation
            if (indicators.ContainsKey("Volume"))
            {
                var volume = indicators["Volume"];
                if (volume > 1000000) // High volume adds weight
                {
                    if (bullishSignals > bearishSignals) bullishSignals++;
                    else if (bearishSignals > bullishSignals) bearishSignals++;
                }
            }

            // Determine action based on signal strength
            if (bullishSignals > bearishSignals && bullishSignals >= 2)
                return "BUY";
            else if (bearishSignals > bullishSignals && bearishSignals >= 2)
                return "SELL";
            else
                return "HOLD";
        }
    }
}