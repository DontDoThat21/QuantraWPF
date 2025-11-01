using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Quantra.Models;
using LiveCharts;
using System.Drawing;

namespace Quantra.Controls.Components
{
    public partial class PredictionDetailView : UserControl, INotifyPropertyChanged, IDisposable
    {
        // Events
        public event PropertyChangedEventHandler PropertyChanged;

        // Chart data collections for binding
        public ChartValues<double> PriceValues { get; set; }
        public ChartValues<double> VwapValues { get; set; } 
        public ChartValues<double> PredictionValues { get; set; }
        public Func<double, string> DateFormatter { get; set; }

        // Constants for visualization
        private const int MaxFeaturesToDisplay = 5;
        private const double FeatureBarHeight = 20;
        
        // Track if disposed
        private bool _disposed = false;

        // Constructor
        public PredictionDetailView()
        {
            InitializeComponent();

            // Initialize chart data
            PriceValues = new ChartValues<double>();
            VwapValues = new ChartValues<double>();
            PredictionValues = new ChartValues<double>();
            
            // Set date formatter
            DateFormatter = value => new DateTime((long)value).ToString("MM/dd");
            
            // Set data context
            this.DataContext = this;
            
            // Initialize with empty state
            ResetView();
            
            // Register for unloaded event to ensure cleanup
            Unloaded += PredictionDetailView_Unloaded;
        }

        private void PredictionDetailView_Unloaded(object sender, RoutedEventArgs e)
        {
            // Clean up when control is unloaded
            Dispose();
        }

        // Public methods
        public void UpdatePrediction(PredictionModel prediction)
        {
            if (_disposed) return;
            
            if (prediction == null)
            {
                ResetView();
                return;
            }

            // Update header information
            SelectedSymbolTitle.Text = $"{prediction.Symbol} - {prediction.PredictedAction}";
            
            string summaryText = $"Prediction generated on {DateTime.Now:g}. " +
                $"Confidence: {prediction.Confidence:P0}. " +
                $"Current price: ${prediction.CurrentPrice:F2}, Target: ${prediction.TargetPrice:F2} " +
                $"({prediction.PotentialReturn:P2})";
            PredictionSummaryText.Text = summaryText;

            // Update RSI indicators if available
            UpdateRsiDisplay(prediction);
            
            // Update volume analysis
            UpdateVolumeDisplay(prediction);

            // Update ML model confidence and analysis
            UpdateMLModelAnalysis(prediction);

            // Generate sample chart data if needed
            List<double> prices = GenerateSamplePrices(prediction);
            List<double> vwap = GenerateSampleVwap(prediction);
            List<double> predictionLine = GenerateSamplePredictionLine(prediction);

            // Update chart data
            UpdateChartData(prices, vwap, predictionLine);
        }
        
        // Helper methods
        private void ResetView()
        {
            if (_disposed) return;
            
            // Reset text
            SelectedSymbolTitle.Text = "Select a symbol from the list";
            PredictionSummaryText.Text = "";
            
            // Reset indicators
            RsiValueText.Text = "RSI: --";
            RsiTrendText.Text = "Trend: --";
            RsiSignalText.Text = "Signal: --";
            
            VolumeText.Text = "Volume: --";
            VwapText.Text = "VWAP: --";
            VolumeTrendText.Text = "Trend: --";
            
            // Reset ML model analysis
            ConfidenceFill.Width = 0;
            ConfidenceText.Text = "--";
            ModelTypeText.Text = "Model Type: --";
            InferenceTimeText.Text = "Inference Time: --";
            PredictionQualityText.Text = "Quality Score: --";
            RiskScoreText.Text = "Risk Score: --";
            ValueAtRiskText.Text = "Value at Risk: --";
            SharpeRatioText.Text = "Sharpe Ratio: --";
            
            // Reset ML performance metrics
            PredictionAccuracyText.Text = "Historical Accuracy: --";
            ConsecutiveCorrectText.Text = "Consecutive Correct: --";
            ConfidenceTrendText.Text = "Confidence Trend: --";
            PerformanceDashboardImage.Source = null;
            NoDashboardText.Visibility = Visibility.Visible;
            PerformanceInsightsText.Text = "Performance insights will appear here after generating a dashboard.";
            
            // Clear feature importance panel
            FeatureImportancePanel.Children.Clear();
            FeatureImportancePanel.Children.Add(NoFeaturesText);
            NoFeaturesText.Visibility = Visibility.Visible;
            
            // Clear chart data
            PriceValues.Clear();
            VwapValues.Clear();
            PredictionValues.Clear();
            
            // Update chart
            PredictionChart.Update();
        }

        private void UpdateRsiDisplay(PredictionModel prediction)
        {
            if (_disposed) return;
            
            if (prediction.Indicators?.ContainsKey("RSI") == true)
            {
                double rsi = prediction.Indicators["RSI"];
                RsiValueText.Text = $"RSI: {rsi:F2}";

                string trend = rsi > 50 ? "Bullish" : "Bearish";
                RsiTrendText.Text = $"Trend: {trend}";

                string signal = "Neutral";
                if (rsi > 70) 
                {
                    signal = "Overbought";
                    RsiSignalText.Foreground = System.Windows.Media.Brushes.Red;
                }
                else if (rsi < 30) 
                {
                    signal = "Oversold";
                    RsiSignalText.Foreground = System.Windows.Media.Brushes.Green;
                }
                else
                {
                    RsiSignalText.Foreground = System.Windows.Media.Brushes.White;
                }
                RsiSignalText.Text = $"Signal: {signal}";
            }
        }

        private void UpdateVolumeDisplay(PredictionModel prediction)
        {
            if (_disposed) return;
            
            if (prediction.Indicators?.ContainsKey("Volume") == true)
            {
                double volume = prediction.Indicators["Volume"];
                VolumeText.Text = $"Volume: {volume:N0}";
            }
            
            if (prediction.Indicators?.ContainsKey("VWAP") == true)
            {
                double vwap = prediction.Indicators["VWAP"];
                VwapText.Text = $"VWAP: ${vwap:F2}";
                
                string volumeTrend = prediction.CurrentPrice > vwap ? "Above VWAP (Bullish)" : "Below VWAP (Bearish)";
                VolumeTrendText.Text = $"Trend: {volumeTrend}";
                
                if (prediction.CurrentPrice > vwap)
                {
                    VolumeTrendText.Foreground = System.Windows.Media.Brushes.LightGreen;
                }
                else
                {
                    VolumeTrendText.Foreground = System.Windows.Media.Brushes.LightCoral;
                }
            }
        }

        private void UpdateMLModelAnalysis(PredictionModel prediction)
        {
            if (_disposed) return;
            
            // Update Confidence Meter
            UpdateConfidenceMeter(prediction.Confidence);
            
            // Update model info
            ModelTypeText.Text = $"Model Type: {(string.IsNullOrEmpty(prediction.ModelType) ? "Ensemble" : prediction.ModelType)}";
            InferenceTimeText.Text = $"Inference Time: {prediction.InferenceTimeMs:F1}ms";
            
            // Quality score calculation (using extension method if available or using confidence as fallback)
            double qualityScore = prediction.PredictionAccuracy > 0 
                ? prediction.PredictionAccuracy 
                : (prediction.ConsecutiveCorrectPredictions > 0 ? 0.5 + (prediction.ConsecutiveCorrectPredictions * 0.1) : prediction.Confidence);
            
            PredictionQualityText.Text = $"Quality Score: {qualityScore:P0}";
            
            // Update risk metrics
            UpdateRiskMetrics(prediction);
            
            // Update feature importance visualization
            UpdateFeatureImportance(prediction);
            
            // Update ML performance metrics
            UpdatePerformanceMetrics(prediction);
            
            // Store the current prediction for size change updates
            _currentPrediction = prediction;
            
            // Register confidence background size change handler
            if (!_confidenceSizeChangedHandlerRegistered)
            {
                ConfidenceBackground.SizeChanged += ConfidenceBackground_SizeChanged;
                _confidenceSizeChangedHandlerRegistered = true;
            }
        }
        
        private PredictionModel _currentPrediction;
        private bool _confidenceSizeChangedHandlerRegistered = false;
        
        private void ConfidenceBackground_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (_disposed || _currentPrediction == null) return;
            
            UpdateConfidenceMeter(_currentPrediction.Confidence);
        }
        
        private void UpdateConfidenceMeter(double confidence)
        {
            if (_disposed) return;
            
            // Get current background width to calculate the fill width
            double backgroundWidth = ConfidenceBackground.ActualWidth;
            double confidenceWidth = backgroundWidth * confidence;
            
            // Ensure minimum visual width for very low confidence values
            ConfidenceFill.Width = confidenceWidth > 0 ? Math.Max(confidenceWidth, 3) : 0;
            ConfidenceText.Text = $"{confidence:P0}";
            
            // Set confidence color based on level
            if (confidence >= 0.75)
                ConfidenceFill.Fill = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(32, 192, 64)); // Green
            else if (confidence >= 0.5)
                ConfidenceFill.Fill = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(192, 192, 32)); // Yellow
            else
                ConfidenceFill.Fill = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(192, 32, 32)); // Red
        }
        
        private void UpdateRiskMetrics(PredictionModel prediction)
        {
            if (_disposed) return;
            
            // Risk Score
            double riskScore = prediction.RiskScore;
            RiskScoreText.Text = $"Risk Score: {riskScore:P0}";
            
            // Set risk score color
            if (riskScore < 0.3)
                RiskScoreText.Foreground = System.Windows.Media.Brushes.LightGreen;
            else if (riskScore < 0.7)
                RiskScoreText.Foreground = System.Windows.Media.Brushes.Yellow;
            else
                RiskScoreText.Foreground = System.Windows.Media.Brushes.Red;
            
            // Value at Risk (VAR)
            ValueAtRiskText.Text = $"Value at Risk: ${prediction.ValueAtRisk:F2}";
            
            // Sharpe Ratio
            double sharpe = prediction.SharpeRatio;
            SharpeRatioText.Text = $"Sharpe Ratio: {sharpe:F2}";
            
            // Set Sharpe ratio color
            if (sharpe >= 1.0)
                SharpeRatioText.Foreground = System.Windows.Media.Brushes.LightGreen;
            else if (sharpe >= 0)
                SharpeRatioText.Foreground = System.Windows.Media.Brushes.Yellow;
            else
                SharpeRatioText.Foreground = System.Windows.Media.Brushes.Red;
        }
        
        private void UpdateFeatureImportance(PredictionModel prediction)
        {
            if (_disposed) return;
            
            // Clear previous feature visualization
            FeatureImportancePanel.Children.Clear();
            
            // Get feature weights from prediction
            var featureWeights = prediction.FeatureWeights;
            
            // If no features, show the "no features" message
            if (featureWeights == null || featureWeights.Count == 0)
            {
                NoFeaturesText.Visibility = Visibility.Visible;
                FeatureImportancePanel.Children.Add(NoFeaturesText);
                return;
            }
            
            NoFeaturesText.Visibility = Visibility.Collapsed;
            
            // Get top features by importance
            var orderedFeatures = featureWeights
                .OrderByDescending(kv => Math.Abs(kv.Value))
                .Take(MaxFeaturesToDisplay)
                .ToList();
            
            // Find max absolute weight for scaling
            double maxAbsWeight = orderedFeatures.Max(kv => Math.Abs(kv.Value));
            if (maxAbsWeight == 0) maxAbsWeight = 1.0; // Prevent division by zero
            
            // Create feature bars
            foreach (var feature in orderedFeatures)
            {
                FeatureImportancePanel.Children.Add(CreateFeatureImportanceBar(
                    feature.Key, 
                    feature.Value, 
                    maxAbsWeight,
                    FeatureImportancePanel.ActualWidth));
            }
            
            // Register to size changed event to update bar widths when container resizes
            if (!_sizeChangedHandlerRegistered)
            {
                FeatureImportancePanel.SizeChanged += FeatureImportancePanel_SizeChanged;
                _sizeChangedHandlerRegistered = true;
            }
            
            // Store current feature data for resize updates
            _currentFeatureWeights = orderedFeatures;
            _currentMaxWeight = maxAbsWeight;
        }
        
        private bool _sizeChangedHandlerRegistered = false;
        private List<KeyValuePair<string, double>> _currentFeatureWeights;
        private double _currentMaxWeight = 1.0;
        
        private void FeatureImportancePanel_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (_disposed || _currentFeatureWeights == null || _currentFeatureWeights.Count == 0) return;
            
            // Recreate all feature bars with new size
            FeatureImportancePanel.Children.Clear();
            
            foreach (var feature in _currentFeatureWeights)
            {
                FeatureImportancePanel.Children.Add(CreateFeatureImportanceBar(
                    feature.Key,
                    feature.Value,
                    _currentMaxWeight,
                    e.NewSize.Width));
            }
        }
        
        private UIElement CreateFeatureImportanceBar(string featureName, double value, double maxValue, double containerWidth)
        {
            if (_disposed) return null;
            
            // Create container for this feature
            Grid featureGrid = new Grid 
            { 
                Margin = new Thickness(0, 3, 0, 3),
                Height = FeatureBarHeight
            };
            
            featureGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(0.4, GridUnitType.Star) });
            featureGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(0.6, GridUnitType.Star) });
            
            // Create feature name label
            TextBlock nameBlock = new TextBlock
            {
                Text = FormatFeatureName(featureName),
                Foreground = System.Windows.Media.Brushes.White,
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 8, 0),
                TextTrimming = TextTrimming.CharacterEllipsis
            };
            Grid.SetColumn(nameBlock, 0);
            
            // Add tooltip with full feature name and description
            nameBlock.ToolTip = CreateFeatureDescriptionTooltip(featureName, value);
            
            // Create container for bar and value
            Grid barGrid = new Grid();
            Grid.SetColumn(barGrid, 1);
            
            // Calculate bar width
            double maxBarWidth = containerWidth * 0.5; // Max width is 50% of panel width
            double normalizedValue = Math.Abs(value) / maxValue;
            double barWidth = Math.Max(normalizedValue * maxBarWidth, 5); // Min width of 5px
            
            // Create feature value bar
            Border bar = new Border
            {
                Background = value > 0 
                    ? new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(32, 192, 64)) // Green for positive
                    : new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(192, 32, 32)), // Red for negative
                Width = barWidth,
                Height = FeatureBarHeight - 6,
                HorizontalAlignment = HorizontalAlignment.Left,
                CornerRadius = new CornerRadius(2)
            };
            
            // Create feature value label
            TextBlock valueBlock = new TextBlock
            {
                Text = value.ToString("F2"),
                Foreground = System.Windows.Media.Brushes.White,
                VerticalAlignment = VerticalAlignment.Center,
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(0, 0, 5, 0)
            };
            
            // Add elements to layout
            barGrid.Children.Add(bar);
            barGrid.Children.Add(valueBlock);
            featureGrid.Children.Add(nameBlock);
            featureGrid.Children.Add(barGrid);
            
            return featureGrid;
        }
        
        private object CreateFeatureDescriptionTooltip(string featureName, double value)
        {
            ToolTip tooltip = new ToolTip();
            StackPanel panel = new StackPanel { Width = 200 };
            
            // Feature name as header
            panel.Children.Add(new TextBlock 
            { 
                Text = GetFullFeatureName(featureName), 
                FontWeight = FontWeights.Bold,
                TextWrapping = TextWrapping.Wrap
            });
            
            // Feature description
            panel.Children.Add(new TextBlock 
            { 
                Text = GetFeatureDescription(featureName), 
                Margin = new Thickness(0, 5, 0, 0),
                TextWrapping = TextWrapping.Wrap
            });
            
            // Influence description
            string influenceText = value > 0 
                ? $"Positive influence: Supports bullish prediction with strength of {Math.Abs(value):F2}"
                : $"Negative influence: Supports bearish prediction with strength of {Math.Abs(value):F2}";
            
            panel.Children.Add(new TextBlock 
            { 
                Text = influenceText, 
                Margin = new Thickness(0, 5, 0, 0),
                TextWrapping = TextWrapping.Wrap,
                Foreground = value > 0 ? System.Windows.Media.Brushes.LightGreen : System.Windows.Media.Brushes.LightCoral
            });
            
            tooltip.Content = panel;
            return tooltip;
        }
        
        // Get full feature name from abbreviation or code
        private string GetFullFeatureName(string featureName)
        {
            switch (featureName.ToLower())
            {
                case "rsi": return "Relative Strength Index";
                case "macd": return "Moving Average Convergence Divergence";
                case "adx": return "Average Directional Index";
                case "cci": return "Commodity Channel Index";
                case "vwap": return "Volume Weighted Average Price";
                case "ema": return "Exponential Moving Average";
                case "sma": return "Simple Moving Average";
                case "bbands": return "Bollinger Bands";
                case "obv": return "On-Balance Volume";
                case "vol": return "Volume";
                case "volat": return "Volatility";
                case "atr": return "Average True Range";
                case "mom": return "Momentum";
                case "roc": return "Rate of Change";
                case "stoch": return "Stochastic Oscillator";
                case "sentmt": return "Market Sentiment";
                case "vix": return "Volatility Index";
                case "sp500": return "S&P 500 Correlation";
                case "trend": return "Price Trend";
                // Add more mappings as needed
                default: return featureName; // Return original if no mapping exists
            }
        }
        
        // Get feature description
        private string GetFeatureDescription(string featureName)
        {
            switch (featureName.ToLower())
            {
                case "rsi": return "Measures the speed and magnitude of price movements. Values above 70 indicate overbought conditions, below 30 indicate oversold conditions.";
                case "macd": return "Trend-following momentum indicator showing the relationship between two moving averages of a security's price.";
                case "adx": return "Indicates the strength of a trend (not direction). Higher values indicate stronger trends.";
                case "cci": return "Measures a security's variation from its statistical mean. Used to identify cyclical trends.";
                case "vwap": return "Average price weighted by volume. Important support/resistance level for intraday trading.";
                case "ema": return "Moving average giving more weight to recent prices. Responds more quickly to price changes than SMA.";
                case "sma": return "Arithmetic moving average calculated by adding recent prices and dividing by the number of time periods.";
                case "bbands": return "Shows volatility and relative price levels over a period. Consists of upper, middle, and lower bands.";
                case "obv": return "Cumulative indicator that relates volume to price change, showing buying/selling pressure.";
                case "vol": return "Trading volume, indicating the liquidity and activity level of the security.";
                case "volat": return "Measure of a security's price dispersion, indicating risk or uncertainty.";
                case "atr": return "Measures market volatility by decomposing the entire range of an asset price for a period.";
                case "mom": return "Rate of change of a security's price over a fixed time period.";
                case "roc": return "Percentage change in price over a fixed time period.";
                case "stoch": return "Compares a security's closing price to its price range over a given time period.";
                case "sentmt": return "Market sentiment indicators from news, social media, or analyst opinions.";
                case "vix": return "Market's expectation of 30-day forward-looking volatility, derived from S&P 500 index options.";
                case "sp500": return "Correlation with the broader market, as represented by the S&P 500 index.";
                case "trend": return "Direction of the overall price movement over the analyzed time period.";
                // Add more descriptions as needed
                default: return "Technical indicator used in market analysis."; // Generic description
            }
        }

        // Format feature name for display by shortening and capitalizing
        private string FormatFeatureName(string featureName)
        {
            if (string.IsNullOrEmpty(featureName))
                return "Unknown";
            
            // Make first letter uppercase
            string formattedName = char.ToUpperInvariant(featureName[0]) + 
                (featureName.Length > 1 ? featureName.Substring(1) : "");
            
            // Shorten if too long
            if (formattedName.Length > 12)
                return formattedName.Substring(0, 10) + "...";
                
            return formattedName;
        }

        private void UpdateChartData(List<double> prices, List<double> vwap, List<double> predictionLine)
        {
            if (_disposed) return;
            
            // Clear old data
            PriceValues.Clear();
            VwapValues.Clear(); 
            PredictionValues.Clear();
            
            // Add new data
            if (prices != null) PriceValues.AddRange(prices);
            if (vwap != null) VwapValues.AddRange(vwap);
            if (predictionLine != null) PredictionValues.AddRange(predictionLine);
            
            // Update the chart
            PredictionChart.Update(true);
        }
        
        // Sample data generators for demo purposes - in a real app, these would use actual historical data
        private List<double> GenerateSamplePrices(PredictionModel prediction)
        {
            // Simple demo data
            List<double> prices = new List<double>();
            double basePrice = prediction.CurrentPrice * 0.95;
            
            // Generate 20 days of price data
            Random random = new Random((int)(prediction.Symbol.GetHashCode() + DateTime.Now.Ticks % 10000));
            
            for (int i = 0; i < 20; i++)
            {
                double change = (random.NextDouble() - 0.5) * 0.02;
                basePrice *= (1 + change);
                prices.Add(basePrice);
            }
            
            // Last price should be the current price
            prices[prices.Count - 1] = prediction.CurrentPrice;
            
            return prices;
        }
        
        private List<double> GenerateSampleVwap(PredictionModel prediction)
        {
            // Generate sample VWAP data
            List<double> vwap = new List<double>();
            double basePrice = prediction.CurrentPrice * 0.98;
            
            Random random = new Random((int)(prediction.Symbol.GetHashCode() + DateTime.Now.Ticks % 20000));
            
            for (int i = 0; i < 20; i++)
            {
                double change = (random.NextDouble() - 0.5) * 0.01;
                basePrice *= (1 + change);
                vwap.Add(basePrice);
            }
            
            return vwap;
        }
        
        // ML Performance Visualization Methods
        private void UpdatePerformanceMetrics(PredictionModel prediction)
        {
            if (_disposed) return;
            
            // Update the accuracy text
            if (prediction.PredictionAccuracy > 0)
            {
                PredictionAccuracyText.Text = $"Historical Accuracy: {prediction.PredictionAccuracy:P0}";
                
                // Set color based on accuracy
                if (prediction.PredictionAccuracy >= 0.7)
                    PredictionAccuracyText.Foreground = System.Windows.Media.Brushes.LightGreen;
                else if (prediction.PredictionAccuracy >= 0.5)
                    PredictionAccuracyText.Foreground = System.Windows.Media.Brushes.Yellow;
                else
                    PredictionAccuracyText.Foreground = System.Windows.Media.Brushes.Red;
            }
            else
            {
                PredictionAccuracyText.Text = "Historical Accuracy: N/A";
                PredictionAccuracyText.Foreground = System.Windows.Media.Brushes.White;
            }
            
            // Update consecutive correct predictions
            if (prediction.ConsecutiveCorrectPredictions > 0)
            {
                ConsecutiveCorrectText.Text = $"Consecutive Correct: {prediction.ConsecutiveCorrectPredictions}";
                
                // Set color based on streak
                if (prediction.ConsecutiveCorrectPredictions >= 3)
                    ConsecutiveCorrectText.Foreground = System.Windows.Media.Brushes.LightGreen;
                else
                    ConsecutiveCorrectText.Foreground = System.Windows.Media.Brushes.White;
            }
            else
            {
                ConsecutiveCorrectText.Text = "Consecutive Correct: 0";
                ConsecutiveCorrectText.Foreground = System.Windows.Media.Brushes.White;
            }
            
            // Determine confidence trend (this would typically come from historical data)
            string confidenceTrend = "Stable";
            if (prediction.Confidence > 0.8)
                confidenceTrend = "Increasing";
            else if (prediction.Confidence < 0.4)
                confidenceTrend = "Decreasing";
            
            // Update confidence trend text
            ConfidenceTrendText.Text = $"Confidence Trend: {confidenceTrend}";
            
            // Set color based on trend
            if (confidenceTrend == "Increasing")
                ConfidenceTrendText.Foreground = System.Windows.Media.Brushes.LightGreen;
            else if (confidenceTrend == "Decreasing")
                ConfidenceTrendText.Foreground = System.Windows.Media.Brushes.Red;
            else
                ConfidenceTrendText.Foreground = System.Windows.Media.Brushes.White;
            
            // Reset dashboard status
            NoDashboardText.Visibility = Visibility.Visible;
            PerformanceDashboardImage.Source = null;
            PerformanceInsightsText.Text = "Click 'Generate Dashboard' to view detailed model performance insights.";
        }
        
        private async void GenerateDashboardButton_Click(object sender, RoutedEventArgs e)
        {
            if (_disposed || _currentPrediction == null) return;
            
            try
            {
                // Show loading indicator
                DashboardLoadingIndicator.Visibility = Visibility.Visible;
                NoDashboardText.Visibility = Visibility.Collapsed;
                PerformanceInsightsText.Text = "Generating performance dashboard...";
                
                // Generate the dashboard (simulate with a delay for now)
                string dashboardPath = await GeneratePerformanceDashboard(_currentPrediction);
                
                if (!string.IsNullOrEmpty(dashboardPath) && File.Exists(dashboardPath))
                {
                    // Load and display the image
                    var bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.UriSource = new Uri(dashboardPath);
                    bitmap.EndInit();
                    
                    PerformanceDashboardImage.Source = bitmap;
                    NoDashboardText.Visibility = Visibility.Collapsed;
                    
                    // Update insights text based on the model
                    UpdatePerformanceInsights(_currentPrediction);
                }
                else
                {
                    // Dashboard generation failed
                    NoDashboardText.Text = "Failed to generate dashboard. Please try again.";
                    NoDashboardText.Visibility = Visibility.Visible;
                    PerformanceInsightsText.Text = "Dashboard generation failed. Check logs for details.";
                }
            }
            catch (Exception ex)
            {
                // Handle errors
                NoDashboardText.Text = "Error generating dashboard.";
                NoDashboardText.Visibility = Visibility.Visible;
                PerformanceInsightsText.Text = "Error: " + ex.Message;
                //DatabaseMonolith.Log("Error", "Failed to generate ML performance dashboard", ex.ToString());
            }
            finally
            {
                // Hide loading indicator
                DashboardLoadingIndicator.Visibility = Visibility.Collapsed;
            }
        }
        
        private async Task<string> GeneratePerformanceDashboard(PredictionModel prediction)
        {
            // This method would call into Python to generate a dashboard visualization
            // For now, we'll simulate this with a delay and sample data
            
            // In a real implementation, this would call into a Python service
            await Task.Delay(1000); // Simulate processing time
            
            try
            {
                string modelType = string.IsNullOrEmpty(prediction.ModelType) ? "ensemble" : prediction.ModelType.ToLower();
                string outputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "temp");
                Directory.CreateDirectory(outputDir);
                
                string outputPath = Path.Combine(outputDir, $"model_dashboard_{prediction.Symbol}_{Guid.NewGuid().ToString().Substring(0, 8)}.png");
                
                // Call Python script to generate dashboard
                var python = new PythonModelIntegration();
                bool success = await python.GenerateModelDashboard(
                    modelType,
                    prediction.Symbol,
                    prediction.FeatureWeights,
                    outputPath
                );
                
                return success ? outputPath : null;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to generate performance dashboard", ex.ToString());
                return null;
            }
        }
        
        private void UpdatePerformanceInsights(PredictionModel prediction)
        {
            StringBuilder insights = new StringBuilder();
            
            // Generate insights based on the model data
            insights.AppendLine("Performance Insights:");
            
            if (prediction.PredictionAccuracy > 0)
            {
                insights.AppendLine($"• Model accuracy is {prediction.PredictionAccuracy:P0}, " +
                                  $"which is {(prediction.PredictionAccuracy >= 0.7 ? "strong" : 
                                             prediction.PredictionAccuracy >= 0.5 ? "moderate" : "weak")}.");
            }
            
            // Add confidence analysis
            insights.AppendLine($"• Current prediction confidence is {prediction.Confidence:P0}, " +
                             $"indicating {(prediction.Confidence >= 0.75 ? "high" : 
                                          prediction.Confidence >= 0.5 ? "moderate" : "low")} certainty.");
            
            // Add risk analysis
            insights.AppendLine($"• Risk assessment: {(prediction.RiskScore <= 0.3 ? "Low" : 
                                                    prediction.RiskScore <= 0.6 ? "Moderate" : "High")} " +
                             $"risk with Sharpe ratio of {prediction.SharpeRatio:F2}.");
            
            // Add feature influence summary if available
            if (prediction.FeatureWeights?.Count > 0)
            {
                var topFeature = prediction.FeatureWeights
                    .OrderByDescending(kv => Math.Abs(kv.Value))
                    .FirstOrDefault();
                
                if (!string.IsNullOrEmpty(topFeature.Key))
                {
                    insights.AppendLine($"• {GetFullFeatureName(topFeature.Key)} is the strongest factor " +
                                     $"in this prediction ({topFeature.Value:F2}).");
                }
            }
            
            // Add confidence trend analysis
            string trendDirection = prediction.Confidence >= 0.7 ? "upward" : 
                                  prediction.Confidence <= 0.4 ? "downward" : "stable";
            
            insights.AppendLine($"• Model confidence shows a {trendDirection} trend over recent predictions.");
            
            // Set the insights text
            PerformanceInsightsText.Text = insights.ToString();
        }
        
        private List<double> GenerateSamplePredictionLine(PredictionModel prediction)
        {
            // Generate future prediction line
            List<double> futurePrediction = new List<double>();
            
            // Fill with nulls for past data (to align with other series)
            for (int i = 0; i < 19; i++)
            {
                futurePrediction.Add(double.NaN);
            }
            
            // Current price and future predictions
            futurePrediction.Add(prediction.CurrentPrice);
            futurePrediction.Add(prediction.TargetPrice);
            
            return futurePrediction;
        }

        // INotifyPropertyChanged implementation
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        
        // IDisposable implementation
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Unregister event handlers
                    if (_sizeChangedHandlerRegistered)
                        FeatureImportancePanel.SizeChanged -= FeatureImportancePanel_SizeChanged;
                        
                    if (_confidenceSizeChangedHandlerRegistered)
                        ConfidenceBackground.SizeChanged -= ConfidenceBackground_SizeChanged;
                        
                    Unloaded -= PredictionDetailView_Unloaded;
                    
                    // Clear references
                    _currentPrediction = null;
                    _currentFeatureWeights = null;
                }
                
                _disposed = true;
            }
        }
        
        ~PredictionDetailView()
        {
            Dispose(false);
        }
    }

    // Python integration helper class
    internal class PythonModelIntegration
    {
        public async Task<bool> GenerateModelDashboard(
            string modelType, 
            string symbol, 
            Dictionary<string, double> featureWeights, 
            string outputPath)
        {
            // In a production app, this would use proper Python interop
            // For this demo, we'll simulate success
            
            await Task.Delay(1500); // Simulate Python execution time
            
            try
            {
                // Create a simple dashboard with System.Drawing
                using (var bitmap = new System.Drawing.Bitmap(800, 600))
                using (var graphics = System.Drawing.Graphics.FromImage(bitmap))
                {
                    // Fill background
                    graphics.FillRectangle(System.Drawing.Brushes.MidnightBlue, 0, 0, 800, 600);
                    
                    // Add title
                    using (var font = new System.Drawing.Font("Arial", 24, System.Drawing.FontStyle.Bold))
                    {
                        graphics.DrawString($"{symbol}: ML Model Performance Dashboard", 
                                          font, System.Drawing.Brushes.White, 20, 20);
                    }
                    
                    // Add model type
                    using (var font = new System.Drawing.Font("Arial", 16))
                    {
                        graphics.DrawString($"Model Type: {modelType}", 
                                          font, System.Drawing.Brushes.LightGreen, 20, 70);
                    }
                    
                    // Add timestamp
                    using (var font = new System.Drawing.Font("Arial", 12))
                    {
                        graphics.DrawString($"Generated: {DateTime.Now}", 
                                          font, System.Drawing.Brushes.LightGray, 20, 100);
                    }
                    
                    // Draw some placeholder charts
                    // Confidence trend chart
                    graphics.DrawRectangle(System.Drawing.Pens.Gray, 50, 150, 700, 150);
                    graphics.DrawString("Confidence Trend (Last 30 Days)", 
                                      new System.Drawing.Font("Arial", 14), 
                                      System.Drawing.Brushes.White, 60, 160);
                    
                    // Draw sample confidence line
                    var points = new System.Drawing.Point[10];
                    for (int i = 0; i < 10; i++)
                    {
                        points[i] = new System.Drawing.Point(
                            100 + i * 60,
                            250 - (int)(new Random(i + symbol.GetHashCode()).NextDouble() * 80)
                        );
                    }
                    graphics.DrawLines(System.Drawing.Pens.Cyan, points);
                    
                    // Feature importance chart
                    graphics.DrawRectangle(System.Drawing.Pens.Gray, 50, 330, 700, 200);
                    graphics.DrawString("Feature Importance", 
                                      new System.Drawing.Font("Arial", 14), 
                                      System.Drawing.Brushes.White, 60, 340);
                    
                    // Draw sample feature bars if we have feature weights
                    if (featureWeights != null && featureWeights.Count > 0)
                    {
                        var topFeatures = featureWeights
                            .OrderByDescending(kv => Math.Abs(kv.Value))
                            .Take(5)
                            .ToList();
                        
                        for (int i = 0; i < topFeatures.Count; i++)
                        {
                            var feature = topFeatures[i];
                            var barWidth = (int)(Math.Abs(feature.Value) * 500);
                            var brush = feature.Value >= 0 ? 
                                System.Drawing.Brushes.LimeGreen : 
                                System.Drawing.Brushes.Crimson;
                            
                            graphics.FillRectangle(brush, 150, 380 + i * 30, barWidth, 20);
                            graphics.DrawString(feature.Key, 
                                              new System.Drawing.Font("Arial", 10), 
                                              System.Drawing.Brushes.White, 60, 380 + i * 30);
                        }
                    }
                    
                    // Save the bitmap as a file
                    bitmap.Save(outputPath, System.Drawing.Imaging.ImageFormat.Png);
                }

                return true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to generate model dashboard image", ex.ToString());
                return false;
            }
        }
    }
}

