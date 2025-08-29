using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using Quantra.Modules; // Add this for SentimentPriceVisualData, SentimentShiftEvent

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for SentimentVisualizationControl.xaml
    /// </summary>
    public partial class SentimentVisualizationControl : UserControl
    {
        public SeriesCollection SentimentSeries { get; private set; }
        public SeriesCollection CorrelationSeries { get; private set; }
        public List<string> SourceLabels { get; private set; }
        
        public SentimentVisualizationControl()
        {
            InitializeComponent();
            
            // Initialize chart collections
            SentimentSeries = new SeriesCollection();
            CorrelationSeries = new SeriesCollection();
            SourceLabels = new List<string>();
            
            // Set chart series
            SentimentTrendChart.Series = SentimentSeries;
            SourceCorrelationChart.Series = CorrelationSeries;
            
            // Set data context for binding
            SourceCorrelationChartAxisX.DataContext = this;
        }

        /// <summary>
        /// Updates the visualization with sentiment data
        /// </summary>
        /// <param name="data">The sentiment visualization data</param>
        public void UpdateVisualization(SentimentPriceVisualData data)
        {
            if (data == null)
                return;

            try
            {
                // Clear existing chart data
                SentimentSeries.Clear();
                CorrelationSeries.Clear();
                SourceLabels.Clear();

                // Update metrics displays
                PredictiveAccuracyTextBlock.Text = $"{data.PredictiveAccuracy:P0}";
                LeadLagTextBlock.Text = $"{data.LeadLagRelationship:F1} days";
                
                // Format LeadLag text with descriptive text
                if (data.LeadLagRelationship > 0)
                {
                    LeadLagTextBlock.Text += " (leads price)";
                    LeadLagTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
                }
                else if (data.LeadLagRelationship < 0)
                {
                    LeadLagTextBlock.Text += " (lags price)";
                    LeadLagTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
                }
                else
                {
                    LeadLagTextBlock.Text += " (neutral)";
                    LeadLagTextBlock.Foreground = new SolidColorBrush(Colors.Cyan);
                }

                // Format PredictiveAccuracy text color based on value
                if (data.PredictiveAccuracy >= 0.7)
                {
                    PredictiveAccuracyTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
                }
                else if (data.PredictiveAccuracy <= 0.4)
                {
                    PredictiveAccuracyTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
                }
                else
                {
                    PredictiveAccuracyTextBlock.Foreground = new SolidColorBrush(Colors.Cyan);
                }
                
                // Update date labels for sentiment trend chart
                var dateLabels = data.Dates.Select(d => d.ToString("MM/dd")).ToList();
                SentimentTrendChartAxisX.Labels = dateLabels;
                
                // Add combined sentiment series
                if (data.CombinedSentiment != null && data.CombinedSentiment.Count > 0)
                {
                    SentimentSeries.Add(new LineSeries
                    {
                        Title = "Combined",
                        Values = new ChartValues<double>(data.CombinedSentiment),
                        PointGeometry = DefaultGeometries.None,
                        Stroke = new SolidColorBrush(Colors.White),
                        StrokeThickness = 3,
                        LineSmoothness = 0.3
                    });
                }
                
                // Add source-specific sentiment series
                if (data.SentimentBySource != null)
                {
                    // Define consistent colors for each source
                    var sourceColors = new Dictionary<string, Color>
                    {
                        { "Twitter", Colors.DeepSkyBlue },
                        { "Reddit", Colors.Orange },
                        { "News", Colors.Yellow },
                        { "AnalystRatings", Colors.LimeGreen },
                        { "InsiderTrading", Colors.MediumPurple }
                    };

                    foreach (var source in data.SentimentBySource.Keys)
                    {
                        var values = data.SentimentBySource[source];
                        if (values.Count > 0)
                        {
                            // Get color or use a default color
                            var color = sourceColors.ContainsKey(source) ? sourceColors[source] : Colors.Gray;
                            
                            SentimentSeries.Add(new LineSeries
                            {
                                Title = source,
                                Values = new ChartValues<double>(values),
                                PointGeometry = null,
                                Stroke = new SolidColorBrush(color),
                                StrokeThickness = 1.5,
                                LineSmoothness = 0.3
                            });
                        }
                    }
                }
                
                // Add price correlation series (bar chart)
                if (data.SourceCorrelations != null && data.SourceCorrelations.Count > 0)
                {
                    // Get source labels for the X axis
                    SourceLabels.AddRange(data.SourceCorrelations.Keys);
                    
                    // Create a column series for correlations
                    var correlationValues = new ChartValues<double>();
                    foreach (var source in SourceLabels)
                    {
                        correlationValues.Add(data.SourceCorrelations[source]);
                    }
                    
                    CorrelationSeries.Add(new ColumnSeries
                    {
                        Title = "Correlation",
                        Values = correlationValues,
                        Fill = new SolidColorBrush(Color.FromRgb(0, 180, 200)), // Cyan-ish
                        DataLabels = true,
                        LabelPoint = point => $"{point.Y:F2}"
                    });
                }
                
                // Update sentiment shift events
                UpdateSentimentShiftEvents(data.SentimentShiftEvents);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to update sentiment visualization", ex.ToString());
            }
        }
        
        /// <summary>
        /// Updates the sentiment shift events list
        /// </summary>
        private void UpdateSentimentShiftEvents(List<SentimentShiftEvent> events)
        {
            SentimentShiftEventsListView.Items.Clear();
            
            if (events == null || events.Count == 0)
                return;
                
            // Sort events by date (descending) and take the 5 most recent
            var recentEvents = events
                .OrderByDescending(e => e.Date)
                .Take(5)
                .ToList();
                
            foreach (var evt in recentEvents)
            {
                SentimentShiftEventsListView.Items.Add(new
                {
                    Date = evt.Date,
                    Source = evt.Source,
                    Direction = evt.SentimentShift > 0 ? "positive" : "negative",
                    SentimentShift = evt.SentimentShift,
                    SubsequentPriceChange = evt.SubsequentPriceChange
                });
            }
        }
        
        /// <summary>
        /// Clears all visualization data
        /// </summary>
        public void ClearVisualization()
        {
            SentimentSeries.Clear();
            CorrelationSeries.Clear();
            SourceLabels.Clear();
            SentimentShiftEventsListView.Items.Clear();
            
            PredictiveAccuracyTextBlock.Text = "0.00%";
            LeadLagTextBlock.Text = "0.0 days";
        }
    }
}