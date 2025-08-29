using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using Quantra.Models;

namespace Quantra.Controls.Components
{
    /// <summary>
    /// View component for visualizing sector-specific sentiment analysis data
    /// </summary>
    public partial class SectorSentimentVisualizationView : UserControl, INotifyPropertyChanged
    {
        // INotifyPropertyChanged implementation
        public event PropertyChangedEventHandler PropertyChanged;
        
        // Events
        public event EventHandler<string> SectorSelectionChanged;
        
        // Properties for data binding
        public ChartValues<double> SentimentValues { get; set; } = new ChartValues<double>();
        public ChartValues<double> HistoricalSentimentValues { get; set; } = new ChartValues<double>();
        public List<string> SectorLabels { get; set; } = new List<string>();
        public List<string> DateLabels { get; set; } = new List<string>();
        
        // Selected sector information
        private string _selectedSector;
        public string SelectedSector
        {
            get => _selectedSector;
            set
            {
                if (_selectedSector != value)
                {
                    _selectedSector = value;
                    OnPropertyChanged(nameof(SelectedSector));
                    SectorSelectionChanged?.Invoke(this, value);
                }
            }
        }
        
        // Sentiment comparison mode
        private string _comparisonMode = "All Sectors";
        public string ComparisonMode
        {
            get => _comparisonMode;
            set
            {
                if (_comparisonMode != value)
                {
                    _comparisonMode = value;
                    OnPropertyChanged(nameof(ComparisonMode));
                }
            }
        }
        
        // Constructor
        public SectorSentimentVisualizationView()
        {
            InitializeComponent();
            DataContext = this;
            
            // Initialize empty charts
            InitializeCharts();
            
            // Initialize sector selection
            SelectedSector = "Technology"; // Default sector
            
            // Initialize comparison modes
            ComparisonModeComboBox.Items.Add("All Sectors");
            ComparisonModeComboBox.Items.Add("Top 5 Sectors");
            ComparisonModeComboBox.Items.Add("Bottom 5 Sectors");
            ComparisonModeComboBox.Items.Add("Technology vs. Financial");
            ComparisonModeComboBox.SelectedIndex = 0;
        }
        
        // Initialize chart settings
        private void InitializeCharts()
        {
            // Configure sector comparison chart
            SectorComparisonChart.Series = new SeriesCollection
            {
                new ColumnSeries
                {
                    Title = "Sector Sentiment",
                    Values = SentimentValues,
                    DataLabels = true,
                    LabelPoint = point => $"{point.Y:F2}",
                    Fill = new SolidColorBrush(Color.FromRgb(0, 120, 215))
                }
            };
            
            // Configure historical sentiment chart
            HistoricalSentimentChart.Series = new SeriesCollection
            {
                new LineSeries
                {
                    Title = "Historical Sentiment",
                    Values = HistoricalSentimentValues,
                    PointGeometry = DefaultGeometries.Circle,
                    PointGeometrySize = 8
                }
            };
        }
        
        // Public methods
        
        /// <summary>
        /// Updates the sector comparison sentiment chart
        /// </summary>
        /// <param name="sectorSentiments">Dictionary of sector sentiment scores</param>
        public void UpdateSectorComparison(Dictionary<string, double> sectorSentiments)
        {
            SentimentValues.Clear();
            SectorLabels.Clear();
            
            // Apply filters based on comparison mode
            var filteredSentiments = FilterSectorsByComparisonMode(sectorSentiments);
            
            // Populate chart data
            foreach (var kvp in filteredSentiments)
            {
                SentimentValues.Add(kvp.Value);
                SectorLabels.Add(kvp.Key);
            }
            
            // Update X-axis labels
            ((SectorComparisonChart.AxisX.First() as LiveCharts.Wpf.Axis)).Labels = SectorLabels;
            
            // Format Y-axis
            ((SectorComparisonChart.AxisY.First() as LiveCharts.Wpf.Axis)).MinValue = -1;
            ((SectorComparisonChart.AxisY.First() as LiveCharts.Wpf.Axis)).MaxValue = 1;
            
            SectorComparisonChart.Update(true);
        }
        
        /// <summary>
        /// Updates the historical sentiment trend chart for the selected sector
        /// </summary>
        /// <param name="sentimentTrend">List of date/sentiment pairs</param>
        public void UpdateSentimentTrend(List<(DateTime Date, double Sentiment)> sentimentTrend)
        {
            HistoricalSentimentValues.Clear();
            DateLabels.Clear();
            
            foreach (var point in sentimentTrend)
            {
                HistoricalSentimentValues.Add(point.Sentiment);
                DateLabels.Add(point.Date.ToString("MM/dd"));
            }
            
            // Update X-axis labels (show a subset for readability)
            var labelStep = Math.Max(1, DateLabels.Count / 10); // Show ~10 labels
            var filteredLabels = new List<string>();
            
            for (int i = 0; i < DateLabels.Count; i++)
            {
                filteredLabels.Add(i % labelStep == 0 ? DateLabels[i] : "");
            }
            
            ((HistoricalSentimentChart.AxisX.First() as LiveCharts.Wpf.Axis)).Labels = filteredLabels;
            
            // Format Y-axis
            ((HistoricalSentimentChart.AxisY.First() as LiveCharts.Wpf.Axis)).MinValue = -1;
            ((HistoricalSentimentChart.AxisY.First() as LiveCharts.Wpf.Axis)).MaxValue = 1;
            
            HistoricalSentimentChart.Update(true);
        }
        
        /// <summary>
        /// Updates the sentiment source breakdown for a sector
        /// </summary>
        /// <param name="sourceBreakdown">Dictionary mapping sources to sentiment scores</param>
        public void UpdateSourceBreakdown(Dictionary<string, double> sourceBreakdown)
        {
            // Clear existing items
            SourceBreakdownList.Items.Clear();
            
            // Add sources sorted by sentiment magnitude (absolute value)
            foreach (var source in sourceBreakdown.OrderByDescending(s => Math.Abs(s.Value)))
            {
                string sentimentText = FormatSentimentScore(source.Value);
                SourceBreakdownList.Items.Add($"{source.Key}: {sentimentText}");
            }
        }
        
        /// <summary>
        /// Updates the sector news articles list
        /// </summary>
        /// <param name="articles">List of news articles</param>
        public void UpdateSectorNewsArticles(List<NewsArticle> articles)
        {
            // Clear existing items
            NewsArticlesList.Items.Clear();
            
            // Add articles sorted by date (most recent first)
            foreach (var article in articles.OrderByDescending(a => a.PublishedAt))
            {
                // Create article item with sentiment indicator
                var articleItem = new ListBoxItem();
                var panel = new StackPanel { Orientation = Orientation.Vertical };
                
                // Add title with sentiment color
                var titleBlock = new TextBlock
                {
                    Text = article.Title,
                    FontWeight = FontWeights.Bold,
                    TextWrapping = TextWrapping.Wrap
                };
                
                // Color based on sentiment
                if (article.SentimentScore >= 0.2)
                    titleBlock.Foreground = Brushes.Green;
                else if (article.SentimentScore <= -0.2)
                    titleBlock.Foreground = Brushes.Red;
                
                panel.Children.Add(titleBlock);
                
                // Add source and date
                panel.Children.Add(new TextBlock 
                { 
                    Text = $"{article.SourceName} - {article.PublishedAt.ToString("g")}", 
                    FontStyle = FontStyles.Italic 
                });
                
                // Add sentiment and relevance information
                panel.Children.Add(new TextBlock 
                { 
                    Text = $"Sentiment: {FormatSentimentScore(article.SentimentScore)} | " + 
                           $"Sector Relevance: {article.SectorRelevance:P0}" 
                });
                
                articleItem.Content = panel;
                NewsArticlesList.Items.Add(articleItem);
            }
        }
        
        /// <summary>
        /// Updates the trending sector topics
        /// </summary>
        /// <param name="topics">List of trending topics</param>
        public void UpdateTrendingTopics(List<string> topics)
        {
            TrendingTopicsList.Items.Clear();
            
            foreach (var topic in topics)
            {
                TrendingTopicsList.Items.Add(topic);
            }
        }
        
        // Event handlers
        
        private void SectorComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (SectorComboBox.SelectedItem is ComboBoxItem selectedItem)
            {
                SelectedSector = selectedItem.Content.ToString();
            }
        }
        
        private void ComparisonModeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ComparisonModeComboBox.SelectedItem is string selectedMode)
            {
                ComparisonMode = selectedMode;
            }
        }
        
        // Helper methods
        
        private Dictionary<string, double> FilterSectorsByComparisonMode(Dictionary<string, double> allSectors)
        {
            switch (ComparisonMode)
            {
                case "Top 5 Sectors":
                    return allSectors
                        .OrderByDescending(s => s.Value)
                        .Take(5)
                        .ToDictionary(k => k.Key, v => v.Value);
                
                case "Bottom 5 Sectors":
                    return allSectors
                        .OrderBy(s => s.Value)
                        .Take(5)
                        .ToDictionary(k => k.Key, v => v.Value);
                
                case "Technology vs. Financial":
                    return allSectors
                        .Where(s => s.Key == "Technology" || s.Key == "Financial")
                        .ToDictionary(k => k.Key, v => v.Value);
                
                case "All Sectors":
                default:
                    return allSectors;
            }
        }
        
        private string FormatSentimentScore(double score)
        {
            string sentiment;
            
            if (score >= 0.6) sentiment = "Very Positive";
            else if (score >= 0.2) sentiment = "Positive";
            else if (score >= -0.2) sentiment = "Neutral";
            else if (score >= -0.6) sentiment = "Negative";
            else sentiment = "Very Negative";
            
            return $"{sentiment} ({score:F2})";
        }
        
        // INotifyPropertyChanged implementation
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}