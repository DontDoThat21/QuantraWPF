using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using LiveCharts;
using LiveCharts.Wpf;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.Services.Interfaces;

namespace Quantra.Controls.Components
{
    /// <summary>
    /// Interaction logic for IndicatorCorrelationView.xaml
    /// </summary>
    public partial class IndicatorCorrelationView : UserControl, INotifyPropertyChanged
    {
        // Events
        public event PropertyChangedEventHandler PropertyChanged;
        public event EventHandler CloseRequested;

        // Services
        private readonly ITechnicalIndicatorService _indicatorService;
        
        // Data
        private List<IndicatorCorrelationResult> _correlationResults;
        private List<IndicatorConfirmationPattern> _confirmationPatterns;
        
        // Properties
        private string _symbol = "AAPL";
        public string Symbol
        {
            get { return _symbol; }
            set
            {
                if (_symbol != value)
                {
                    _symbol = value;
                    OnPropertyChanged(nameof(Symbol));
                    SymbolTextBlock.Text = $"Symbol: {_symbol}";
                    RefreshDataAsync();
                }
            }
        }
        
        private string _timeframe = "1day";
        public string Timeframe
        {
            get { return _timeframe; }
            set
            {
                if (_timeframe != value)
                {
                    _timeframe = value;
                    OnPropertyChanged(nameof(Timeframe));
                    RefreshDataAsync();
                }
            }
        }

        // Constructor
        public IndicatorCorrelationView()
        {
            InitializeComponent();
            
            // Get indicator service
            _indicatorService = ServiceLocator.GetService<ITechnicalIndicatorService>();
            
            // Set data context
            DataContext = this;
            
            // Initialize data
            _correlationResults = new List<IndicatorCorrelationResult>();
            _confirmationPatterns = new List<IndicatorConfirmationPattern>();
            
            // Load data asynchronously
            Loaded += (s, e) => RefreshDataAsync();
        }

        // Async methods for loading data
        private async Task RefreshDataAsync()
        {
            try
            {
                StatusTextBlock.Text = "Loading data...";
                
                // Load correlation data
                _correlationResults = await _indicatorService.CalculateAllIndicatorCorrelations(Symbol, Timeframe);
                CorrelationsListView.ItemsSource = _correlationResults;
                
                // Load confirmation patterns
                _confirmationPatterns = await _indicatorService.FindConfirmationPatterns(Symbol, Timeframe);
                PatternsListView.ItemsSource = _confirmationPatterns;
                
                // Update heatmap
                UpdateHeatmap();
                
                // Update status
                StatusTextBlock.Text = $"Loaded {_correlationResults.Count} correlations and {_confirmationPatterns.Count} patterns.";
            }
            catch (Exception ex)
            {
                StatusTextBlock.Text = $"Error: {ex.Message}";
                MessageBox.Show($"Failed to load correlation data: {ex.Message}", "Error", 
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        
        private void UpdateHeatmap()
        {
            // Clear existing heatmap
            HeatmapCanvas.Children.Clear();
            
            // Get unique indicators from correlation results
            var uniqueIndicators = new HashSet<string>();
            foreach (var corr in _correlationResults)
            {
                uniqueIndicators.Add(corr.FirstIndicator);
                uniqueIndicators.Add(corr.SecondIndicator);
            }
            
            // Sort indicators
            var indicators = uniqueIndicators.OrderBy(i => i).ToList();
            int count = indicators.Count;
            
            if (count == 0)
                return;
            
            // Calculate cell size
            double cellSize = Math.Min(
                (HeatmapCanvas.ActualWidth - 150) / count,
                (HeatmapCanvas.ActualHeight - 50) / count);
            
            if (cellSize < 5)
                cellSize = 5;
            
            // Start drawing position
            double startX = 150;
            double startY = 50;
            
            // Draw row headers (indicator names)
            for (int i = 0; i < count; i++)
            {
                var txtBlock = new TextBlock
                {
                    Text = indicators[i],
                    FontSize = 10,
                    Foreground = Brushes.White,
                    TextAlignment = TextAlignment.Right,
                    Width = 140,
                    Height = cellSize
                };
                Canvas.SetLeft(txtBlock, 5);
                Canvas.SetTop(txtBlock, startY + i * cellSize + cellSize / 2 - 10);
                HeatmapCanvas.Children.Add(txtBlock);
            }
            
            // Draw column headers (indicator names)
            for (int i = 0; i < count; i++)
            {
                var txtBlock = new TextBlock
                {
                    Text = indicators[i],
                    FontSize = 10,
                    Foreground = Brushes.White,
                    TextAlignment = TextAlignment.Left,
                    LayoutTransform = new RotateTransform(-90),
                    Width = 140,
                    Height = cellSize
                };
                Canvas.SetLeft(txtBlock, startX + i * cellSize - txtBlock.ActualHeight / 2);
                Canvas.SetTop(txtBlock, 5);
                HeatmapCanvas.Children.Add(txtBlock);
            }
            
            // Draw correlation cells
            for (int i = 0; i < count; i++)
            {
                for (int j = 0; j < count; j++)
                {
                    // Find correlation coefficient (if exists)
                    double correlation = 0;
                    
                    if (i == j)
                    {
                        // Diagonal is always 1.0 (perfect correlation with self)
                        correlation = 1.0;
                    }
                    else
                    {
                        var corrResult = _correlationResults.FirstOrDefault(c =>
                            (c.FirstIndicator == indicators[i] && c.SecondIndicator == indicators[j]) ||
                            (c.FirstIndicator == indicators[j] && c.SecondIndicator == indicators[i]));
                            
                        if (corrResult != null)
                        {
                            correlation = corrResult.CorrelationCoefficient;
                        }
                    }
                    
                    // Calculate color from correlation (-1 to +1)
                    var color = GetColorFromCorrelation(correlation);
                    
                    // Create rectangle for cell
                    var rect = new Rectangle
                    {
                        Width = cellSize,
                        Height = cellSize,
                        Fill = new SolidColorBrush(color),
                        Stroke = Brushes.Gray,
                        StrokeThickness = 0.5,
                        ToolTip = $"{indicators[i]} vs {indicators[j]}: {correlation:F2}"
                    };
                    
                    Canvas.SetLeft(rect, startX + j * cellSize);
                    Canvas.SetTop(rect, startY + i * cellSize);
                    HeatmapCanvas.Children.Add(rect);
                    
                    // Add correlation value text for significant correlations
                    if (Math.Abs(correlation) > 0.5)
                    {
                        var txtBlock = new TextBlock
                        {
                            Text = $"{correlation:F2}",
                            FontSize = 9,
                            Foreground = Math.Abs(correlation) > 0.8 ? Brushes.White : Brushes.Black,
                            TextAlignment = TextAlignment.Center,
                            Width = cellSize,
                            Height = cellSize
                        };
                        Canvas.SetLeft(txtBlock, startX + j * cellSize);
                        Canvas.SetTop(txtBlock, startY + i * cellSize + cellSize / 2 - 7);
                        HeatmapCanvas.Children.Add(txtBlock);
                    }
                }
            }
        }
        
        private Color GetColorFromCorrelation(double correlation)
        {
            // Map correlation coefficient to color:
            // -1.0 (negative correlation) = Red
            //  0.0 (no correlation) = White
            // +1.0 (positive correlation) = Green
            
            byte r, g, b;
            
            if (correlation >= 0)
            {
                // Positive correlation: white to green
                double factor = correlation;
                r = (byte)(255 * (1 - factor));
                g = 255;
                b = (byte)(255 * (1 - factor));
            }
            else
            {
                // Negative correlation: white to red
                double factor = -correlation;
                r = 255;
                g = (byte)(255 * (1 - factor));
                b = (byte)(255 * (1 - factor));
            }
            
            return Color.FromRgb(r, g, b);
        }

        // Event handlers
        private void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            RefreshDataAsync();
        }
        
        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            CloseRequested?.Invoke(this, EventArgs.Empty);
        }
        
        private void PatternsListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var selectedPattern = PatternsListView.SelectedItem as IndicatorConfirmationPattern;
            if (selectedPattern == null)
                return;
                
            // Show pattern details in a more detailed view or popup
            StatusTextBlock.Text = $"Selected pattern: {selectedPattern.GetPatternDescription()}";
            
            // Show the correlation chart for this pattern
            UpdateCorrelationChartForPattern(selectedPattern);
        }
        
        private void CorrelationsListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var selectedCorrelation = CorrelationsListView.SelectedItem as IndicatorCorrelationResult;
            if (selectedCorrelation == null)
                return;
                
            // Show correlation details
            StatusTextBlock.Text = $"Selected correlation: {selectedCorrelation.FirstIndicator} vs {selectedCorrelation.SecondIndicator} = {selectedCorrelation.CorrelationCoefficient:F2}";
            
            // Update chart
            UpdateCorrelationChart(selectedCorrelation);
        }
        
        private void UpdateCorrelationChart(IndicatorCorrelationResult correlation)
        {
            if (correlation == null || 
                correlation.FirstIndicatorValues == null || 
                correlation.SecondIndicatorValues == null)
                return;
                
            // Clear existing series
            CorrelationChart.Series.Clear();
            
            // Create series for first indicator
            var firstSeries = new LineSeries
            {
                Title = correlation.FirstIndicator,
                Values = new ChartValues<double>(correlation.FirstIndicatorValues),
                PointGeometry = DefaultGeometries.Circle,
                PointGeometrySize = 6,
                LineSmoothness = 0.5,
                Stroke = Brushes.DodgerBlue,
                Fill = Brushes.Transparent
            };
            
            // Create series for second indicator (scale to match first)
            var secondValues = correlation.SecondIndicatorValues.Select(
                val => NormalizeValue(val, correlation.SecondIndicatorValues, correlation.FirstIndicatorValues)).ToList();
                
            var secondSeries = new LineSeries
            {
                Title = correlation.SecondIndicator,
                Values = new ChartValues<double>(secondValues),
                PointGeometry = DefaultGeometries.Square,
                PointGeometrySize = 6,
                LineSmoothness = 0.5,
                Stroke = Brushes.OrangeRed,
                Fill = Brushes.Transparent
            };
            
            // Add series to chart
            CorrelationChart.Series.Add(firstSeries);
            CorrelationChart.Series.Add(secondSeries);
            
            // Update title
            //CorrelationChart.Title = $"{correlation.FirstIndicator} vs {correlation.SecondIndicator} (r={correlation.CorrelationCoefficient:F2})";
        }
        
        private void UpdateCorrelationChartForPattern(IndicatorConfirmationPattern pattern)
        {
            if (pattern == null || pattern.SupportingCorrelations.Count == 0)
                return;
                
            // Use the first supporting correlation
            UpdateCorrelationChart(pattern.SupportingCorrelations[0]);
        }
        
        // Helper methods
        private double NormalizeValue(double value, List<double> sourceList, List<double> targetList)
        {
            if (sourceList.Count == 0 || targetList.Count == 0)
                return value;
                
            // Get min/max of both lists
            double sourceMin = sourceList.Min();
            double sourceMax = sourceList.Max();
            double targetMin = targetList.Min();
            double targetMax = targetList.Max();
            
            // Avoid division by zero
            if (sourceMax - sourceMin == 0)
                return targetMin;
                
            // Scale the value from source range to target range
            return targetMin + (value - sourceMin) * (targetMax - targetMin) / (sourceMax - sourceMin);
        }
        
        // INotifyPropertyChanged implementation
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}