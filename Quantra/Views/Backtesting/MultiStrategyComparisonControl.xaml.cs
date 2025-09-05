using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;

namespace Quantra.Views.Backtesting
{
    public partial class MultiStrategyComparisonControl : UserControl
    {
        private StrategyComparisonResult _comparisonResult;
        private Dictionary<string, bool> _selectedStrategies = new Dictionary<string, bool>();
        private BacktestingEngine.BacktestResult _combinedPortfolioResult;
        private Dictionary<string, double> _portfolioWeights = new Dictionary<string, double>();
        
        // Collection of available strategies with selection state for UI binding
        private ObservableCollection<StrategySelectionItem> _strategySelections = new ObservableCollection<StrategySelectionItem>();
        
        // Colors for consistent strategy display across charts
        private readonly Dictionary<string, Brush> _strategyColors = new Dictionary<string, Brush>
        {
            { "Strategy1", Brushes.DarkRed },
            { "Strategy2", Brushes.DarkBlue },
            { "Strategy3", Brushes.DarkGreen },
            { "Strategy4", Brushes.DarkOrange },
            { "Strategy5", Brushes.Purple },
            { "Strategy6", Brushes.Teal },
            { "Strategy7", Brushes.Brown },
            { "Strategy8", Brushes.Navy },
            { "Strategy9", Brushes.Maroon },
            { "Strategy10", Brushes.Olive }
        };
        
        // Constructor
        public MultiStrategyComparisonControl()
        {
            InitializeComponent();
            
            // Set formatter for drawdown charts
            DrawdownFormatter = value => (value * 100).ToString("F2") + "%";
        }
        
        // Property to hold the drawdown formatting function
        public Func<double, string> DrawdownFormatter { get; set; }
        
        // Event handler for control loaded
        private void MultiStrategyComparisonControl_Loaded(object sender, RoutedEventArgs e)
        {
            // Initial UI setup
            StrategyCheckBoxPanel.ItemsSource = _strategySelections;
        }
        
        /// <summary>
        /// Load comparison results into the control
        /// </summary>
        public void LoadComparisonResult(StrategyComparisonResult comparisonResult)
        {
            _comparisonResult = comparisonResult;
            if (comparisonResult == null)
                return;
                
            // Update header info
            SymbolText.Text = comparisonResult.Symbol;
            DateRangeText.Text = $"{comparisonResult.StartDate.ToShortDateString()} to {comparisonResult.EndDate.ToShortDateString()}";
            
            // Initialize strategy selections
            _strategySelections.Clear();
            _selectedStrategies.Clear();
            
            foreach (var strategyResult in comparisonResult.StrategyResults)
            {
                _strategySelections.Add(new StrategySelectionItem
                {
                    StrategyName = strategyResult.StrategyName,
                    IsSelected = true
                });
                
                _selectedStrategies[strategyResult.StrategyName] = true;
            }
            
            // Update UI components with results
            UpdateBestStrategyDisplay();
            UpdatePerformanceMetricsGrid();
            UpdateEquityCurvesChart();
            UpdateCorrelationMatrix();
            UpdateDrawdownComparison();
            
            // Calculate optimal portfolio
            RecalculateOptimalPortfolio();
        }
        
        /// <summary>
        /// Update the best strategy summary display
        /// </summary>
        private void UpdateBestStrategyDisplay()
        {
            if (_comparisonResult == null || !_comparisonResult.StrategyResults.Any())
                return;
                
            string bestStrategyName = _comparisonResult.GetBestOverallStrategy();
            var bestStrategy = _comparisonResult.StrategyResults
                .FirstOrDefault(s => s.StrategyName == bestStrategyName);
                
            if (bestStrategy == null)
                return;
                
            // Update best strategy summary
            BestStrategyNameText.Text = bestStrategy.StrategyName;
            BestStrategyTypeText.Text = bestStrategy.StrategyType;
            
            var result = bestStrategy.Result;
            BestStrategyReturnText.Text = result.TotalReturn.ToString("P2");
            BestStrategyCagrText.Text = result.CAGR.ToString("P2");
            BestStrategyDrawdownText.Text = result.MaxDrawdown.ToString("P2");
            BestStrategySharpeText.Text = result.SharpeRatio.ToString("F2");
            BestStrategySortinoText.Text = result.SortinoRatio.ToString("F2");
            BestStrategyWinRateText.Text = result.WinRate.ToString("P2");
            BestStrategyProfitFactorText.Text = result.ProfitFactor.ToString("F2");
        }
        
        /// <summary>
        /// Update the performance metrics grid
        /// </summary>
        private void UpdatePerformanceMetricsGrid()
        {
            if (_comparisonResult == null)
                return;
                
            var metricsRows = new ObservableCollection<PerformanceMetricsRow>();
            
            foreach (var strategyResult in _comparisonResult.StrategyResults)
            {
                var result = strategyResult.Result;
                
                metricsRows.Add(new PerformanceMetricsRow
                {
                    StrategyName = strategyResult.StrategyName,
                    TotalReturn = result.TotalReturn.ToString("P2"),
                    CAGR = result.CAGR.ToString("P2"),
                    MaxDrawdown = result.MaxDrawdown.ToString("P2"),
                    SharpeRatio = result.SharpeRatio.ToString("F2"),
                    SortinoRatio = result.SortinoRatio.ToString("F2"),
                    WinRate = result.WinRate.ToString("P2"),
                    ProfitFactor = result.ProfitFactor.ToString("F2"),
                    Consistency = strategyResult.ConsistencyScore.ToString("F2"),
                    Volatility = strategyResult.AnnualizedVolatility.ToString("P2")
                });
            }
            
            // Add combined portfolio if available
            if (_combinedPortfolioResult != null)
            {
                metricsRows.Add(new PerformanceMetricsRow
                {
                    StrategyName = "Combined Portfolio",
                    TotalReturn = _combinedPortfolioResult.TotalReturn.ToString("P2"),
                    CAGR = _combinedPortfolioResult.CAGR.ToString("P2"),
                    MaxDrawdown = _combinedPortfolioResult.MaxDrawdown.ToString("P2"),
                    SharpeRatio = _combinedPortfolioResult.SharpeRatio.ToString("F2"),
                    SortinoRatio = _combinedPortfolioResult.SortinoRatio.ToString("F2"),
                    WinRate = _combinedPortfolioResult.WinRate.ToString("P2"),
                    ProfitFactor = _combinedPortfolioResult.ProfitFactor.ToString("F2"),
                    Consistency = "N/A",
                    Volatility = "N/A"
                });
            }
            
            PerformanceMetricsGrid.ItemsSource = metricsRows;
        }
        
        /// <summary>
        /// Update the equity curves chart
        /// </summary>
        private void UpdateEquityCurvesChart()
        {
            if (_comparisonResult == null)
                return;
                
            EquityCurvesChart.Series = new SeriesCollection();
            
            int strategyIndex = 0;
            foreach (var strategyResult in _comparisonResult.StrategyResults)
            {
                // Skip if not selected
                if (!_selectedStrategies.ContainsKey(strategyResult.StrategyName) || 
                    !_selectedStrategies[strategyResult.StrategyName])
                    continue;
                    
                // Normalize equity curve based on initial capital
                double initialEquity = _comparisonResult.InitialCapital;
                var equityCurve = strategyResult.Result.EquityCurve;
                
                // Create series for this strategy
                var series = new LineSeries
                {
                    Title = strategyResult.StrategyName,
                    Values = new ChartValues<double>(equityCurve.Select(e => e.Equity)),
                    PointGeometry = null, // Faster rendering without points
                    Stroke = GetStrategyColor(strategyResult.StrategyName, strategyIndex)
                };
                
                EquityCurvesChart.Series.Add(series);
                strategyIndex++;
            }
            
            // Set the X-axis labels to dates
            if (_comparisonResult.StrategyResults.Any() && _comparisonResult.StrategyResults[0].Result.EquityCurve.Any())
            {
                var dates = _comparisonResult.StrategyResults[0].Result.EquityCurve
                    .Select(e => e.Date.ToShortDateString())
                    .ToArray();
                    
                EquityCurvesChart.AxisX[0].Labels = dates;
                
                // Set labels to show only every Nth date to prevent overcrowding
                int skipFactor = Math.Max(1, dates.Length / 15);
                EquityCurvesChart.AxisX[0].Separator.Step = skipFactor;
            }
        }
        
        /// <summary>
        /// Update the correlation matrix and heatmap
        /// </summary>
        private void UpdateCorrelationMatrix()
        {
            if (_comparisonResult == null || _comparisonResult.CorrelationMatrix == null)
                return;
                
            int n = _comparisonResult.StrategyResults.Count;
            if (n <= 1)
                return;
                
            // Setup the correlation matrix data grid
            CorrelationMatrixGrid.Columns.Clear();
            
            // Add the header column
            var headerColumn = new DataGridTextColumn
            {
                Header = "Strategy",
                Binding = new System.Windows.Data.Binding("StrategyName"),
                FontWeight = FontWeights.Bold
            };
            CorrelationMatrixGrid.Columns.Add(headerColumn);
            
            // Add a column for each strategy
            for (int i = 0; i < n; i++)
            {
                var strategyName = _comparisonResult.StrategyResults[i].StrategyName;
                var column = new DataGridTextColumn
                {
                    Header = strategyName,
                    Binding = new System.Windows.Data.Binding($"Values[{i}]")
                };
                CorrelationMatrixGrid.Columns.Add(column);
            }
            
            // Create the data rows
            var correlationRows = new ObservableCollection<CorrelationRow>();
            
            for (int i = 0; i < n; i++)
            {
                var strategyName = _comparisonResult.StrategyResults[i].StrategyName;
                var row = new CorrelationRow { StrategyName = strategyName };
                
                for (int j = 0; j < n; j++)
                {
                    var correlationValue = _comparisonResult.CorrelationMatrix[i, j];
                    row.Values.Add(correlationValue.ToString("F2"));
                }
                
                correlationRows.Add(row);
            }
            
            CorrelationMatrixGrid.ItemsSource = correlationRows;
            
            // Draw the correlation heatmap
            UpdateCorrelationHeatmap(_comparisonResult.CorrelationMatrix);
        }
        
        /// <summary>
        /// Update the correlation heatmap visualization
        /// </summary>
        private void UpdateCorrelationHeatmap(double[,] correlationMatrix)
        {
            CorrelationHeatmapCanvas.Children.Clear();
            
            if (correlationMatrix == null)
                return;
                
            int n = correlationMatrix.GetLength(0);
            if (n == 0)
                return;
                
            // Get the strategy names
            var strategyNames = _comparisonResult.StrategyResults
                .Select(s => s.StrategyName)
                .ToList();
                
            // Prepare to draw the heatmap
            double cellWidth = CorrelationHeatmapCanvas.ActualWidth / (n + 1);
            double cellHeight = CorrelationHeatmapCanvas.ActualHeight / (n + 1);
            
            // Draw grid cells
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double correlation = correlationMatrix[i, j];
                    
                    // Create a rectangle for the cell
                    var rect = new System.Windows.Shapes.Rectangle
                    {
                        Width = cellWidth,
                        Height = cellHeight,
                        Fill = GetCorrelationColor(correlation)
                    };
                    
                    // Position the rectangle
                    Canvas.SetLeft(rect, (j + 1) * cellWidth);
                    Canvas.SetTop(rect, (i + 1) * cellHeight);
                    
                    CorrelationHeatmapCanvas.Children.Add(rect);
                    
                    // Add correlation value text
                    var text = new TextBlock
                    {
                        Text = correlation.ToString("F2"),
                        FontSize = 10,
                        Foreground = correlation > 0.7 || correlation < -0.7 ? Brushes.White : Brushes.Black,
                        HorizontalAlignment = HorizontalAlignment.Center,
                        VerticalAlignment = VerticalAlignment.Center
                    };
                    
                    Canvas.SetLeft(text, (j + 1) * cellWidth + (cellWidth - text.ActualWidth) / 2);
                    Canvas.SetTop(text, (i + 1) * cellHeight + (cellHeight - text.ActualHeight) / 2);
                    
                    CorrelationHeatmapCanvas.Children.Add(text);
                }
            }
            
            // Add labels for strategies
            for (int i = 0; i < n; i++)
            {
                // Row labels
                var rowLabel = new TextBlock
                {
                    Text = strategyNames[i],
                    FontSize = 10,
                    FontWeight = FontWeights.Bold,
                    VerticalAlignment = VerticalAlignment.Center
                };
                
                Canvas.SetLeft(rowLabel, 0);
                Canvas.SetTop(rowLabel, (i + 1) * cellHeight + (cellHeight - rowLabel.ActualHeight) / 2);
                
                CorrelationHeatmapCanvas.Children.Add(rowLabel);
                
                // Column labels
                var colLabel = new TextBlock
                {
                    Text = strategyNames[i],
                    FontSize = 10,
                    FontWeight = FontWeights.Bold,
                    HorizontalAlignment = HorizontalAlignment.Center
                };
                
                Canvas.SetLeft(colLabel, (i + 1) * cellWidth + (cellWidth - colLabel.ActualWidth) / 2);
                Canvas.SetTop(colLabel, 0);
                
                CorrelationHeatmapCanvas.Children.Add(colLabel);
            }
        }
        
        /// <summary>
        /// Update the drawdown comparison charts
        /// </summary>
        private void UpdateDrawdownComparison()
        {
            if (_comparisonResult == null)
                return;
                
            // Drawdown comparison chart
            DrawdownComparisonChart.Series = new SeriesCollection();
            
            int strategyIndex = 0;
            foreach (var strategyResult in _comparisonResult.StrategyResults)
            {
                // Skip if not selected
                if (!_selectedStrategies.ContainsKey(strategyResult.StrategyName) || 
                    !_selectedStrategies[strategyResult.StrategyName])
                    continue;
                    
                var drawdownCurve = strategyResult.Result.DrawdownCurve;
                
                // Create series for this strategy
                var series = new LineSeries
                {
                    Title = strategyResult.StrategyName,
                    Values = new ChartValues<double>(drawdownCurve.Select(d => d.Drawdown)),
                    PointGeometry = null,
                    Stroke = GetStrategyColor(strategyResult.StrategyName, strategyIndex)
                };
                
                DrawdownComparisonChart.Series.Add(series);
                strategyIndex++;
            }
            
            // Set the X-axis labels to dates
            if (_comparisonResult.StrategyResults.Any() && _comparisonResult.StrategyResults[0].Result.DrawdownCurve.Any())
            {
                var dates = _comparisonResult.StrategyResults[0].Result.DrawdownCurve
                    .Select(d => d.Date.ToShortDateString())
                    .ToArray();
                    
                DrawdownComparisonChart.AxisX[0].Labels = dates;
                
                // Set labels to show only every Nth date to prevent overcrowding
                int skipFactor = Math.Max(1, dates.Length / 15);
                DrawdownComparisonChart.AxisX[0].Separator.Step = skipFactor;
            }
            
            // Max drawdown comparison chart (bar chart)
            MaxDrawdownComparisonChart.Series = new SeriesCollection
            {
                new ColumnSeries
                {
                    Title = "Maximum Drawdown",
                    Values = new ChartValues<double>(
                        _comparisonResult.StrategyResults
                            .Where(s => _selectedStrategies.ContainsKey(s.StrategyName) && _selectedStrategies[s.StrategyName])
                            .Select(s => s.Result.MaxDrawdown)
                    )
                }
            };
            
            // Set the strategy names as labels
            MaxDrawdownComparisonChart.AxisX[0].Labels = _comparisonResult.StrategyResults
                .Where(s => _selectedStrategies.ContainsKey(s.StrategyName) && _selectedStrategies[s.StrategyName])
                .Select(s => s.StrategyName)
                .ToArray();
        }
        
        /// <summary>
        /// Recalculate the optimal portfolio allocation based on the selected strategies
        /// </summary>
        private void RecalculateOptimalPortfolio()
        {
            if (_comparisonResult == null)
                return;
                
            double riskAversion = RiskAversionSlider.Value;
            
            // Filter to only selected strategies
            var selectedStrategies = _comparisonResult.StrategyResults
                .Where(s => _selectedStrategies.ContainsKey(s.StrategyName) && _selectedStrategies[s.StrategyName])
                .ToList();
                
            if (selectedStrategies.Count == 0)
                return;
                
            // Create a new StrategyComparisonResult with only the selected strategies
            var filteredComparison = new StrategyComparisonResult
            {
                Symbol = _comparisonResult.Symbol,
                TimeFrame = _comparisonResult.TimeFrame,
                StartDate = _comparisonResult.StartDate,
                EndDate = _comparisonResult.EndDate,
                AssetClass = _comparisonResult.AssetClass,
                InitialCapital = _comparisonResult.InitialCapital,
                StrategyResults = selectedStrategies,
                CorrelationMatrix = _comparisonResult.CorrelationMatrix
            };
            
            // Calculate optimal portfolio weights
            _portfolioWeights = filteredComparison.CalculateOptimalPortfolioWeights(riskAversion);
            
            // Simulate the combined portfolio
            _combinedPortfolioResult = filteredComparison.SimulateCombinedPortfolio(_portfolioWeights);
            
            // Update the portfolio allocation chart
            UpdatePortfolioAllocationChart();
            
            // Update the combined portfolio performance chart
            UpdateCombinedPortfolioChart();
            
            // Update the combined portfolio metrics
            UpdateCombinedPortfolioMetrics();
            
            // Update the performance metrics grid to include combined portfolio
            UpdatePerformanceMetricsGrid();
        }
        
        /// <summary>
        /// Update the portfolio allocation pie chart
        /// </summary>
        private void UpdatePortfolioAllocationChart()
        {
            if (_portfolioWeights == null || !_portfolioWeights.Any())
                return;
                
            PortfolioAllocationChart.Series = new SeriesCollection();
            
            int strategyIndex = 0;
            foreach (var kvp in _portfolioWeights)
            {
                string strategyName = kvp.Key;
                double weight = kvp.Value;
                
                // Skip if tiny weight
                if (weight < 0.01)
                    continue;
                    
                PortfolioAllocationChart.Series.Add(new PieSeries
                {
                    Title = $"{strategyName} ({weight:P1})",
                    Values = new ChartValues<double> { weight },
                    DataLabels = true,
                    LabelPoint = point => $"{strategyName}: {point.Y:P1}",
                    Fill = GetStrategyColor(strategyName, strategyIndex)
                });
                
                strategyIndex++;
            }
        }
        
        /// <summary>
        /// Update the combined portfolio performance chart
        /// </summary>
        private void UpdateCombinedPortfolioChart()
        {
            if (_combinedPortfolioResult == null)
                return;
                
            CombinedEquityCurveChart.Series = new SeriesCollection();
            
            // Add combined portfolio equity curve
            var combinedSeries = new LineSeries
            {
                Title = "Combined Portfolio",
                Values = new ChartValues<double>(_combinedPortfolioResult.EquityCurve.Select(e => e.Equity)),
                PointGeometry = null,
                Stroke = Brushes.Gold,
                StrokeThickness = 3
            };
            
            CombinedEquityCurveChart.Series.Add(combinedSeries);
            
            // Add individual strategies for comparison
            int strategyIndex = 0;
            foreach (var strategyResult in _comparisonResult.StrategyResults)
            {
                // Skip if not selected
                if (!_selectedStrategies.ContainsKey(strategyResult.StrategyName) || 
                    !_selectedStrategies[strategyResult.StrategyName])
                    continue;
                    
                var equityCurve = strategyResult.Result.EquityCurve;
                
                // Create series for this strategy
                var series = new LineSeries
                {
                    Title = strategyResult.StrategyName,
                    Values = new ChartValues<double>(equityCurve.Select(e => e.Equity)),
                    PointGeometry = null,
                    Stroke = GetStrategyColor(strategyResult.StrategyName, strategyIndex),
                    StrokeThickness = 1 // Thinner to highlight the combined portfolio
                };
                
                CombinedEquityCurveChart.Series.Add(series);
                strategyIndex++;
            }
            
            // Set the X-axis labels to dates
            if (_combinedPortfolioResult.EquityCurve.Any())
            {
                var dates = _combinedPortfolioResult.EquityCurve
                    .Select(e => e.Date.ToShortDateString())
                    .ToArray();
                    
                CombinedEquityCurveChart.AxisX[0].Labels = dates;
                
                // Set labels to show only every Nth date to prevent overcrowding
                int skipFactor = Math.Max(1, dates.Length / 15);
                CombinedEquityCurveChart.AxisX[0].Separator.Step = skipFactor;
            }
        }
        
        /// <summary>
        /// Update the combined portfolio metrics display
        /// </summary>
        private void UpdateCombinedPortfolioMetrics()
        {
            if (_combinedPortfolioResult == null)
                return;
                
            CombinedReturnText.Text = _combinedPortfolioResult.TotalReturn.ToString("P2");
            CombinedCagrText.Text = _combinedPortfolioResult.CAGR.ToString("P2");
            CombinedDrawdownText.Text = _combinedPortfolioResult.MaxDrawdown.ToString("P2");
            CombinedSharpeText.Text = _combinedPortfolioResult.SharpeRatio.ToString("F2");
            CombinedSortinoText.Text = _combinedPortfolioResult.SortinoRatio.ToString("F2");
            CombinedCalmarText.Text = _combinedPortfolioResult.CalmarRatio.ToString("F2");
        }
        
        /// <summary>
        /// Get a consistent color for a strategy based on its name or index
        /// </summary>
        private Brush GetStrategyColor(string strategyName, int strategyIndex)
        {
            // First try to get a color based on the strategy name
            if (_strategyColors.ContainsKey(strategyName))
                return _strategyColors[strategyName];
                
            // If not found, use a color from the predefined list
            var colors = _strategyColors.Values.ToArray();
            return colors[strategyIndex % colors.Length];
        }
        
        /// <summary>
        /// Get a color for a correlation value (-1 to 1)
        /// </summary>
        private Brush GetCorrelationColor(double correlation)
        {
            // Red for negative correlation, blue for positive
            if (correlation < 0)
            {
                // Negative correlation: shades of red (stronger negative = darker red)
                byte intensity = (byte)(255 * (1 + correlation)); // Scale from 0-255
                return new SolidColorBrush(Color.FromRgb(255, intensity, intensity));
            }
            else
            {
                // Positive correlation: shades of blue (stronger positive = darker blue)
                byte intensity = (byte)(255 * (1 - correlation)); // Scale from 0-255
                return new SolidColorBrush(Color.FromRgb(intensity, intensity, 255));
            }
        }
        
        // Event Handlers
        
        private void StrategyCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            // Update the selected strategies dictionary
            var checkbox = sender as CheckBox;
            if (checkbox == null)
                return;
                
            string strategyName = checkbox.Content.ToString();
            bool isSelected = checkbox.IsChecked ?? false;
            
            _selectedStrategies[strategyName] = isSelected;
            
            // Update charts with selected strategies
            UpdateEquityCurvesChart();
            UpdateDrawdownComparison();
            
            // Recalculate portfolio if necessary
            RecalculateOptimalPortfolio();
        }
        
        private void AddStrategyButton_Click(object sender, RoutedEventArgs e)
        {
            // This functionality would require a dialog to select and configure a new strategy
            // Show a dialog to select a strategy from available types
            MessageBox.Show("Add Strategy feature not implemented yet. You would select a strategy type and configure its parameters here.");
        }
        
        private void OptimizeWeightsButton_Click(object sender, RoutedEventArgs e)
        {
            RecalculateOptimalPortfolio();
        }
        
        private void RiskAversionSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (RiskAversionValueText != null)
            {
                RiskAversionValueText.Text = e.NewValue.ToString("F1");
            }
        }
        
        private void RecalculatePortfolioButton_Click(object sender, RoutedEventArgs e)
        {
            RecalculateOptimalPortfolio();
        }
        
        private void ExportResultsButton_Click(object sender, RoutedEventArgs e)
        {
            // Export functionality would save results to CSV or PDF
            MessageBox.Show("Export feature not implemented yet. This would export performance metrics and charts to a file format of your choice.");
        }
    }
    
    /// <summary>
    /// Helper class for binding strategy selections in the UI
    /// </summary>
    public class StrategySelectionItem
    {
        public string StrategyName { get; set; }
        public bool IsSelected { get; set; }
    }
    
    /// <summary>
    /// Helper class for showing performance metrics in the grid
    /// </summary>
    public class PerformanceMetricsRow
    {
        public string StrategyName { get; set; }
        public string TotalReturn { get; set; }
        public string CAGR { get; set; }
        public string MaxDrawdown { get; set; }
        public string SharpeRatio { get; set; }
        public string SortinoRatio { get; set; }
        public string WinRate { get; set; }
        public string ProfitFactor { get; set; }
        public string Consistency { get; set; }
        public string Volatility { get; set; }
    }
    
    /// <summary>
    /// Helper class for showing correlation matrix rows in the grid
    /// </summary>
    public class CorrelationRow
    {
        public string StrategyName { get; set; }
        public List<string> Values { get; set; } = new List<string>();
    }
}