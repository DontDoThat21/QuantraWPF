using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.Modules;
using Quantra.DAL.Services; // Add this for SentimentPriceVisualData, SentimentShiftEvent

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for SentimentDashboardControl.xaml
    /// </summary>
    public partial class SentimentDashboardControl : UserControl
    {
        #region Chart Series Collections
        public SeriesCollection SentimentSeries { get; private set; }
        public SeriesCollection RatingDistributionSeries { get; private set; }
        public SeriesCollection PriceTargetSeries { get; private set; }
        public SeriesCollection InsiderActivitySeries { get; private set; }
        public SeriesCollection SentimentShiftSeries { get; private set; }
        #endregion

        #region Services
        // Disambiguate SentimentPriceCorrelationAnalysis
        private readonly Quantra.Modules.SentimentPriceCorrelationAnalysis _sentimentCorrelationAnalysis;
        private readonly IAnalystRatingService _analystRatingService;
        private readonly IInsiderTradingService _insiderTradingService;
        #endregion

        #region Data Caching
        private SentimentPriceVisualData _currentSentimentData;
        private AnalystRatingAggregate _currentAnalystData;
        private List<InsiderTransaction> _currentInsiderData;
        private int _currentLookbackDays = 30;
        #endregion

        #region Properties
        private string _symbol;
        public string Symbol 
        { 
            get => _symbol; 
            set 
            {
                _symbol = value;
                SymbolTextBlock.Text = value.ToUpper();
            }
        }
        #endregion

        /// <summary>
        /// Initializes a new instance of the SentimentDashboardControl
        /// </summary>
        public SentimentDashboardControl(UserSettings userSettings,
            UserSettingsService userSettingsService,
            LoggingService loggingService
            )
        {
            InitializeComponent();
            
            // Initialize chart collections
            SentimentSeries = new SeriesCollection();
            RatingDistributionSeries = new SeriesCollection();
            PriceTargetSeries = new SeriesCollection();
            InsiderActivitySeries = new SeriesCollection();
            SentimentShiftSeries = new SeriesCollection();
            
            // Set chart series
            SentimentTrendChart.Series = SentimentSeries;
            RatingDistributionChart.Series = RatingDistributionSeries;
            PriceTargetTrendChart.Series = PriceTargetSeries;
            InsiderActivityChart.Series = InsiderActivitySeries;
            SentimentShiftChart.Series = SentimentShiftSeries;
            
            // Initialize services
            _sentimentCorrelationAnalysis = new Quantra.Modules.SentimentPriceCorrelationAnalysis(userSettings, userSettingsService, loggingService);
            _analystRatingService = ServiceLocator.Resolve<IAnalystRatingService>();
            _insiderTradingService = ServiceLocator.Resolve<IInsiderTradingService>();
            
            // Default to empty symbol
            Symbol = "--";
        }

        #region Public Methods
        /// <summary>
        /// Updates the entire dashboard with data for the specified symbol
        /// </summary>
        /// <param name="symbol">Stock symbol to display</param>
        public async void UpdateDashboard(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return;

            Symbol = symbol;
            
            try
            {
                // Update the timeframe from the UI selection
                UpdateTimeframeFromUI();
                
                // Get sentiment data
                // TODO: Replace with actual implementation when available
                // _currentSentimentData = await _sentimentCorrelationAnalysis.AnalyzeSentimentPriceCorrelation(
                //     symbol, 
                //     _currentLookbackDays,
                //     null);
                // For now, set to null or mock
                _currentSentimentData = null;
                
                // Get analyst data
                _currentAnalystData = await _analystRatingService.GetAggregatedRatingsAsync(symbol);
                
                // Get insider trading data
                _currentInsiderData = await _insiderTradingService.GetInsiderTransactionsAsync(symbol);
                
                // Update all tabs
                UpdateHistoricalTrendsTab();
                UpdateAnalystRatingsTab();
                UpdateInsiderTradingTab();
                UpdateNewsAndShiftsTab();
                
                // Update overall sentiment score
                UpdateOverallSentiment();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to update sentiment dashboard", ex.ToString());
                MessageBox.Show($"Error updating sentiment dashboard: {ex.Message}", "Error", 
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        
        /// <summary>
        /// Clears all visualization data from the dashboard
        /// </summary>
        public void ClearDashboard()
        {
            // Clear sentiment trends
            SentimentSeries.Clear();
            
            // Clear analyst ratings
            RatingDistributionSeries.Clear();
            PriceTargetSeries.Clear();
            AnalystRatingListView.Items.Clear();
            
            // Clear insider trading
            InsiderActivitySeries.Clear();
            InsiderTransactionListView.Items.Clear();
            
            // Clear sentiment shifts
            SentimentShiftSeries.Clear();
            SentimentShiftListView.Items.Clear();
            
            // Reset metrics
            ResetMetrics();
            
            // Reset symbol
            Symbol = "--";
        }
        #endregion

        #region Event Handlers
        /// <summary>
        /// Handles click on the refresh button
        /// </summary>
        private void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            UpdateDashboard(Symbol);
        }
        
        /// <summary>
        /// Handles selection change for the trend timeframe combo box
        /// </summary>
        private void TrendTimeframeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (string.IsNullOrEmpty(Symbol) || Symbol == "--")
                return;
                
            UpdateTimeframeFromUI();
            UpdateDashboard(Symbol);
        }
        
        /// <summary>
        /// Handles check/uncheck events for source checkboxes
        /// </summary>
        private void SourceCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            if (_currentSentimentData == null)
                return;
                
            UpdateSentimentSeriesVisibility();
        }
        
        /// <summary>
        /// Handles selection change for the insider view combo box
        /// </summary>
        private void InsiderViewComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_currentInsiderData == null)
                return;
                
            UpdateInsiderActivityChart();
        }
        
        /// <summary>
        /// Handles selection change for the sentiment shift filter combo box
        /// </summary>
        private void ShiftFilterComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_currentSentimentData == null)
                return;
                
            UpdateSentimentShiftList();
        }
        #endregion

        #region Tab Updates
        /// <summary>
        /// Updates the Historical Trends tab with current data
        /// </summary>
        private void UpdateHistoricalTrendsTab()
        {
            if (_currentSentimentData == null)
                return;
                
            var data = _currentSentimentData;
            
            // Clear existing chart data
            SentimentSeries.Clear();
            
            // Update date labels for sentiment trend chart
            var dateLabels = data.Dates.Select(d => d.ToString("MM/dd")).ToList();
            SentimentTrendChartAxisX.Labels = dateLabels;
            
            // Define consistent colors for each source
            var sourceColors = new Dictionary<string, Color>
            {
                { "Twitter", Colors.DeepSkyBlue },
                { "News", Colors.Yellow },
                { "AnalystRatings", Colors.LimeGreen },
                { "InsiderTrading", Colors.MediumPurple }
            };
            
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
            
            // Apply source visibility settings
            UpdateSentimentSeriesVisibility();
            
            // Update metrics
            UpdateHistoricalTrendsMetrics();
        }
        
        /// <summary>
        /// Updates the Analyst Ratings tab with current data
        /// </summary>
        private void UpdateAnalystRatingsTab()
        {
            if (_currentAnalystData == null)
                return;
                
            var data = _currentAnalystData;
            
            // Update consensus metrics
            ConsensusRatingText.Text = data.ConsensusRating;
            BuyCountText.Text = data.BuyCount.ToString();
            HoldCountText.Text = data.HoldCount.ToString();
            SellCountText.Text = data.SellCount.ToString();
            
            // Update price target information
            if (data.AveragePriceTarget > 0)
            {
                AvgPriceTargetText.Text = $"${data.AveragePriceTarget:F2}";
                PriceTargetRangeText.Text = $"${data.LowestPriceTarget:F2} - ${data.HighestPriceTarget:F2}";
                
                // Calculate change from current price (assuming we have access to current price)
                // This is a simplified calculation - in a real implementation we'd get the actual current price
                double currentPrice = GetEstimatedCurrentPrice();
                if (currentPrice > 0)
                {
                    double pctChange = (data.AveragePriceTarget - currentPrice) / currentPrice * 100.0;
                    string direction = pctChange >= 0 ? "+" : "";
                    TargetPctChangeText.Text = $" ({direction}{pctChange:F1}%)";
                    
                    // Set color based on the direction
                    TargetPctChangeText.Foreground = pctChange >= 0
                        ? new SolidColorBrush(Color.FromRgb(32, 192, 64)) // Green
                        : new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
                }
                else
                {
                    TargetPctChangeText.Text = "";
                }
            }
            else
            {
                AvgPriceTargetText.Text = "$0.00";
                PriceTargetRangeText.Text = "N/A";
                TargetPctChangeText.Text = "";
            }
            
            // Style consensus rating based on its value
            switch (data.ConsensusRating)
            {
                case "Buy":
                    ConsensusRatingText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
                    break;
                case "Sell":
                    ConsensusRatingText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
                    break;
                default:
                    ConsensusRatingText.Foreground = new SolidColorBrush(Color.FromRgb(192, 192, 32)); // Yellow
                    break;
            }
            
            // Clear existing chart data
            RatingDistributionSeries.Clear();
            PriceTargetSeries.Clear();
            
            // Update rating distribution chart
            UpdateRatingDistributionChart();
            
            // Update price target trend chart
            UpdatePriceTargetTrendChart();
            
            // Update ratings list
            UpdateAnalystRatingsList();
        }
        
        /// <summary>
        /// Updates the Insider Trading tab with current data
        /// </summary>
        private void UpdateInsiderTradingTab()
        {
            if (_currentInsiderData == null)
                return;
                
            // Calculate metrics
            UpdateInsiderTradingMetrics();
            
            // Update insider activity chart
            UpdateInsiderActivityChart();
            
            // Update transaction list
            UpdateInsiderTransactionList();
        }
        
        /// <summary>
        /// Updates the News and Sentiment Shifts tab with current data
        /// </summary>
        private void UpdateNewsAndShiftsTab()
        {
            if (_currentSentimentData == null)
                return;
                
            // Update sentiment shift chart
            UpdateSentimentShiftChart();
            
            // Update sentiment shift list
            UpdateSentimentShiftList();
        }
        #endregion

        #region Chart Updates
        /// <summary>
        /// Updates the visibility of sentiment series based on checkbox selections
        /// </summary>
        private void UpdateSentimentSeriesVisibility()
        {
            foreach (var series in SentimentSeries)
            {
                if (series is LineSeries lineSeries)
                {
                    switch (series.Title)
                    {
                        case "Twitter":
                            lineSeries.Visibility = TwitterCheckBox.IsChecked == true ? Visibility.Visible : Visibility.Hidden;
                            break;
                        case "News":
                            lineSeries.Visibility = NewsCheckBox.IsChecked == true ? Visibility.Visible : Visibility.Hidden;
                            break;
                        case "AnalystRatings":
                            lineSeries.Visibility = AnalystCheckBox.IsChecked == true ? Visibility.Visible : Visibility.Hidden;
                            break;
                        case "InsiderTrading":
                            lineSeries.Visibility = InsiderCheckBox.IsChecked == true ? Visibility.Visible : Visibility.Hidden;
                            break;
                        case "Combined":
                            lineSeries.Visibility = CombinedCheckBox.IsChecked == true ? Visibility.Visible : Visibility.Hidden;
                            break;
                    }
                }
            }
        }
        
        /// <summary>
        /// Updates the rating distribution chart
        /// </summary>
        private void UpdateRatingDistributionChart()
        {
            if (_currentAnalystData == null || _currentAnalystData.Ratings == null)
                return;
                
            // Group ratings by category
            var ratings = _currentAnalystData.Ratings;
            var ratingGroups = ratings
                .GroupBy(r => r.Rating)
                .Select(g => new { Rating = g.Key, Count = g.Count() })
                .OrderByDescending(g => g.Count)
                .ToList();
                
            // Create chart series
            var ratingLabels = ratingGroups.Select(g => g.Rating).ToList();
            var ratingCounts = new ChartValues<int>(ratingGroups.Select(g => g.Count));
            
            // Set X axis labels
            RatingDistributionChartAxisX.Labels = ratingLabels;
            
            // Add column series
            RatingDistributionSeries.Add(new ColumnSeries
            {
                Title = "Ratings",
                Values = ratingCounts,
                Fill = new SolidColorBrush(Color.FromRgb(0, 180, 200)), // Cyan-ish
                DataLabels = true,
                LabelPoint = point => point.Y.ToString()
            });
        }
        
        /// <summary>
        /// Updates the price target trend chart
        /// </summary>
        private void UpdatePriceTargetTrendChart()
        {
            if (_currentAnalystData == null || _currentAnalystData.Ratings == null)
                return;
                
            // Filter ratings with valid price targets and sort by date
            var ratings = _currentAnalystData.Ratings
                .Where(r => r.PriceTarget > 0)
                .OrderBy(r => r.RatingDate)
                .ToList();
                
            if (ratings.Count == 0)
                return;
                
            // Get dates and price targets
            var dates = ratings.Select(r => r.RatingDate.ToString("MM/dd")).ToList();
            var priceTargets = new ChartValues<double>(ratings.Select(r => r.PriceTarget));
            
            // Set X axis labels
            PriceTargetTrendChartAxisX.Labels = dates;
            
            // Add line series
            PriceTargetSeries.Add(new LineSeries
            {
                Title = "Price Target",
                Values = priceTargets,
                PointGeometry = DefaultGeometries.Diamond,
                PointGeometrySize = 8,
                Stroke = new SolidColorBrush(Colors.Cyan),
                Fill = new SolidColorBrush(Color.FromArgb(50, 0, 255, 255)),
                LineSmoothness = 0
            });
            
            // Add average price target line
            if (_currentAnalystData.AveragePriceTarget > 0)
            {
                var avgValues = new ChartValues<double>(
                    Enumerable.Repeat(_currentAnalystData.AveragePriceTarget, dates.Count));
                    
                PriceTargetSeries.Add(new LineSeries
                {
                    Title = "Average",
                    Values = avgValues,
                    PointGeometry = null,
                    Stroke = new SolidColorBrush(Colors.White),
                    StrokeDashArray = new DoubleCollection(new[] { 4.0, 2.0 }),
                    StrokeThickness = 1
                });
            }
        }
        
        /// <summary>
        /// Updates the insider activity chart based on the selected view
        /// </summary>
        private void UpdateInsiderActivityChart()
        {
            if (_currentInsiderData == null || _currentInsiderData.Count == 0)
                return;
                
            // Clear existing chart data
            InsiderActivitySeries.Clear();
            
            // Get selected view
            var viewMode = InsiderViewComboBox.SelectedIndex;
            
            // Group transactions by date
            var transactionsByDate = _currentInsiderData
                .OrderBy(t => t.TransactionDate)
                .GroupBy(t => t.TransactionDate.Date)
                .ToList();
                
            // Prepare chart data based on selected view
            var dates = new List<string>();
            var buyValues = new ChartValues<double>();
            var sellValues = new ChartValues<double>();
            var netValues = new ChartValues<double>();
            
            switch (viewMode)
            {
                case 0: // Transaction Value
                    foreach (var group in transactionsByDate)
                    {
                        dates.Add(group.Key.ToString("MM/dd"));
                        
                        double buyValue = group.Where(t => t.TransactionType == InsiderTransactionType.Purchase)
                            .Sum(t => t.Value) / 1000; // Convert to thousands
                            
                        double sellValue = group.Where(t => t.TransactionType == InsiderTransactionType.Sale)
                            .Sum(t => t.Value) / 1000; // Convert to thousands
                            
                        buyValues.Add(buyValue);
                        sellValues.Add(sellValue);
                        netValues.Add(buyValue - sellValue);
                    }
                    
                    InsiderActivityChartAxisY.Title = "Value ($ thousands)";
                    break;
                    
                case 1: // Transaction Count
                    foreach (var group in transactionsByDate)
                    {
                        dates.Add(group.Key.ToString("MM/dd"));
                        
                        int buyCount = group.Count(t => t.TransactionType == InsiderTransactionType.Purchase);
                        int sellCount = group.Count(t => t.TransactionType == InsiderTransactionType.Sale);
                        
                        buyValues.Add(buyCount);
                        sellValues.Add(sellCount);
                        netValues.Add(buyCount - sellCount);
                    }
                    
                    InsiderActivityChartAxisY.Title = "Number of Transactions";
                    break;
                    
                case 2: // Buy/Sell Balance
                    foreach (var group in transactionsByDate)
                    {
                        dates.Add(group.Key.ToString("MM/dd"));
                        
                        double buyValue = group.Where(t => t.TransactionType == InsiderTransactionType.Purchase)
                            .Sum(t => t.Value);
                            
                        double sellValue = group.Where(t => t.TransactionType == InsiderTransactionType.Sale)
                            .Sum(t => t.Value);
                            
                        double total = buyValue + sellValue;
                        double ratio = total > 0 ? (buyValue - sellValue) / total : 0;
                        
                        netValues.Add(ratio);
                    }
                    
                    InsiderActivityChartAxisY.Title = "Buy/Sell Balance (-1 to 1)";
                    InsiderActivityChartAxisY.MinValue = -1;
                    InsiderActivityChartAxisY.MaxValue = 1;
                    break;
                    
                case 3: // Sentiment Impact
                    foreach (var group in transactionsByDate)
                    {
                        dates.Add(group.Key.ToString("MM/dd"));
                        
                        // Calculate sentiment impact for each transaction and average
                        var transactions = group.ToList();
                        double totalImpact = 0;
                        
                        foreach (var transaction in transactions)
                        {
                            double sentimentImpact = transaction.GetSentimentScore();
                            totalImpact += sentimentImpact;
                        }
                        
                        double averageImpact = transactions.Count > 0 ? totalImpact / transactions.Count : 0;
                        netValues.Add(averageImpact);
                    }
                    
                    InsiderActivityChartAxisY.Title = "Sentiment Impact (-1 to 1)";
                    InsiderActivityChartAxisY.MinValue = -1;
                    InsiderActivityChartAxisY.MaxValue = 1;
                    break;
            }
            
            // Set X axis labels
            InsiderActivityChartAxisX.Labels = dates;
            
            // Add chart series based on view mode
            if (viewMode == 0 || viewMode == 1) // Transaction Value or Count
            {
                // Add buy series
                InsiderActivitySeries.Add(new ColumnSeries
                {
                    Title = "Buy",
                    Values = buyValues,
                    Fill = new SolidColorBrush(Color.FromRgb(32, 192, 64)), // Green
                    DataLabels = false
                });
                
                // Add sell series
                InsiderActivitySeries.Add(new ColumnSeries
                {
                    Title = "Sell",
                    Values = sellValues,
                    Fill = new SolidColorBrush(Color.FromRgb(192, 32, 32)), // Red
                    DataLabels = false
                });
            }
            
            // Always add net series
            InsiderActivitySeries.Add(new LineSeries
            {
                Title = viewMode <= 1 ? "Net" : viewMode == 2 ? "Balance" : "Sentiment",
                Values = netValues,
                PointGeometry = DefaultGeometries.Diamond,
                PointGeometrySize = 6,
                Stroke = new SolidColorBrush(Colors.Cyan),
                StrokeThickness = 2,
                LineSmoothness = 0.5,
                ScalesYAt = viewMode <= 1 ? 1 : 0 // Secondary Y axis for Net in first two views
            });
        }
        
        /// <summary>
        /// Updates the sentiment shift chart
        /// </summary>
        private void UpdateSentimentShiftChart()
        {
            if (_currentSentimentData == null || _currentSentimentData.SentimentShiftEvents == null)
                return;
                
            // Clear existing chart data
            SentimentShiftSeries.Clear();
            
            // Add combined sentiment series
            if (_currentSentimentData.CombinedSentiment != null && _currentSentimentData.CombinedSentiment.Count > 0)
            {
                var dates = _currentSentimentData.Dates.Select(d => d.ToString("MM/dd")).ToList();
                SentimentShiftChartAxisX.Labels = dates;
                
                SentimentShiftSeries.Add(new LineSeries
                {
                    Title = "Combined Sentiment",
                    Values = new ChartValues<double>(_currentSentimentData.CombinedSentiment),
                    PointGeometry = null,
                    Stroke = new SolidColorBrush(Colors.White),
                    StrokeThickness = 2,
                    LineSmoothness = 0.3
                });
                
                // Add price change series as columns in background
                if (_currentSentimentData.PriceChanges != null && _currentSentimentData.PriceChanges.Count > 0)
                {
                    // Scale to fit in the chart
                    var scaledChanges = _currentSentimentData.PriceChanges.Select(p => p / 100.0).ToList();
                    
                    SentimentShiftSeries.Add(new ColumnSeries
                    {
                        Title = "Price Changes",
                        Values = new ChartValues<double>(scaledChanges),
                        Fill = new SolidColorBrush(Color.FromArgb(40, 0, 180, 200)), // Semi-transparent cyan
                        MaxColumnWidth = double.PositiveInfinity,
                        DataLabels = false
                    });
                }
                
                // Mark significant sentiment shifts on the chart with points
                foreach (Quantra.Modules.SentimentShiftEvent shift in _currentSentimentData.SentimentShiftEvents)
                {
                    // Find the index of this date in the original dates list
                    int index = _currentSentimentData.Dates.FindIndex(d => d.Date == shift.Date.Date);
                    
                    if (index >= 0 && index < _currentSentimentData.CombinedSentiment.Count)
                    {
                        // Get the sentiment value at this point
                        double sentimentValue = _currentSentimentData.CombinedSentiment[index];
                        
                        // Create a single-point series to highlight this shift
                        var pointValue = new ChartValues<double> { sentimentValue };
                        var pointDates = new[] { shift.Date.ToString("MM/dd") };
                        
                        var color = shift.SentimentShift > 0 ? 
                            Color.FromRgb(32, 192, 64) : // Green for positive
                            Color.FromRgb(192, 32, 32);  // Red for negative
                            
                        SentimentShiftSeries.Add(new LineSeries
                        {
                            Title = $"{shift.Source} Shift",
                            Values = pointValue,
                            PointGeometry = DefaultGeometries.Circle,
                            PointGeometrySize = 10,
                            Stroke = new SolidColorBrush(color),
                            Fill = new SolidColorBrush(color),
                            DataLabels = false,
                            LineSmoothness = 0
                        });
                    }
                }
            }
        }
        #endregion

        #region List Updates
        /// <summary>
        /// Updates the list of analyst ratings
        /// </summary>
        private void UpdateAnalystRatingsList()
        {
            if (_currentAnalystData == null || _currentAnalystData.Ratings == null)
                return;
                
            AnalystRatingListView.Items.Clear();
            
            // Sort by rating date (most recent first)
            var sortedRatings = _currentAnalystData.Ratings
                .OrderByDescending(r => r.RatingDate)
                .Take(10) // Show only the 10 most recent
                .ToList();
                
            // Calculate price target change percentage
            foreach (var rating in sortedRatings)
            {
                // Add property for price target change percent
                var priceTargetChangePercent = 0.0;
                if (rating.PreviousPriceTarget > 0 && rating.PriceTarget > 0)
                {
                    priceTargetChangePercent = (rating.PriceTarget - rating.PreviousPriceTarget) / 
                                               rating.PreviousPriceTarget;
                }
                
                // Create and add display item
                var displayItem = new
                {
                    AnalystName = rating.AnalystName,
                    Rating = rating.Rating,
                    RatingDate = rating.RatingDate,
                    PriceTarget = rating.PriceTarget,
                    PreviousPriceTarget = rating.PreviousPriceTarget,
                    ChangeType = rating.ChangeType.ToString(),
                    SentimentScore = rating.SentimentScore,
                    PriceTargetChangePercent = priceTargetChangePercent
                };
                
                AnalystRatingListView.Items.Add(displayItem);
            }
        }
        
        /// <summary>
        /// Updates the list of insider transactions
        /// </summary>
        private void UpdateInsiderTransactionList()
        {
            if (_currentInsiderData == null)
                return;
                
            InsiderTransactionListView.Items.Clear();
            
            // Sort by transaction date (most recent first)
            var sortedTransactions = _currentInsiderData
                .OrderByDescending(t => t.TransactionDate)
                .Take(10) // Show only the 10 most recent
                .ToList();
                
            // Add transactions to the list
            foreach (var transaction in sortedTransactions)
            {
                // Calculate value
                double value = transaction.Quantity * transaction.Price;
                
                // Create display item
                var displayItem = new
                {
                    InsiderName = transaction.InsiderName,
                    InsiderTitle = transaction.InsiderTitle,
                    TransactionDate = transaction.TransactionDate,
                    TransactionType = transaction.TransactionType.ToString(),
                    Quantity = transaction.Quantity,
                    Price = transaction.Price,
                    Value = value,
                    IsNotableFigure = transaction.IsNotableFigure ? "Yes" : "No"
                };
                
                InsiderTransactionListView.Items.Add(displayItem);
            }
        }
        
        /// <summary>
        /// Updates the list of sentiment shift events
        /// </summary>
        private void UpdateSentimentShiftList()
        {
            if (_currentSentimentData == null || _currentSentimentData.SentimentShiftEvents == null)
                return;
                
            SentimentShiftListView.Items.Clear();
            
            // Get filter selection
            string filter = ShiftFilterComboBox.SelectedItem != null 
                ? (ShiftFilterComboBox.SelectedItem as ComboBoxItem).Content.ToString() 
                : "All Sources";
                
            // Filter events based on selection
            var filteredEvents = _currentSentimentData.SentimentShiftEvents.Where(e => 
            {
                if (filter == "All Sources")
                    return true;
                if (filter == "News Only")
                    return e.Source == "News";
                if (filter == "Social Media Only")
                    return e.Source == "Twitter";
                if (filter == "Analyst Only")
                    return e.Source == "AnalystRatings";
                if (filter == "Insider Only")
                    return e.Source == "InsiderTrading";
                return true;
            })
            .OrderByDescending(e => e.Date)
            .ToList();
            
            // Add events to the list
            foreach (Quantra.Modules.SentimentShiftEvent evt in filteredEvents)
            {
                // Create display item
                var displayItem = new
                {
                    Date = evt.Date,
                    Source = evt.Source,
                    Direction = evt.SentimentShift > 0 ? "Positive" : "Negative",
                    SentimentShift = evt.SentimentShift,
                    SubsequentPriceChange = evt.SubsequentPriceChange,
                    PriceFollowedSentiment = evt.PriceFollowedSentiment,
                    Notes = GetEventNotes(evt)
                };
                
                SentimentShiftListView.Items.Add(displayItem);
            }
        }
        #endregion

        #region Metrics Updates
        /// <summary>
        /// Updates the historical trends metrics
        /// </summary>
        private void UpdateHistoricalTrendsMetrics()
        {
            if (_currentSentimentData == null)
                return;
                
            var data = _currentSentimentData;
            
            // Lead/Lag relationship
            LeadLagDaysText.Text = $"{data.LeadLagRelationship:F1} days";
            
            if (data.LeadLagRelationship > 0)
            {
                LeadLagRelationshipText.Text = "Sentiment leads price";
                LeadLagRelationshipText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
            }
            else if (data.LeadLagRelationship < 0)
            {
                LeadLagRelationshipText.Text = "Price leads sentiment";
                LeadLagRelationshipText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
            }
            else
            {
                LeadLagRelationshipText.Text = "Neutral";
                LeadLagRelationshipText.Foreground = new SolidColorBrush(Colors.White);
            }
            
            // Predictive accuracy
            PredictiveAccuracyText.Text = $"{data.PredictiveAccuracy:P0}";
            
            if (data.PredictiveAccuracy >= 0.7)
            {
                PredictiveAccuracyText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
            }
            else if (data.PredictiveAccuracy <= 0.4)
            {
                PredictiveAccuracyText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
            }
            else
            {
                PredictiveAccuracyText.Foreground = new SolidColorBrush(Colors.Cyan);
            }
            
            // Source correlations
            if (data.SourceCorrelations != null)
            {
                UpdateCorrelationText("Twitter", TwitterCorrelationText, data.SourceCorrelations);
                UpdateCorrelationText("News", NewsCorrelationText, data.SourceCorrelations);
                UpdateCorrelationText("AnalystRatings", AnalystCorrelationText, data.SourceCorrelations);
                UpdateCorrelationText("InsiderTrading", InsiderCorrelationText, data.SourceCorrelations);
            }
            
            // Overall correlation
            OverallCorrelationText.Text = $"{data.OverallCorrelation:F2}";
            
            if (data.OverallCorrelation > 0.5)
            {
                OverallCorrelationText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
            }
            else if (data.OverallCorrelation < -0.5)
            {
                OverallCorrelationText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
            }
            else
            {
                OverallCorrelationText.Foreground = new SolidColorBrush(Colors.Cyan);
            }
            
            // Sentiment trend
            if (data.CombinedSentiment != null && data.CombinedSentiment.Count > 0)
            {
                // Calculate trend over the most recent periods
                int count = Math.Min(5, data.CombinedSentiment.Count);
                if (count > 1)
                {
                    double recentTrend = data.CombinedSentiment[data.CombinedSentiment.Count - 1] - 
                                         data.CombinedSentiment[data.CombinedSentiment.Count - count];
                                         
                    if (recentTrend > 0.1)
                    {
                        SentimentTrendText.Text = "Improving";
                        SentimentTrendText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
                    }
                    else if (recentTrend < -0.1)
                    {
                        SentimentTrendText.Text = "Declining";
                        SentimentTrendText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
                    }
                    else
                    {
                        SentimentTrendText.Text = "Stable";
                        SentimentTrendText.Foreground = new SolidColorBrush(Colors.White);
                    }
                }
                else
                {
                    SentimentTrendText.Text = "Insufficient Data";
                    SentimentTrendText.Foreground = new SolidColorBrush(Colors.White);
                }
            }
            else
            {
                SentimentTrendText.Text = "No Data";
                SentimentTrendText.Foreground = new SolidColorBrush(Colors.White);
            }
            
            // Price impact
            if (data.PredictiveAccuracy > 0.6 && Math.Abs(data.OverallCorrelation) > 0.4)
            {
                PriceImpactText.Text = "High";
                PriceImpactText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
            }
            else if (data.PredictiveAccuracy > 0.5 || Math.Abs(data.OverallCorrelation) > 0.3)
            {
                PriceImpactText.Text = "Medium";
                PriceImpactText.Foreground = new SolidColorBrush(Colors.Cyan);
            }
            else
            {
                PriceImpactText.Text = "Low";
                PriceImpactText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
            }
        }
        
        /// <summary>
        /// Updates the insider trading metrics
        /// </summary>
        private void UpdateInsiderTradingMetrics()
        {
            if (_currentInsiderData == null || _currentInsiderData.Count == 0)
                return;
                
            // Calculate buy/sell ratio
            int buyCount = _currentInsiderData.Count(t => t.TransactionType == InsiderTransactionType.Purchase);
            int sellCount = _currentInsiderData.Count(t => t.TransactionType == InsiderTransactionType.Sale);
            double buySellRatio = sellCount > 0 ? (double)buyCount / sellCount : buyCount;
            
            BuySellRatioText.Text = $"{buySellRatio:F1}";
            
            if (buySellRatio > 1.5)
            {
                BuySellRatioText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green (more buying)
            }
            else if (buySellRatio < 0.7)
            {
                BuySellRatioText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red (more selling)
            }
            else
            {
                BuySellRatioText.Foreground = new SolidColorBrush(Colors.White); // Balanced
            }
            
            // Calculate net insider value
            double buyValue = _currentInsiderData
                .Where(t => t.TransactionType == InsiderTransactionType.Purchase)
                .Sum(t => t.Quantity * t.Price);
                
            double sellValue = _currentInsiderData
                .Where(t => t.TransactionType == InsiderTransactionType.Sale)
                .Sum(t => t.Quantity * t.Price);
                
            double netValue = buyValue - sellValue;
            
            // Format with K/M/B suffix
            NetInsiderValueText.Text = FormatCurrency(netValue);
            
            if (netValue > 0)
            {
                NetInsiderValueText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green (net buying)
            }
            else if (netValue < 0)
            {
                NetInsiderValueText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red (net selling)
            }
            else
            {
                NetInsiderValueText.Foreground = new SolidColorBrush(Colors.White); // Balanced
            }
            
            // Calculate CEO activity
            var ceoTransactions = _currentInsiderData
                .Where(t => t.InsiderTitle?.Contains("CEO") == true || 
                           t.InsiderTitle?.Contains("Chief Executive") == true)
                .ToList();
                
            if (ceoTransactions.Count > 0)
            {
                double ceoBuyValue = ceoTransactions
                    .Where(t => t.TransactionType == InsiderTransactionType.Purchase)
                    .Sum(t => t.Quantity * t.Price);
                    
                double ceoSellValue = ceoTransactions
                    .Where(t => t.TransactionType == InsiderTransactionType.Sale)
                    .Sum(t => t.Quantity * t.Price);
                    
                double ceoNetValue = ceoBuyValue - ceoSellValue;
                
                if (ceoNetValue > 0)
                {
                    CEOActivityText.Text = "Buying";
                    CEOActivityText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
                }
                else if (ceoNetValue < 0)
                {
                    CEOActivityText.Text = "Selling";
                    CEOActivityText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
                }
                else
                {
                    CEOActivityText.Text = "Neutral";
                    CEOActivityText.Foreground = new SolidColorBrush(Colors.White);
                }
            }
            else
            {
                CEOActivityText.Text = "No Activity";
                CEOActivityText.Foreground = new SolidColorBrush(Colors.White);
            }
            
            // Calculate notable figure activity
            var notableTransactions = _currentInsiderData
                .Where(t => t.IsNotableFigure)
                .ToList();
                
            if (notableTransactions.Count > 0)
            {
                double notableBuyValue = notableTransactions
                    .Where(t => t.TransactionType == InsiderTransactionType.Purchase)
                    .Sum(t => t.Quantity * t.Price);
                    
                double notableSellValue = notableTransactions
                    .Where(t => t.TransactionType == InsiderTransactionType.Sale)
                    .Sum(t => t.Quantity * t.Price);
                    
                double notableNetValue = notableBuyValue - notableSellValue;
                
                if (notableNetValue > 0)
                {
                    NotableFigureActivityText.Text = "Bullish";
                    NotableFigureActivityText.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
                }
                else if (notableNetValue < 0)
                {
                    NotableFigureActivityText.Text = "Bearish";
                    NotableFigureActivityText.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
                }
                else
                {
                    NotableFigureActivityText.Text = "Mixed";
                    NotableFigureActivityText.Foreground = new SolidColorBrush(Colors.White);
                }
            }
            else
            {
                NotableFigureActivityText.Text = "None";
                NotableFigureActivityText.Foreground = new SolidColorBrush(Colors.White);
            }
        }
        
        /// <summary>
        /// Updates the overall sentiment score display
        /// </summary>
        private void UpdateOverallSentiment()
        {
            if (_currentSentimentData == null)
                return;
                
            double overallSentiment = _currentSentimentData.OverallCorrelation;
            
            // Display the overall sentiment
            OverallSentimentTextBlock.Text = $"{overallSentiment:F2}";
            
            // Set color based on sentiment value
            if (overallSentiment > 0.3)
            {
                OverallSentimentTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
            }
            else if (overallSentiment < -0.3)
            {
                OverallSentimentTextBlock.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
            }
            else
            {
                OverallSentimentTextBlock.Foreground = new SolidColorBrush(Colors.Cyan); // Neutral
            }
        }
        
        /// <summary>
        /// Resets all metrics to default values
        /// </summary>
        private void ResetMetrics()
        {
            // Reset header
            OverallSentimentTextBlock.Text = "0.00";
            OverallSentimentTextBlock.Foreground = new SolidColorBrush(Colors.Cyan);
            
            // Reset historical trends
            LeadLagDaysText.Text = "0.0 days";
            LeadLagRelationshipText.Text = "Neutral";
            PredictiveAccuracyText.Text = "0.0%";
            TwitterCorrelationText.Text = "0.00";
            NewsCorrelationText.Text = "0.00";
            AnalystCorrelationText.Text = "0.00";
            InsiderCorrelationText.Text = "0.00";
            OverallCorrelationText.Text = "0.00";
            SentimentTrendText.Text = "Neutral";
            PriceImpactText.Text = "Low";
            
            // Reset analyst ratings
            ConsensusRatingText.Text = "--";
            BuyCountText.Text = "0";
            HoldCountText.Text = "0";
            SellCountText.Text = "0";
            AvgPriceTargetText.Text = "$0.00";
            TargetPctChangeText.Text = "";
            PriceTargetRangeText.Text = "$0.00 - $0.00";
            
            // Reset insider trading
            BuySellRatioText.Text = "0.0";
            NetInsiderValueText.Text = "$0";
            CEOActivityText.Text = "No Activity";
            NotableFigureActivityText.Text = "None";
            
            // Reset all text colors to default
            var defaultColor = new SolidColorBrush(Colors.White);
            LeadLagRelationshipText.Foreground = defaultColor;
            PredictiveAccuracyText.Foreground = defaultColor;
            TwitterCorrelationText.Foreground = defaultColor;
            NewsCorrelationText.Foreground = defaultColor;
            AnalystCorrelationText.Foreground = defaultColor;
            InsiderCorrelationText.Foreground = defaultColor;
            OverallCorrelationText.Foreground = defaultColor;
            SentimentTrendText.Foreground = defaultColor;
            PriceImpactText.Foreground = defaultColor;
            ConsensusRatingText.Foreground = defaultColor;
            BuySellRatioText.Foreground = defaultColor;
            NetInsiderValueText.Foreground = defaultColor;
            CEOActivityText.Foreground = defaultColor;
            NotableFigureActivityText.Foreground = defaultColor;
        }
        #endregion
        
        #region Helper Methods
        /// <summary>
        /// Updates the lookback timeframe based on UI selection
        /// </summary>
        private void UpdateTimeframeFromUI()
        {
            if (TrendTimeframeComboBox.SelectedItem != null)
            {
                string selectedItem = (TrendTimeframeComboBox.SelectedItem as ComboBoxItem).Content.ToString();
                switch (selectedItem)
                {
                    case "7 Days":
                        _currentLookbackDays = 7;
                        break;
                    case "30 Days":
                        _currentLookbackDays = 30;
                        break;
                    case "90 Days":
                        _currentLookbackDays = 90;
                        break;
                    case "1 Year":
                        _currentLookbackDays = 365;
                        break;
                    default:
                        _currentLookbackDays = 30;
                        break;
                }
            }
        }
        
        /// <summary>
        /// Updates a correlation text block with data from source correlations
        /// </summary>
        private void UpdateCorrelationText(string source, TextBlock textBlock, Dictionary<string, double> correlations)
        {
            if (correlations.TryGetValue(source, out double correlation))
            {
                textBlock.Text = $"{correlation:F2}";
                
                if (correlation > 0.5)
                {
                    textBlock.Foreground = new SolidColorBrush(Color.FromRgb(32, 192, 64)); // Green
                }
                else if (correlation < -0.5)
                {
                    textBlock.Foreground = new SolidColorBrush(Color.FromRgb(192, 32, 32)); // Red
                }
                else
                {
                    textBlock.Foreground = new SolidColorBrush(Colors.Cyan);
                }
            }
            else
            {
                textBlock.Text = "N/A";
                textBlock.Foreground = new SolidColorBrush(Colors.Gray);
            }
        }
        
        /// <summary>
        /// Gets notes for a sentiment shift event
        /// </summary>
        private string GetEventNotes(Quantra.Modules.SentimentShiftEvent evt)
        {
            if (evt == null)
                return string.Empty;
                
            string priceBehavior = evt.PriceFollowedSentiment ? "confirmed" : "contradicted";
            
            switch (evt.Source)
            {
                case "Twitter":
                    return $"Social media sentiment {evt.SentimentShift:F2} shift {priceBehavior} by price movement";
                case "News":
                    return $"News sentiment {evt.SentimentShift:F2} shift {priceBehavior} by price movement";
                case "AnalystRatings":
                    return $"Analyst sentiment {evt.SentimentShift:F2} shift {priceBehavior} by price movement";
                case "InsiderTrading":
                    return $"Insider trading sentiment {evt.SentimentShift:F2} shift {priceBehavior} by price movement";
                default:
                    return $"Sentiment {evt.SentimentShift:F2} shift {priceBehavior} by price movement";
            }
        }
        
        /// <summary>
        /// Gets an estimated current price for price target comparison
        /// In a real implementation, this would come from market data
        /// </summary>
        private double GetEstimatedCurrentPrice()
        {
            // In a real implementation, we'd get current price from a market data service
            // For now, use the lowest price target as a conservative approach
            if (_currentAnalystData != null && _currentAnalystData.LowestPriceTarget > 0)
            {
                return _currentAnalystData.LowestPriceTarget * 0.9; // 90% of lowest target as an estimate
            }
            
            return 0;
        }
        
        /// <summary>
        /// Formats a currency value with K/M/B suffix for readability
        /// </summary>
        private string FormatCurrency(double value)
        {
            string sign = value < 0 ? "-" : "";
            value = Math.Abs(value);
            
            if (value >= 1_000_000_000)
            {
                return $"{sign}${value / 1_000_000_000:F1}B";
            }
            if (value >= 1_000_000)
            {
                return $"{sign}${value / 1_000_000:F1}M";
            }
            if (value >= 1_000)
            {
                return $"{sign}${value / 1_000:F0}K";
            }
            
            return $"{sign}${value:F0}";
        }
        #endregion
    }
}