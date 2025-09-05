using LiveCharts;
using LiveCharts.Configurations;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using Quantra.Models;
using System.Collections.ObjectModel;
using System.Text.Json;
using Quantra.DAL.Services.Interfaces;
using Quantra.CrossCutting.Monitoring;
using Quantra.DAL.Services;

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for TechnicalIndicatorVisualizationControl.xaml
    /// </summary>
    public partial class TechnicalIndicatorVisualizationControl : UserControl, INotifyPropertyChanged
    {
        #region Fields and Properties

        // Services
        private readonly ITechnicalIndicatorService _indicatorService;
        private readonly IStockDataCacheService _stockDataService;
        private readonly IMonitoringManager _monitoringManager;
        
        // Chart data collections
        public ChartValues<OhlcPoint> CandleValues { get; } = new ChartValues<OhlcPoint>();
        public ChartValues<double> PriceValues { get; } = new ChartValues<double>();
        public ChartValues<double> VolumeValues { get; } = new ChartValues<double>();
        
        // SeriesCollection for PriceChart binding
        private SeriesCollection _priceSeries = new SeriesCollection();
        public SeriesCollection PriceSeries
        {
            get => _priceSeries;
            set
            {
                _priceSeries = value;
                OnPropertyChanged(nameof(PriceSeries));
            }
        }
        
        // Bollinger Bands
        public ChartValues<double> UpperBandValues { get; } = new ChartValues<double>();
        public ChartValues<double> MiddleBandValues { get; } = new ChartValues<double>();
        public ChartValues<double> LowerBandValues { get; } = new ChartValues<double>();
        
        // Moving Averages
        public ChartValues<double> SMA20Values { get; } = new ChartValues<double>();
        public ChartValues<double> SMA50Values { get; } = new ChartValues<double>();
        public ChartValues<double> SMA200Values { get; } = new ChartValues<double>();
        public ChartValues<double> EMA12Values { get; } = new ChartValues<double>();
        public ChartValues<double> EMA26Values { get; } = new ChartValues<double>();
        
        // VWAP
        public ChartValues<double> VWAPValues { get; } = new ChartValues<double>();
        
        // Oscillators
        public ChartValues<double> RSIValues { get; } = new ChartValues<double>();
        public ChartValues<double> MACDValues { get; } = new ChartValues<double>();
        public ChartValues<double> MACDSignalValues { get; } = new ChartValues<double>();
        public ChartValues<double> MACDHistogramValues { get; } = new ChartValues<double>();
        public ChartValues<double> StochKValues { get; } = new ChartValues<double>();
        public ChartValues<double> StochDValues { get; } = new ChartValues<double>();
        public ChartValues<double> StochRSIValues { get; } = new ChartValues<double>();
        public ChartValues<double> WilliamsRValues { get; } = new ChartValues<double>();
        public ChartValues<double> CCIValues { get; } = new ChartValues<double>();
        
        // Data labels
        public List<string> DateLabels { get; private set; } = new List<string>();
        
        // Currently displayed stock
        private string _currentSymbol;
        public string CurrentSymbol
        {
            get => _currentSymbol;
            set
            {
                if (_currentSymbol != value)
                {
                    _currentSymbol = value;
                    OnPropertyChanged(nameof(CurrentSymbol));
                    LoadDataForSymbol(_currentSymbol);
                }
            }
        }
        
        // Chart state
        private bool _isLoading = false;
        public bool IsLoading
        {
            get => _isLoading;
            set
            {
                if (_isLoading != value)
                {
                    _isLoading = value;
                    OnPropertyChanged(nameof(IsLoading));
                    LoadingOverlay.Visibility = _isLoading ? Visibility.Visible : Visibility.Collapsed;
                }
            }
        }
        
        // Chart formatting
        public Func<double, string> DateTimeFormatter { get; set; }

        // Dictionary of dynamic oscillator panels
        private Dictionary<string, (Grid Panel, CartesianChart Chart)> _oscillatorPanels = 
            new Dictionary<string, (Grid Panel, CartesianChart Chart)>();
            
        // Dictionary to track which overlay indicators are currently displayed
        private Dictionary<string, bool> _overlayIndicatorsVisible = new Dictionary<string, bool>
        {
            { "BB", false },   // Bollinger Bands
            { "MA", false },   // Moving Averages
            { "VWAP", false }, // VWAP
        };
        
        // Current mouse position for crosshair
        private Point _currentMousePosition;
        private bool _crosshairEnabled = true;
        
        // Dictionary of crosshair lines for each chart
        private Dictionary<CartesianChart, (Line Horizontal, Line Vertical)> _crosshairLines = 
            new Dictionary<CartesianChart, (Line Horizontal, Line Vertical)>();
            
        #endregion

        #region Initialization
        
        public TechnicalIndicatorVisualizationControl()
        {
            InitializeComponent();
            
            // Get services
            _indicatorService = ServiceLocator.GetService<ITechnicalIndicatorService>();
            _stockDataService = ServiceLocator.GetService<IStockDataCacheService>();
            _monitoringManager = MonitoringManager.Instance;
            
            // Set data context
            DataContext = this;
            
            // Initialize crosshair lines for main charts
            InitializeCrosshairLines(PriceChart, PriceCrosshairCanvas);
            InitializeCrosshairLines(VolumeChart, VolumeCrosshairCanvas);
            
            // Set default formatters
            DateTimeFormatter = value =>
            {
                int index = (int)value;
                if (index >= 0 && index < DateLabels.Count)
                    return DateLabels[index];
                return string.Empty;
            };
        }
        
        private void InitializeCrosshairLines(CartesianChart chart, Canvas canvas)
        {
            // Create horizontal line
            var horizontalLine = new Line
            {
                Stroke = new SolidColorBrush(Color.FromArgb(120, 255, 255, 255)),
                StrokeThickness = 1,
                X1 = 0,
                Y1 = 0,
                X2 = 0,
                Y2 = 0,
                Visibility = Visibility.Collapsed
            };
            
            // Create vertical line
            var verticalLine = new Line
            {
                Stroke = new SolidColorBrush(Color.FromArgb(120, 255, 255, 255)),
                StrokeThickness = 1,
                X1 = 0,
                Y1 = 0,
                X2 = 0,
                Y2 = 0,
                Visibility = Visibility.Collapsed
            };
            
            // Add lines to canvas
            canvas.Children.Add(horizontalLine);
            canvas.Children.Add(verticalLine);
            
            // Store lines in dictionary
            _crosshairLines[chart] = (horizontalLine, verticalLine);
        }
        
        #endregion

        #region Data Loading
        
        private async void LoadDataForSymbol(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return;
                
            IsLoading = true;
            
            try
            {
                // Update symbol display
                SymbolTextBlock.Text = $"Symbol: {symbol}";
                
                // Clear existing data
                ClearChartData();
                
                // Get data from service (implement proper async/await pattern)
                var data = await _stockDataService.GetStockDataAsync(symbol);
                if (data == null || data.Count == 0)
                {
                    MessageBox.Show($"No data available for {symbol}", "Data Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }
                
                // Update data collections
                PopulateChartData(data);
                
                // Calculate and populate indicators
                PopulateIndicatorData(data);
                
                // Update price display
                var latestPrice = data.Last().Close;
                var previousPrice = data.Count > 1 ? data[data.Count - 2].Close : latestPrice;
                var priceChange = latestPrice - previousPrice;
                var percentChange = previousPrice != 0 ? priceChange / previousPrice * 100 : 0;
                
                PriceTextBlock.Text = $"Price: ${latestPrice:F2}";
                ChangeTextBlock.Text = $"{priceChange:+0.00;-0.00;0.00} ({percentChange:+0.00;-0.00;0.00}%)";
                ChangeTextBlock.Foreground = priceChange >= 0 ? 
                    new SolidColorBrush(Color.FromRgb(76, 175, 80)) :  // Green
                    new SolidColorBrush(Color.FromRgb(244, 67, 54));   // Red
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading data for {symbol}: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                IsLoading = false;
            }
        }
        
        private void ClearChartData()
        {
            // Clear all data collections
            CandleValues.Clear();
            PriceValues.Clear();
            VolumeValues.Clear();
            UpperBandValues.Clear();
            MiddleBandValues.Clear();
            LowerBandValues.Clear();
            SMA20Values.Clear();
            SMA50Values.Clear();
            SMA200Values.Clear();
            EMA12Values.Clear();
            EMA26Values.Clear();
            VWAPValues.Clear();
            RSIValues.Clear();
            MACDValues.Clear();
            MACDSignalValues.Clear();
            MACDHistogramValues.Clear();
            StochKValues.Clear();
            StochDValues.Clear();
            StochRSIValues.Clear();
            WilliamsRValues.Clear();
            CCIValues.Clear();
            DateLabels.Clear();
        }
        
        private void PopulateChartData(List<HistoricalPrice> data)
        {
            // Populate date labels
            DateLabels = data.Select(item => item.Date.ToString("MM/dd")).ToList();
            
            // Populate price data
            foreach (var item in data)
            {
                // Add candle data
                CandleValues.Add(new OhlcPoint(item.Open, item.High, item.Low, item.Close));
                
                // Add line chart data
                PriceValues.Add(item.Close);
                
                // Add volume data
                VolumeValues.Add(item.Volume);
            }
        }
        
        private void PopulateIndicatorData(List<HistoricalPrice> data)
        {
            _monitoringManager.RecordExecutionTime($"PopulateIndicatorData_{data?.Count ?? 0}_items", () =>
            {
                // Extract relevant price data
                var closePrices = data.Select(item => item.Close).ToList();
                var highPrices = data.Select(item => item.High).ToList();
                var lowPrices = data.Select(item => item.Low).ToList();
                var openPrices = data.Select(item => item.Open).ToList();
                var volumes = data.Select(item => item.Volume).ToList();
                
                // Calculate indicators using the service with timing
                // Note: In a real implementation, these would call the actual indicator service methods
                
                // Bollinger Bands (20-period, 2 standard deviations)
                var (bollingerBands, bbDuration) = _monitoringManager.RecordExecutionTime("CalculateBollingerBands", 
                    () => _indicatorService.CalculateBollingerBands(closePrices, 20, 2));
                UpperBandValues.AddRange(bollingerBands.Upper);
                MiddleBandValues.AddRange(bollingerBands.Middle);
                LowerBandValues.AddRange(bollingerBands.Lower);
                
                // Moving Averages
                var (sma20, sma20Duration) = _monitoringManager.RecordExecutionTime("CalculateSMA20", 
                    () => _indicatorService.CalculateSMA(closePrices, 20));
                var (sma50, sma50Duration) = _monitoringManager.RecordExecutionTime("CalculateSMA50", 
                    () => _indicatorService.CalculateSMA(closePrices, 50));
                var (sma200, sma200Duration) = _monitoringManager.RecordExecutionTime("CalculateSMA200", 
                    () => _indicatorService.CalculateSMA(closePrices, 200));
                var (ema12, ema12Duration) = _monitoringManager.RecordExecutionTime("CalculateEMA12", 
                    () => _indicatorService.CalculateEMA(closePrices, 12));
                var (ema26, ema26Duration) = _monitoringManager.RecordExecutionTime("CalculateEMA26", 
                    () => _indicatorService.CalculateEMA(closePrices, 26));
                
                SMA20Values.AddRange(sma20);
                SMA50Values.AddRange(sma50);
                SMA200Values.AddRange(sma200);
                EMA12Values.AddRange(ema12);
                EMA26Values.AddRange(ema26);
                
                // VWAP
                var (vwap, vwapDuration) = _monitoringManager.RecordExecutionTime("CalculateVWAP", 
                    () => _indicatorService.CalculateVWAP(highPrices, lowPrices, closePrices, volumes.Select(v => (double)v).ToList()));
                VWAPValues.AddRange(vwap);
                
                // RSI (14-period)
                var (rsi, rsiDuration) = _monitoringManager.RecordExecutionTime("CalculateRSI", 
                    () => _indicatorService.CalculateRSI(closePrices, 14));
                RSIValues.AddRange(rsi);
                
                // MACD (12, 26, 9)
                var (macd, macdDuration) = _monitoringManager.RecordExecutionTime("CalculateMACD", 
                    () => _indicatorService.CalculateMACD(closePrices, 12, 26, 9));
                MACDValues.AddRange(macd.MacdLine);
                MACDSignalValues.AddRange(macd.SignalLine);
                MACDHistogramValues.AddRange(macd.Histogram);
                
                // Stochastic Oscillator (14, 3, 3)
                var (stoch, stochDuration) = _monitoringManager.RecordExecutionTime("CalculateStochastic", 
                    () => _indicatorService.CalculateStochastic(highPrices, lowPrices, closePrices, 14, 3, 3));
                StochKValues.AddRange(stoch.K);
                StochDValues.AddRange(stoch.D);
                
                // Stochastic RSI
                var (stochRsi, stochRsiDuration) = _monitoringManager.RecordExecutionTime("CalculateStochRSI", 
                    () => _indicatorService.CalculateStochRSI(closePrices, 14, 14, 3, 3));
                StochRSIValues.AddRange(stochRsi);
                
                // Williams %R
                var (williamsR, williamsRDuration) = _monitoringManager.RecordExecutionTime("CalculateWilliamsR", 
                    () => _indicatorService.CalculateWilliamsR(highPrices, lowPrices, closePrices, 14));
                WilliamsRValues.AddRange(williamsR);
                
                // CCI
                var (cci, cciDuration) = _monitoringManager.RecordExecutionTime("CalculateCCI", 
                    () => _indicatorService.CalculateCCI(highPrices, lowPrices, closePrices, 20));
                CCIValues.AddRange(cci);
            });
        }
        
        #endregion

        #region Chart Interaction and UI Events
        
        private void TimeRangeButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string timeRange)
            {
                // Update time range
                // For demo purposes, just show a message
                MessageBox.Show($"Would load {timeRange} data for {CurrentSymbol}", "Time Range Changed", MessageBoxButton.OK, MessageBoxImage.Information);
                
                // In a real implementation, you would:
                // 1. Call the data service with the new time range
                // 2. Update the chart data with the new timeframe
            }
        }
        
        private void ToggleLegend_Checked(object sender, RoutedEventArgs e)
        {
            // Show legends for all charts
            if (PriceChart != null)
                PriceChart.LegendLocation = LegendLocation.Right;
            if (VolumeChart != null)
                VolumeChart.LegendLocation = LegendLocation.Right;
            
            // Show legends for oscillator charts
            foreach (var panel in _oscillatorPanels)
            {
                if (panel.Value.Chart != null)
                    panel.Value.Chart.LegendLocation = LegendLocation.Right;
            }
        }
        
        private void ToggleLegend_Unchecked(object sender, RoutedEventArgs e)
        {
            // Hide legends for all charts
            if (PriceChart != null)
                PriceChart.LegendLocation = LegendLocation.None;
            if (VolumeChart != null)
                VolumeChart.LegendLocation = LegendLocation.None;
            
            // Hide legends for oscillator charts
            foreach (var panel in _oscillatorPanels)
            {
                if (panel.Value.Chart != null)
                    panel.Value.Chart.LegendLocation = LegendLocation.None;
            }
        }
        
        private void ToggleCrosshair_Checked(object sender, RoutedEventArgs e)
        {
            _crosshairEnabled = true;
            // Show existing crosshair if mouse is over a chart
            UpdateCrosshair();
        }
        
        private void ToggleCrosshair_Unchecked(object sender, RoutedEventArgs e)
        {
            _crosshairEnabled = false;
            // Hide all crosshairs
            HideAllCrosshairs();
        }
        
        private void Chart_MouseMove(object sender, MouseEventArgs e)
        {
            if (sender is CartesianChart chart)
            {
                _currentMousePosition = e.GetPosition(chart);
                
                if (_crosshairEnabled)
                {
                    UpdateCrosshair(chart);
                    
                    // Synchronize crosshair across all charts if enabled
                    if (SyncTooltips.IsChecked == true)
                    {
                        SynchronizeCrosshair(chart, _currentMousePosition);
                    }
                }
            }
        }
        
        private void UpdateCrosshair(CartesianChart chart = null)
        {
            if (chart != null && _crosshairLines.ContainsKey(chart))
            {
                var (horizontal, vertical) = _crosshairLines[chart];
                
                // Update crosshair position
                horizontal.X1 = 0;
                horizontal.Y1 = _currentMousePosition.Y;
                horizontal.X2 = chart.ActualWidth;
                horizontal.Y2 = _currentMousePosition.Y;
                
                vertical.X1 = _currentMousePosition.X;
                vertical.Y1 = 0;
                vertical.X2 = _currentMousePosition.X;
                vertical.Y2 = chart.ActualHeight;
                
                // Show crosshair
                horizontal.Visibility = Visibility.Visible;
                vertical.Visibility = Visibility.Visible;
            }
        }
        
        private void SynchronizeCrosshair(CartesianChart sourceChart, Point mousePosition)
        {
            // Calculate relative X position (as percentage of chart width)
            double relativeX = mousePosition.X / sourceChart.ActualWidth;
            
            // Update all other charts with the same relative X position
            foreach (var entry in _crosshairLines)
            {
                if (entry.Key != sourceChart)
                {
                    var (horizontal, vertical) = entry.Value;
                    
                    // Calculate absolute position for this chart
                    double absoluteX = relativeX * entry.Key.ActualWidth;
                    
                    // Update vertical line position
                    vertical.X1 = absoluteX;
                    vertical.X2 = absoluteX;
                    
                    // Show the vertical line only (horizontal line stays at the mouse position)
                    vertical.Visibility = Visibility.Visible;
                }
            }
        }
        
        private void HideAllCrosshairs()
        {
            foreach (var entry in _crosshairLines)
            {
                entry.Value.Horizontal.Visibility = Visibility.Collapsed;
                entry.Value.Vertical.Visibility = Visibility.Collapsed;
            }
        }
        
        private void ChartSettings_Click(object sender, RoutedEventArgs e)
        {
            // Identify which chart's settings button was clicked
            string chartName = "Unknown";
            if (sender == PriceChartSettingsButton)
                chartName = "Price Chart";
            else if (sender == VolumeChartSettingsButton)
                chartName = "Volume Chart";
            
            // Show settings dialog (placeholder for now)
            MessageBox.Show($"Settings for {chartName}", "Chart Settings", MessageBoxButton.OK, MessageBoxImage.Information);
            
            // In a real implementation, you would:
            // 1. Open a settings dialog for the specific chart
            // 2. Allow customization of colors, line thickness, etc.
        }
        
        private void AddIndicator_Click(object sender, RoutedEventArgs e)
        {
            // Show indicator selection UI (placeholder for now)
            MessageBox.Show("Add indicator to chart", "Add Indicator", MessageBoxButton.OK, MessageBoxImage.Information);
            
            // In a real implementation, you would:
            // 1. Show a popup with available indicators
            // 2. Allow the user to select indicators to add to the chart
        }
        
        private void AddIndicatorPanel_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string indicatorType)
            {
                // Check if this indicator panel already exists
                if (_oscillatorPanels.ContainsKey(indicatorType))
                {
                    // Remove existing panel
                    RemoveIndicatorPanel(indicatorType);
                }
                else
                {
                    // Add new indicator panel
                    AddOscillatorPanel(indicatorType);
                }
            }
        }
        
        private void RemovePanel_Click(object sender, RoutedEventArgs e)
        {
            // Determine which panel's remove button was clicked
            if (sender == RemoveVolumeButton)
            {
                // Toggle volume chart visibility
                VolumeChart.Visibility = VolumeChart.Visibility == Visibility.Visible ? 
                    Visibility.Collapsed : Visibility.Visible;
                
                // Update button content
                RemoveVolumeButton.Content = VolumeChart.Visibility == Visibility.Visible ? "✕" : "+";
            }
            else if (sender is Button button && button.Tag is string indicatorType)
            {
                // Remove the indicator panel
                RemoveIndicatorPanel(indicatorType);
            }
        }
        
        private void ToggleOverlayIndicator_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string indicatorType)
            {
                // Toggle this overlay indicator
                if (_overlayIndicatorsVisible.ContainsKey(indicatorType))
                {
                    bool isVisible = !_overlayIndicatorsVisible[indicatorType];
                    _overlayIndicatorsVisible[indicatorType] = isVisible;
                    
                    // Update the chart display
                    UpdateOverlayIndicatorVisibility(indicatorType, isVisible);
                    
                    // Update button appearance
                    button.Background = isVisible ? 
                        new SolidColorBrush(Color.FromArgb(100, 100, 100, 255)) : 
                        new SolidColorBrush(Colors.Transparent);
                }
            }
        }
        
        private void CustomizeLayout_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Get current layout configuration
                var currentLayout = GetCurrentLayoutConfig();
                
                // Create and show the customize layout dialog
                var dialog = new Quantra.Views.CustomizeLayoutDialog(currentLayout)
                {
                    Owner = Window.GetWindow(this)
                };
                
                if (dialog.ShowDialog() == true && dialog.ResultLayout != null)
                {
                    // Apply the new layout configuration
                    ApplyLayoutConfig(dialog.ResultLayout);
                    
                    // Save the layout to persistence
                    SaveLayoutConfig(dialog.ResultLayout);
                }
            }
            catch (Exception ex)
            {
                // Log error and show user-friendly message
                DatabaseMonolith.Log("Error", $"Failed to open layout customization dialog: {ex.Message}", ex.ToString());
                MessageBox.Show("Failed to open layout customization. Please check the logs for details.", 
                    "Layout Customization Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }
        
        /// <summary>
        /// Gets the current layout configuration from the control state
        /// </summary>
        /// <returns>Current layout configuration</returns>
        private LayoutConfig GetCurrentLayoutConfig()
        {
            var config = new LayoutConfig
            {
                LayoutName = "Current",
                TotalRows = ChartsGrid.RowDefinitions.Count,
                TotalColumns = 1,
                ShowGridLines = true,
                GridLineColor = "#FF00FFFF"
            };
            
            // Add existing panels
            config.Panels.Add(new ChartPanelLayout
            {
                PanelId = "Price",
                DisplayName = "Price Chart",
                Row = 0,
                Column = 0,
                RowSpan = 3,
                ColumnSpan = 1,
                HeightRatio = 3.0,
                IsVisible = true,
                DisplayOrder = 1
            });
            
            config.Panels.Add(new ChartPanelLayout
            {
                PanelId = "Volume",
                DisplayName = "Volume",
                Row = 1,
                Column = 0,
                RowSpan = 1,
                ColumnSpan = 1,
                HeightRatio = 1.0,
                IsVisible = true,
                DisplayOrder = 2
            });
            
            // Add any additional oscillator panels that exist
            for (int i = 2; i < ChartsGrid.RowDefinitions.Count; i++)
            {
                config.Panels.Add(new ChartPanelLayout
                {
                    PanelId = $"Oscillator{i}",
                    DisplayName = $"Oscillator Panel {i}",
                    Row = i,
                    Column = 0,
                    RowSpan = 1,
                    ColumnSpan = 1,
                    HeightRatio = 1.0,
                    IsVisible = true,
                    DisplayOrder = i + 1
                });
            }
            
            return config;
        }
        
        /// <summary>
        /// Applies a layout configuration to the control
        /// </summary>
        /// <param name="layout">Layout configuration to apply</param>
        private void ApplyLayoutConfig(LayoutConfig layout)
        {
            try
            {
                // Update grid dimensions if needed
                while (ChartsGrid.RowDefinitions.Count < layout.TotalRows)
                {
                    ChartsGrid.RowDefinitions.Add(new RowDefinition 
                    { 
                        Height = new GridLength(1, GridUnitType.Star),
                        MinHeight = 100 
                    });
                }
                
                // Apply panel-specific configurations
                foreach (var panel in layout.Panels.Where(p => p.IsVisible))
                {
                    ApplyPanelConfig(panel);
                }
                
                // Update visual settings
                // Note: Grid line visualization would be implemented here
                // if we had a grid overlay system
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to apply layout configuration: {ex.Message}", ex.ToString());
                throw;
            }
        }
        
        /// <summary>
        /// Applies configuration for a specific panel
        /// </summary>
        /// <param name="panel">Panel configuration</param>
        private void ApplyPanelConfig(ChartPanelLayout panel)
        {
            // This would implement the actual panel positioning logic
            // For now, we'll just ensure the basic panels exist
            switch (panel.PanelId)
            {
                case "RSI":
                case "MACD":
                case "StochRSI":
                case "Williams %R":
                case "CCI":
                    // Add oscillator panel if it doesn't exist
                    if (!_oscillatorPanels.ContainsKey(panel.PanelId))
                    {
                        AddOscillatorPanel(panel.PanelId);
                    }
                    break;
            }
        }
        
        /// <summary>
        /// Saves layout configuration to persistence
        /// </summary>
        /// <param name="layout">Layout to save</param>
        private void SaveLayoutConfig(LayoutConfig layout)
        {
            try
            {
                // Convert layout to JSON for storage
                var json = JsonSerializer.Serialize(layout, new JsonSerializerOptions 
                { 
                    WriteIndented = true 
                });
                
                // Save to UIConfig (this would be persisted by the configuration system)
                // Note: This would require access to the configuration manager
                // For now, we'll just log that it would be saved
                DatabaseMonolith.Log("Info", $"Layout '{layout.LayoutName}' configuration saved", json);
                
                // In a full implementation, you would:
                // 1. Get the configuration manager instance
                // 2. Update the UIConfig.ChartLayoutConfig property
                // 3. Save the configuration
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to save layout configuration: {ex.Message}", ex.ToString());
            }
        }
        
        #endregion

        #region Chart Panel Management
        
        private void AddOscillatorPanel(string indicatorType)
        {
            // Create new row definition for the oscillator panel
            var rowDefinition = new RowDefinition
            {
                Height = new GridLength(1, GridUnitType.Star),
                MinHeight = 100
            };
            
            ChartsGrid.RowDefinitions.Add(rowDefinition);
            int rowIndex = ChartsGrid.RowDefinitions.Count - 1;
            
            // Create container for the oscillator chart
            var container = new Grid();
            
            // Create the chart control
            var chart = new CartesianChart
            {
                DisableAnimations = false,
                AnimationsSpeed = TimeSpan.FromMilliseconds(500),
                Zoom = ZoomingOptions.X,
                Hoverable = true
            };
            
            chart.MouseMove += Chart_MouseMove;
            
            // Configure the chart based on indicator type
            ConfigureOscillatorChart(chart, indicatorType);
            
            // Add chart to container
            container.Children.Add(chart);
            
            // Create crosshair canvas
            var crosshairCanvas = new Canvas
            {
                IsHitTestVisible = false
            };
            container.Children.Add(crosshairCanvas);
            
            // Initialize crosshair lines
            InitializeCrosshairLines(chart, crosshairCanvas);
            
            // Create control panel
            var controlPanel = new StackPanel
            {
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Top,
                Margin = new Thickness(0, 5, 5, 0)
            };
            
            // Add settings button
            var settingsButton = new Button
            {
                Content = "⚙️",
                Width = 24,
                Height = 24,
                Margin = new Thickness(0, 0, 0, 5)
            };
            settingsButton.Click += ChartSettings_Click;
            controlPanel.Children.Add(settingsButton);
            
            // Add remove button
            var removeButton = new Button
            {
                Content = "✕",
                Width = 24,
                Height = 24,
                Tag = indicatorType
            };
            removeButton.Click += RemovePanel_Click;
            controlPanel.Children.Add(removeButton);
            
            // Add control panel to container
            container.Children.Add(controlPanel);
            
            // Add container to grid
            Grid.SetRow(container, rowIndex);
            ChartsGrid.Children.Add(container);
            
            // Store reference to the panel
            _oscillatorPanels[indicatorType] = (container, chart);
        }
        
        private void ConfigureOscillatorChart(CartesianChart chart, string indicatorType)
        {
            // Configure chart based on indicator type
            chart.Series = new SeriesCollection();
            
            switch (indicatorType)
            {
                case "RSI":
                    chart.Series.Add(new LineSeries
                    {
                        Title = "RSI",
                        Values = RSIValues,
                        Stroke = new SolidColorBrush(Colors.Orange),
                        Fill = new SolidColorBrush(Color.FromArgb(20, 255, 165, 0)),
                        PointGeometrySize = 0
                    });
                    
                    // Add threshold lines
                    chart.Series.Add(new LineSeries
                    {
                        Title = "Overbought (70)",
                        Values = new ChartValues<double> { 70, 70 },
                        Stroke = new SolidColorBrush(Colors.Red),
                        Fill = Brushes.Transparent,
                        PointGeometrySize = 0,
                        LineSmoothness = 0,
                        StrokeDashArray = new DoubleCollection(new[] { 3d, 3d })
                    });
                    
                    chart.Series.Add(new LineSeries
                    {
                        Title = "Oversold (30)",
                        Values = new ChartValues<double> { 30, 30 },
                        Stroke = new SolidColorBrush(Colors.Green),
                        Fill = Brushes.Transparent,
                        PointGeometrySize = 0,
                        LineSmoothness = 0,
                        StrokeDashArray = new DoubleCollection(new[] { 3d, 3d })
                    });
                    
                    // Configure axes
                    chart.AxisX = new AxesCollection
                    {
                        new Axis
                        {
                            ShowLabels = false,
                            MinValue = 0,
                            LabelFormatter = DateTimeFormatter,
                            Separator = new LiveCharts.Wpf.Separator
                            {
                                StrokeThickness = 1,
                                Stroke = new SolidColorBrush(Color.FromArgb(48, 255, 255, 255)),
                                Step = 1
                            }
                        }
                    };
                    
                    chart.AxisY = new AxesCollection
                    {
                        new Axis
                        {
                            Title = "RSI",
                            MinValue = 0,
                            MaxValue = 100,
                            Foreground = Brushes.White,
                            Separator = new LiveCharts.Wpf.Separator
                            {
                                StrokeThickness = 1,
                                Stroke = new SolidColorBrush(Color.FromArgb(48, 255, 255, 255))
                            }
                        }
                    };
                    break;
                    
                case "MACD":
                    chart.Series.Add(new LineSeries
                    {
                        Title = "MACD",
                        Values = MACDValues,
                        Stroke = new SolidColorBrush(Colors.Blue),
                        Fill = Brushes.Transparent,
                        PointGeometrySize = 0
                    });
                    
                    chart.Series.Add(new LineSeries
                    {
                        Title = "Signal",
                        Values = MACDSignalValues,
                        Stroke = new SolidColorBrush(Colors.Red),
                        Fill = Brushes.Transparent,
                        PointGeometrySize = 0
                    });
                    
                    chart.Series.Add(new ColumnSeries
                    {
                        Title = "Histogram",
                        Values = MACDHistogramValues,
                        Fill = new SolidColorBrush(Colors.Green)
                    });
                    
                    // Add zero line
                    chart.Series.Add(new LineSeries
                    {
                        Title = "Zero Line",
                        Values = new ChartValues<double> { 0, 0 },
                        Stroke = new SolidColorBrush(Color.FromArgb(120, 255, 255, 255)),
                        Fill = Brushes.Transparent,
                        PointGeometrySize = 0,
                        LineSmoothness = 0,
                        StrokeDashArray = new DoubleCollection(new[] { 3d, 3d })
                    });
                    
                    // Configure axes
                    chart.AxisX = new AxesCollection
                    {
                        new Axis
                        {
                            ShowLabels = false,
                            MinValue = 0,
                            LabelFormatter = DateTimeFormatter,
                            Separator = new LiveCharts.Wpf.Separator
                            {
                                StrokeThickness = 1,
                                Stroke = new SolidColorBrush(Color.FromArgb(48, 255, 255, 255)),
                                Step = 1
                            }
                        }
                    };
                    
                    chart.AxisY = new AxesCollection
                    {
                        new Axis
                        {
                            Title = "MACD",
                            Foreground = Brushes.White,
                            Separator = new LiveCharts.Wpf.Separator
                            {
                                StrokeThickness = 1,
                                Stroke = new SolidColorBrush(Color.FromArgb(48, 255, 255, 255))
                            }
                        }
                    };
                    break;
                    
                case "STOCHRSI":
                    chart.Series.Add(new LineSeries
                    {
                        Title = "StochRSI",
                        Values = StochRSIValues,
                        Stroke = new SolidColorBrush(Colors.LightCoral),
                        Fill = new SolidColorBrush(Color.FromArgb(20, 240, 128, 128)),
                        PointGeometrySize = 0
                    });
                    
                    // Add threshold lines
                    chart.Series.Add(new LineSeries
                    {
                        Title = "Overbought (0.8)",
                        Values = new ChartValues<double> { 0.8, 0.8 },
                        Stroke = new SolidColorBrush(Colors.Red),
                        Fill = Brushes.Transparent,
                        PointGeometrySize = 0,
                        LineSmoothness = 0,
                        StrokeDashArray = new DoubleCollection(new[] { 3d, 3d })
                    });
                    
                    chart.Series.Add(new LineSeries
                    {
                        Title = "Oversold (0.2)",
                        Values = new ChartValues<double> { 0.2, 0.2 },
                        Stroke = new SolidColorBrush(Colors.Green),
                        Fill = Brushes.Transparent,
                        PointGeometrySize = 0,
                        LineSmoothness = 0,
                        StrokeDashArray = new DoubleCollection(new[] { 3d, 3d })
                    });
                    
                    // Configure axes
                    chart.AxisX = new AxesCollection
                    {
                        new Axis
                        {
                            ShowLabels = false,
                            MinValue = 0,
                            LabelFormatter = DateTimeFormatter,
                            Separator = new LiveCharts.Wpf.Separator
                            {
                                StrokeThickness = 1,
                                Stroke = new SolidColorBrush(Color.FromArgb(48, 255, 255, 255)),
                                Step = 1
                            }
                        }
                    };
                    
                    chart.AxisY = new AxesCollection
                    {
                        new Axis
                        {
                            Title = "StochRSI",
                            MinValue = 0,
                            MaxValue = 1,
                            Foreground = Brushes.White,
                            Separator = new LiveCharts.Wpf.Separator
                            {
                                StrokeThickness = 1,
                                Stroke = new SolidColorBrush(Color.FromArgb(48, 255, 255, 255))
                            }
                        }
                    };
                    break;
                    
                // Add other cases for WILLIAMSR, CCI, etc.
                default:
                    // Default configuration
                    chart.Series.Add(new LineSeries
                    {
                        Title = indicatorType,
                        Values = new ChartValues<double>(),
                        Stroke = new SolidColorBrush(Colors.White),
                        Fill = Brushes.Transparent,
                        PointGeometrySize = 0
                    });
                    
                    // Configure axes
                    chart.AxisX = new AxesCollection
                    {
                        new Axis
                        {
                            ShowLabels = false,
                            MinValue = 0,
                            LabelFormatter = DateTimeFormatter,
                            Separator = new LiveCharts.Wpf.Separator
                            {
                                StrokeThickness = 1,
                                Stroke = new SolidColorBrush(Color.FromArgb(48, 255, 255, 255)),
                                Step = 1
                            }
                        }
                    };
                    
                    chart.AxisY = new AxesCollection
                    {
                        new Axis
                        {
                            Title = indicatorType,
                            Foreground = Brushes.White,
                            Separator = new LiveCharts.Wpf.Separator
                            {
                                StrokeThickness = 1,
                                Stroke = new SolidColorBrush(Color.FromArgb(48, 255, 255, 255))
                            }
                        }
                    };
                    break;
            }
        }
        
        private void RemoveIndicatorPanel(string indicatorType)
        {
            if (_oscillatorPanels.TryGetValue(indicatorType, out var panel))
            {
                // Find the row index
                int rowIndex = Grid.GetRow(panel.Panel);
                
                // Remove the panel from the grid
                ChartsGrid.Children.Remove(panel.Panel);
                
                // Remove crosshair lines
                _crosshairLines.Remove(panel.Chart);
                
                // Remove from dictionary
                _oscillatorPanels.Remove(indicatorType);
                
                // Remove row definition
                // Note: This requires shifting all controls above this row
                // For simplicity, we're not implementing that here
                // In a real implementation, you would handle reindexing of rows
            }
        }
        
        private void UpdateOverlayIndicatorVisibility(string indicatorType, bool isVisible)
        {
            switch (indicatorType)
            {
                case "BB": // Bollinger Bands
                    // Add or update Bollinger Bands series
                    UpdateOrAddSeries(PriceChart, "Upper Band", UpperBandValues, Colors.LightBlue, isVisible);
                    UpdateOrAddSeries(PriceChart, "Middle Band", MiddleBandValues, Colors.Gray, isVisible);
                    UpdateOrAddSeries(PriceChart, "Lower Band", LowerBandValues, Colors.LightBlue, isVisible);
                    break;
                
                case "MA": // Moving Averages
                    UpdateOrAddSeries(PriceChart, "SMA 20", SMA20Values, Colors.Yellow, isVisible);
                    UpdateOrAddSeries(PriceChart, "SMA 50", SMA50Values, Colors.Magenta, isVisible);
                    UpdateOrAddSeries(PriceChart, "SMA 200", SMA200Values, Colors.White, isVisible);
                    break;
                    
                case "VWAP": // VWAP
                    UpdateOrAddSeries(PriceChart, "VWAP", VWAPValues, Colors.Cyan, isVisible);
                    break;
            }
        }
        
        private void UpdateOrAddSeries(CartesianChart chart, string title, IChartValues values, 
                                      Color color, bool isVisible)
        {
            // Check if series already exists
            var existingSeries = chart.Series.FirstOrDefault(s => s is LineSeries ls && ls.Title == title);
            
            if (existingSeries != null)
            {
                // Remove or add series based on visibility
                if (!isVisible)
                {
                    chart.Series.Remove(existingSeries);
                }
                // If visible, do nothing (already present)
            }
            else if (isVisible)
            {
                // Add new series
                chart.Series.Add(new LineSeries
                {
                    Title = title,
                    Values = values,
                    Stroke = new SolidColorBrush(color),
                    Fill = Brushes.Transparent,
                    PointGeometrySize = 0,
                    Visibility = Visibility.Visible
                });
            }
        }
        
        #endregion

        #region INotifyPropertyChanged Implementation
        
        public event PropertyChangedEventHandler PropertyChanged;
        
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        
        #endregion
    }
}