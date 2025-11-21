using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.Views.ChartExtensions;
using Quantra.DAL.Services;

namespace Quantra.Views.SupportResistance
{
    /// <summary>
    /// Interaction logic for SupportResistanceDemoView.xaml
    /// </summary>
    public partial class SupportResistanceDemoView : UserControl, INotifyPropertyChanged
    {
        private string _selectedSymbol = "AAPL";
        private string _selectedTimeframe = "1day";
        private SupportResistanceStrategy _strategy;
        private List<HistoricalPrice> _historicalPrices;
        private HistoricalDataService _dataService;
        private bool _isLoading = false;
        private List<string> _dateLabels = new List<string>();
        
        // Parameterless constructor for XAML designer support
        public SupportResistanceDemoView()
        {
            InitializeComponent();
            DataContext = this;
            _strategy = new SupportResistanceStrategy();
            _dateLabels = new List<string>();
        }
        
        public SupportResistanceDemoView(UserSettingsService userSettingsService, LoggingService loggingService)
        {
            InitializeComponent();
            DataContext = this;
            
            // Initialize services and strategy
            _dataService = new HistoricalDataService(userSettingsService, loggingService);
            _strategy = new SupportResistanceStrategy();
            
            // Initialize config control
            ConfigControl.Initialize(_strategy);
            ConfigControl.SettingsApplied += ConfigControl_SettingsApplied;
            
            // Load initial data
            LoadHistoricalData();
        }
        
        #region Properties
        
        public bool IsLoading
        {
            get { return _isLoading; }
            set
            {
                _isLoading = value;
                OnPropertyChanged(nameof(IsLoading));
            }
        }
        
        public List<string> DateLabels
        {
            get { return _dateLabels; }
            set
            {
                _dateLabels = value;
                OnPropertyChanged(nameof(DateLabels));
            }
        }
        
        #endregion
        
        #region Data Handling
        
        /// <summary>
        /// Load historical price data for the selected symbol and timeframe
        /// </summary>
        private async void LoadHistoricalData()
        {
            try
            {
                IsLoading = true;
                StatusText.Text = $"Loading data for {_selectedSymbol} ({_selectedTimeframe})...";
                
                // Convert timeframe to range and interval
                string range = ConvertTimeframeToRange(_selectedTimeframe);
                string interval = ConvertTimeframeToInterval(_selectedTimeframe);
                
                // Get historical data
                _historicalPrices = await _dataService.GetHistoricalPrices(_selectedSymbol, range, interval);
                
                if (_historicalPrices != null && _historicalPrices.Count > 0)
                {
                    StatusText.Text = $"Loaded {_historicalPrices.Count} data points for {_selectedSymbol}";
                    
                    // Analyze support/resistance levels
                    AnalyzePriceLevels();
                    
                    // Update chart
                    UpdatePriceChart();
                    
                    // Update detected levels in the config control
                    ConfigControl.UpdateDetectedLevels();
                }
                else
                {
                    StatusText.Text = $"No data available for {_selectedSymbol}";
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
            }
            finally
            {
                IsLoading = false;
            }
        }
        
        /// <summary>
        /// Analyze price data to detect support/resistance levels
        /// </summary>
        private void AnalyzePriceLevels()
        {
            if (_historicalPrices == null || _historicalPrices.Count == 0)
                return;
                
            // Generate trading signal (this internally detects levels)
            _strategy.GenerateSignal(_historicalPrices);
        }
        
        /// <summary>
        /// Update the price chart with historical data and support/resistance levels
        /// </summary>
        private void UpdatePriceChart()
        {
            if (_historicalPrices == null || _historicalPrices.Count == 0)
                return;
                
            // Clear existing series
            PriceChart.Series.Clear();
            
            // Create price series
            var priceSeries = new LineSeries
            {
                Title = _selectedSymbol,
                Values = new ChartValues<double>(_historicalPrices.Select(p => p.Close)),
                LineSmoothness = 0,
                PointGeometry = null,
                StrokeThickness = 2,
                Stroke = System.Windows.Media.Brushes.Blue
            };
            
            // Update date labels
            DateLabels = _historicalPrices
                .Select(p => p.Date.ToString("MM/dd"))
                .ToList();
            
            // Add price series to chart
            PriceChart.Series.Add(priceSeries);
            
            // Add support/resistance levels to chart
            PriceChart.SetupPriceLevelsOnChart(_strategy);
        }
        
        #endregion
        
        #region Helper Methods
        
        /// <summary>
        /// Convert timeframe string to data range
        /// </summary>
        private string ConvertTimeframeToRange(string timeframe)
        {
            switch (timeframe.ToLower())
            {
                case "1min": return "1d";
                case "5min": return "5d";
                case "15min": return "5d";
                case "30min": return "5d";
                case "1hour": return "1mo";
                case "1day": return "1y";
                case "1week": return "2y";
                case "1month": return "5y";
                default: return "1y";
            }
        }
        
        /// <summary>
        /// Convert timeframe string to data interval
        /// </summary>
        private string ConvertTimeframeToInterval(string timeframe)
        {
            switch (timeframe.ToLower())
            {
                case "1min": return "1m";
                case "5min": return "5m";
                case "15min": return "15m";
                case "30min": return "30m";
                case "1hour": return "60m";
                case "1day": return "1d";
                case "1week": return "1wk";
                case "1month": return "1mo";
                default: return "1d";
            }
        }
        
        #endregion
        
        #region Event Handlers
        
        private void SymbolComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var comboBox = sender as ComboBox;
            if (comboBox != null && comboBox.SelectedItem != null)
            {
                var item = comboBox.SelectedItem as ComboBoxItem;
                if (item != null)
                {
                    _selectedSymbol = item.Content.ToString();
                    LoadHistoricalData();
                }
            }
        }
        
        private void TimeframeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var comboBox = sender as ComboBox;
            if (comboBox != null && comboBox.SelectedItem != null)
            {
                var item = comboBox.SelectedItem as ComboBoxItem;
                if (item != null)
                {
                    _selectedTimeframe = item.Content.ToString();
                    LoadHistoricalData();
                }
            }
        }
        
        private void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            LoadHistoricalData();
        }
        
        private void SettingsToggleButton_Click(object sender, RoutedEventArgs e)
        {
            ConfigControl.Visibility = (ConfigControl.Visibility == Visibility.Visible) 
                ? Visibility.Collapsed : Visibility.Visible;
        }
        
        private void ConfigControl_SettingsApplied(object sender, EventArgs e)
        {
            // Re-analyze levels and update chart when settings change
            AnalyzePriceLevels();
            UpdatePriceChart();
            ConfigControl.UpdateDetectedLevels();
        }
        
        #endregion
        
        #region INotifyPropertyChanged
        
        public event PropertyChangedEventHandler PropertyChanged;
        
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        
        #endregion
    }
}