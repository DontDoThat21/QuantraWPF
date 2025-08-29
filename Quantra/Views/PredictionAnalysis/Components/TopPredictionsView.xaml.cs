using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Quantra.Models;  // Explicitly use the Models namespace
using Quantra.Views.Orders;
using LiveCharts;
using LiveCharts.Defaults;

namespace Quantra.Controls.Components
{
    public partial class TopPredictionsView : UserControl, INotifyPropertyChanged
    {
        // Events
        public event PropertyChangedEventHandler PropertyChanged;
        public event EventHandler<Quantra.Models.PredictionModel> CreateRuleRequested;  // Use fully qualified type
        public event EventHandler<Quantra.Models.PredictionModel> TradingRuleClicked;   // Use fully qualified type
        public event EventHandler<Quantra.Models.PredictionModel> RuleCreationRequested; // Use fully qualified type
        public event EventHandler<Quantra.Models.PredictionModel> PredictionSelected;    // Use fully qualified type

        // Dependencies
        private OrdersPage _ordersPage;

        // Properties
        private bool _isOrdersPageAvailable;
        public bool IsOrdersPageAvailable
        {
            get { return _isOrdersPageAvailable; }
            private set 
            {
                if (_isOrdersPageAvailable != value)
                {
                    _isOrdersPageAvailable = value;
                    OnPropertyChanged(nameof(IsOrdersPageAvailable));
                }
            }
        }

        // Selected prediction
        private Quantra.Models.PredictionModel _selectedPrediction;  // Use fully qualified type
        public Quantra.Models.PredictionModel SelectedPrediction    // Use fully qualified type
        { 
            get { return _selectedPrediction; }
            private set 
            {
                if (_selectedPrediction != value)
                {
                    _selectedPrediction = value;
                    OnPropertyChanged(nameof(SelectedPrediction));
                }
            }
        }

        // Chart data collections for binding
        public ChartValues<double> PriceValues { get; set; }
        public ChartValues<double> VwapValues { get; set; }
        public ChartValues<double> PredictionValues { get; set; }
        public ChartValues<double> AdxValues { get; set; }
        public ChartValues<double> RocValues { get; set; }
        public ChartValues<double> UoValues { get; set; }
        public ChartValues<double> CciValues { get; set; }
        public ChartValues<double> AtrValues { get; set; }
        public ChartValues<double> WilliamsRValues { get; set; }
        public ChartValues<double> StochKValues { get; set; }
        public ChartValues<double> StochDValues { get; set; }
        public ChartValues<double> StochRsiValues { get; set; }
        public ChartValues<double> BullPowerValues { get; set; }
        public ChartValues<double> BearPowerValues { get; set; }
        public ChartValues<double> BreadthThrustValues { get; set; }
        public Func<double, string> DateFormatter { get; set; }

        // Constructor
        public TopPredictionsView()
        {
            InitializeComponent();

            // Initialize chart data
            PriceValues = new ChartValues<double>();
            VwapValues = new ChartValues<double>();
            PredictionValues = new ChartValues<double>();
            AdxValues = new ChartValues<double>();
            RocValues = new ChartValues<double>();
            UoValues = new ChartValues<double>();
            CciValues = new ChartValues<double>();
            AtrValues = new ChartValues<double>();
            WilliamsRValues = new ChartValues<double>();
            StochKValues = new ChartValues<double>();
            StochDValues = new ChartValues<double>();
            StochRsiValues = new ChartValues<double>();
            BullPowerValues = new ChartValues<double>();
            BearPowerValues = new ChartValues<double>();
            BreadthThrustValues = new ChartValues<double>();

            DateFormatter = value => new DateTime((long)value).ToString("MM/dd");

            // Set data context
            this.DataContext = this;
        }

        // Methods
        public void SetOrdersPage(OrdersPage ordersPage)
        {
            _ordersPage = ordersPage;
            IsOrdersPageAvailable = (_ordersPage != null);
        }

        public void UpdatePredictions(IEnumerable<Quantra.Models.PredictionModel> predictions)  // Use fully qualified type
        {
            PredictionDataGrid.ItemsSource = predictions;
        }

        // Event handlers
        private void PredictionDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            SelectedPrediction = PredictionDataGrid.SelectedItem as Quantra.Models.PredictionModel;  // Use fully qualified type
            if (SelectedPrediction != null)
            {
                PredictionDetail.UpdatePrediction(SelectedPrediction);
            }
        }

        private void CreateRule_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.DataContext is Quantra.Models.PredictionModel prediction)  // Use fully qualified type
            {
                CreateRuleRequested?.Invoke(this, prediction);
            }
        }

        private void TradingRule_Click(object sender, MouseButtonEventArgs e)
        {
            if (sender is Border border && border.DataContext is Quantra.Models.PredictionModel prediction)  // Use fully qualified type
            {
                TradingRuleClicked?.Invoke(this, prediction);
            }
        }

        // Method to update indicator displays
        public void UpdateIndicatorDisplay(Quantra.Models.PredictionModel prediction)  // Use fully qualified type
        {
            // Delegate the indicator update to the PredictionDetail control
            if (prediction != null)
            {
                PredictionDetail.UpdatePrediction(prediction);
            }
        }

        // Method to update chart data
        public void UpdateChartData(Quantra.Models.PredictionModel prediction, List<double> prices, 
                                   List<double> vwap, 
                                   List<double> predictionLine)  // Use fully qualified type
        {
            // Update chart values
            PriceValues.Clear();
            VwapValues.Clear();
            PredictionValues.Clear();

            PriceValues.AddRange(prices);
            VwapValues.AddRange(vwap);
            PredictionValues.AddRange(predictionLine);

            // Delegate chart updates to PredictionDetail instead of directly accessing PredictionChart
            if (prediction != null)
            {
                PredictionDetail.UpdatePrediction(prediction);
            }
        }

        // INotifyPropertyChanged implementation
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}

// Replace any usage of ItemsControl.Children with ItemsControl.Items or use VisualTreeHelper as appropriate.
