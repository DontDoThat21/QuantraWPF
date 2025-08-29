using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;

namespace Quantra.Controls.Components
{
    public partial class AnalysisParametersView : UserControl, INotifyPropertyChanged
    {
        // Events
        public event PropertyChangedEventHandler PropertyChanged;
        public event EventHandler<EventArgs> AnalyzeRequested;
        public event EventHandler<string> SymbolFilterChanged;
        public event EventHandler<string> TimeframeChanged;
        public event EventHandler<double> ConfidenceThresholdChanged;
        public event EventHandler<Dictionary<string, bool>> IndicatorsChanged;
        
        // Properties
        private string _symbolFilter = "Top Market Cap";
        public string SymbolFilter 
        { 
            get { return _symbolFilter; }
            private set
            {
                if (_symbolFilter != value)
                {
                    _symbolFilter = value;
                    OnPropertyChanged(nameof(SymbolFilter));
                    SymbolFilterChanged?.Invoke(this, _symbolFilter);
                }
            }
        }
        
        private string _timeframe = "1month";
        public string Timeframe
        {
            get { return _timeframe; }
            private set
            {
                if (_timeframe != value)
                {
                    _timeframe = value;
                    OnPropertyChanged(nameof(Timeframe));
                    TimeframeChanged?.Invoke(this, _timeframe);
                }
            }
        }
        
        private double _confidenceThreshold = 0.7;
        public double ConfidenceThreshold
        {
            get { return _confidenceThreshold; }
            private set
            {
                if (_confidenceThreshold != value)
                {
                    _confidenceThreshold = value;
                    OnPropertyChanged(nameof(ConfidenceThreshold));
                    ConfidenceThresholdChanged?.Invoke(this, _confidenceThreshold);
                }
            }
        }
        
        private Dictionary<string, bool> _selectedIndicators;
        public Dictionary<string, bool> SelectedIndicators
        {
            get { return _selectedIndicators; }
            private set
            {
                _selectedIndicators = value;
                OnPropertyChanged(nameof(SelectedIndicators));
                IndicatorsChanged?.Invoke(this, _selectedIndicators);
            }
        }

        // Constructor
        public AnalysisParametersView()
        {
            InitializeComponent();
            
            // Initialize indicators dictionary
            _selectedIndicators = new Dictionary<string, bool>
            {
                { "VWAP", true },
                { "MACD", true },
                { "RSI", true },
                { "Bollinger", true },
                { "Moving Avg", true },
                { "Volume", true },
                { "Breadth Thrust", false },
                { "ADX", false },
                { "ROC", false },
                { "Ult. Osc", false },
                { "Bull/Bear", false },
                { "CCI", false },
                { "ATR", false },
                { "Williams %R", false },
                { "Stochastic", false },
                { "Stoch RSI", false }
            };
            
            // Set data context
            this.DataContext = this;
        }

        // Event Handlers
        private void AnalyzeButton_Click(object sender, RoutedEventArgs e)
        {
            AnalyzeRequested?.Invoke(this, EventArgs.Empty);
        }

        private void SymbolFilterComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var selectedItem = SymbolFilterComboBox.SelectedItem as ComboBoxItem;
            if (selectedItem != null && selectedItem.Content?.ToString() == "All Symbols")
            {
                // TODO: Replace this with your actual logic to select all symbols.
                // For example, raise an event or call a method to update the symbol list.
                SelectAllSymbols();
                return;
            }
            SymbolFilter = selectedItem.Content.ToString();
        }

        private void TimeframeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (TimeframeComboBox.SelectedItem is ComboBoxItem selectedItem && selectedItem.Tag != null)
            {
                Timeframe = selectedItem.Tag.ToString();
            }
        }

        private void ConfidenceComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ConfidenceComboBox.SelectedItem is ComboBoxItem selectedItem && selectedItem.Tag != null)
            {
                ConfidenceThreshold = Convert.ToDouble(selectedItem.Tag);
            }
        }

        private void IndicatorCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            if (sender is CheckBox checkBox)
            {
                string indicatorName = checkBox.Content.ToString();
                if (_selectedIndicators.ContainsKey(indicatorName))
                {
                    _selectedIndicators[indicatorName] = checkBox.IsChecked ?? false;
                    IndicatorsChanged?.Invoke(this, _selectedIndicators);
                }
            }
        }

        // INotifyPropertyChanged implementation
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        // Add this helper method if not present
        private void SelectAllSymbols()
        {
            // Implement logic to select all symbols in your application.
            // This could involve updating a bound property, raising an event, etc.
            // Example placeholder:
            // SymbolList = SymbolService.GetAllSymbols();
        }
    }
}

// Replace any usage of ItemsControl.Children with ItemsControl.Items or use VisualTreeHelper as appropriate.
