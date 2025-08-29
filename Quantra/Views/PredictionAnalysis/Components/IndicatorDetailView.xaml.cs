using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using LiveCharts;

namespace Quantra.Controls.Components
{
    public partial class IndicatorDetailView : UserControl, INotifyPropertyChanged
    {
        // Events
        public event PropertyChangedEventHandler PropertyChanged;
        public event EventHandler CloseRequested;

        // Properties
        public ChartValues<double> HistoricalValues { get; set; }
        public List<string> DateLabels { get; set; }
        
        private string _indicatorName = "RSI";
        public string IndicatorName
        {
            get { return _indicatorName; }
            set
            {
                if (_indicatorName != value)
                {
                    _indicatorName = value;
                    OnPropertyChanged(nameof(IndicatorName));
                    UpdateIndicatorTitle();
                }
            }
        }
        
        private Brush _indicatorColor = Brushes.Cyan;
        public Brush IndicatorColor
        {
            get { return _indicatorColor; }
            set
            {
                if (_indicatorColor != value)
                {
                    _indicatorColor = value;
                    OnPropertyChanged(nameof(IndicatorColor));
                }
            }
        }

        // Constructor
        public IndicatorDetailView()
        {
            InitializeComponent();
            
            // Initialize chart data
            HistoricalValues = new ChartValues<double>();
            DateLabels = new List<string>();
            
            // Set data context
            this.DataContext = this;
            
            // Set initial title
            UpdateIndicatorTitle();
        }

        // Public methods
        public void UpdateIndicatorData(string indicatorName, List<double> values, List<string> dates, Brush color)
        {
            // Update properties
            IndicatorName = indicatorName;
            IndicatorColor = color;
            
            // Clear old data
            HistoricalValues.Clear();
            DateLabels.Clear();
            
            // Add new data
            HistoricalValues.AddRange(values); // Correct: AddRange for ChartValues<double>
            DateLabels.AddRange(dates);
            
            // Update chart
            IndicatorHistoryChart.Update(true);
        }

        // Event handlers
        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            CloseRequested?.Invoke(this, EventArgs.Empty);
        }

        // Helper methods
        private void UpdateIndicatorTitle()
        {
            IndicatorTitle.Text = $"{IndicatorName} Historical Data";
        }

        // INotifyPropertyChanged implementation
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
