using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using Quantra.Models;
using LiveCharts;
using LiveCharts.Defaults;

namespace Quantra.Controls.Components
{
    public partial class PatternRecognitionView : UserControl, INotifyPropertyChanged
    {
        // Events
        public event PropertyChangedEventHandler PropertyChanged;
        
        // Properties
        public ChartValues<OhlcPoint> PatternCandles { get; set; }
        public ChartValues<double> PatternHighlights { get; set; }
        
        // Selected pattern
        private PatternModel _selectedPattern;
        public PatternModel SelectedPattern
        {
            get { return _selectedPattern; }
            private set
            {
                if (_selectedPattern != value)
                {
                    _selectedPattern = value;
                    OnPropertyChanged(nameof(SelectedPattern));
                }
            }
        }

        // Constructor
        public PatternRecognitionView()
        {
            InitializeComponent();
            
            // Initialize chart collections
            PatternCandles = new ChartValues<OhlcPoint>();
            PatternHighlights = new ChartValues<double>();
            
            // Set data context
            this.DataContext = this;
        }

        // Public methods
        public void UpdatePatterns(IEnumerable<PatternModel> patterns)
        {
            PatternListBox.ItemsSource = patterns;
        }

        // Event handlers
        private void PatternListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (PatternListBox.SelectedItem is PatternModel pattern)
            {
                SelectedPattern = pattern;
                UpdatePatternDisplay(pattern);
            }
        }

        // Helper methods
        private void UpdatePatternDisplay(PatternModel pattern)
        {
            // Update header info
            PatternTitleText.Text = pattern.PatternName;
            PatternSymbolText.Text = $"Symbol: {pattern.Symbol}";
            PatternReliabilityText.Text = $"Reliability: {pattern.Reliability:P0}";
            
            // Update description
            PatternDescriptionText.Text = pattern.Description;
            PatternOutcomeText.Text = pattern.PredictedOutcome;
            
            // Update chart data
            UpdatePatternChart(pattern);
        }

        private void UpdatePatternChart(PatternModel pattern)
        {
            // Clear old data
            PatternCandles.Clear();
            PatternHighlights.Clear();
            
            // Generate candle data (this would typically come from the pattern data in a real app)
            // For now, we'll generate some placeholder data
            double basePrice = 100;
            Random random = new Random((int)DateTime.Now.Ticks);
            
            for (int i = 0; i < 30; i++)
            {
                double open = basePrice * (1 + (random.NextDouble() - 0.5) * 0.02);
                double close = open * (1 + (random.NextDouble() - 0.5) * 0.03);
                double high = Math.Max(open, close) * (1 + random.NextDouble() * 0.01);
                double low = Math.Min(open, close) * (1 - random.NextDouble() * 0.01);
                
                PatternCandles.Add(new OhlcPoint(open, high, low, close));
                
                // Simple line for pattern highlights - in a real app this would highlight the pattern
                PatternHighlights.Add(i >= 20 && i <= 25 ? close : double.NaN);
                
                basePrice = close;
            }
            
            // Update chart
            PatternChart.Update(true);
        }

        // INotifyPropertyChanged implementation
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
