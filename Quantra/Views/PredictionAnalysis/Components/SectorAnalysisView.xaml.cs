using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Quantra.Models;
using LiveCharts;

namespace Quantra.Controls.Components
{
    public partial class SectorAnalysisView : UserControl, INotifyPropertyChanged
    {
        // Events
        public event PropertyChangedEventHandler PropertyChanged;
        public event EventHandler<string> SectorSelectionChanged;

        // Properties 
        public ChartValues<double> TopPerformerValues { get; set; }
        public List<string> TopPerformerLabels { get; set; }
        
        // Selected sector
        private string _selectedSector = "Technology";
        public string SelectedSector
        {
            get { return _selectedSector; }
            set
            {
                if (_selectedSector != value)
                {
                    _selectedSector = value;
                    OnPropertyChanged(nameof(SelectedSector));
                    SectorSelectionChanged?.Invoke(this, _selectedSector);
                }
            }
        }

        // Constructor
        public SectorAnalysisView()
        {
            InitializeComponent();
            
            // Initialize chart collections
            TopPerformerValues = new ChartValues<double>();
            TopPerformerLabels = new List<string>();
            
            this.DataContext = this;
        }

        // Public methods
        public void UpdateSectorData(string sector, Dictionary<string, double> sectorPerformance)
        {
            // Update the heatmap image
            try
            {
                GenerateSectorHeatmap(sector, sectorPerformance);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error generating sector heatmap: {ex.Message}");
            }

            // Update top performers
            UpdateTopPerformers(sector, sectorPerformance);
        }

        public void UpdateSectorPredictions(IEnumerable<PredictionModel> predictions)
        {
            // Filter predictions for selected sector
            var sectorPredictions = predictions.Where(p => IsPredictionInSelectedSector(p)).ToList();
            
            // Update data grid
            SectorPredictionDataGrid.ItemsSource = sectorPredictions;
        }

        public void UpdateSectorTrends(List<string> trends)
        {
            SectorTrendsListBox.ItemsSource = trends;
        }

        // Event handlers
        private void SectorComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (SectorComboBox.SelectedItem is ComboBoxItem selectedItem)
            {
                SelectedSector = selectedItem.Content.ToString();
            }
        }

        // Helper methods
        private bool IsPredictionInSelectedSector(PredictionModel prediction)
        {
            // This would typically come from a more sophisticated sector mapping service
            // For now we'll use a simplified mapping based on common tickers
            Dictionary<string, List<string>> sectorMappings = new Dictionary<string, List<string>>
            {
                { "Technology", new List<string> { "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AMD", "INTC", "ADBE", "CRM" } },
                { "Financial", new List<string> { "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK" } },
                { "Healthcare", new List<string> { "JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "ABT", "TMO", "BMY", "AMGN" } },
                { "Consumer Cyclical", new List<string> { "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG", "TJX", "MAR" } },
                { "Communication", new List<string> { "NFLX", "CMCSA", "VZ", "T", "TMUS", "DIS", "CHTR", "ATVI", "EA", "TTWO" } },
                { "Industrial", new List<string> { "HON", "UNP", "UPS", "BA", "CAT", "DE", "LMT", "GE", "MMM", "RTX" } },
                { "Energy", new List<string> { "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC", "OXY", "KMI" } }
            };

            if (sectorMappings.ContainsKey(SelectedSector))
            {
                return sectorMappings[SelectedSector].Contains(prediction.Symbol);
            }
            
            return false;
        }

        private void GenerateSectorHeatmap(string sector, Dictionary<string, double> performance)
        {
            // In a real implementation this would generate a heatmap image
            // For now, we'll just set a placeholder image
            // This would typically involve creating a visual representation of the sector performance
            
            // Placeholder code - in real app would generate actual heatmap
            HeatmapImage.Source = new BitmapImage();
        }

        private void UpdateTopPerformers(string sector, Dictionary<string, double> performance)
        {
            // Clear old data
            TopPerformerValues.Clear();
            TopPerformerLabels.Clear();
            
            // Get top 5 performers in this sector
            var topPerformers = performance
                .OrderByDescending(kv => kv.Value)
                .Take(5)
                .ToList();
            
            // Add to chart values
            foreach (var kv in topPerformers)
            {
                TopPerformerValues.Add(kv.Value);
                TopPerformerLabels.Add(kv.Key);
            }
            
            // Update chart
            TopPerformersChart.Update(true);
        }

        // INotifyPropertyChanged implementation
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
