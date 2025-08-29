using Quantra.Models;
using Quantra.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Collections.ObjectModel; // Added for ObservableCollection
using System.Windows.Data; // Added for CollectionViewSource

namespace Quantra.Controls
{
    public partial class PredictionAnalysisControl
    {
        // Add this field to ensure selectedPrediction is available in this partial class
        // private Quantra.Models.PredictionModel selectedPrediction;

        // Changed from List<PredictionModel> to ObservableCollection<PredictionModel>
        // Renamed to Predictions for clarity and to avoid conflict if xaml.cs has 'predictions'
        public ObservableCollection<Quantra.Models.PredictionModel> Predictions { get; set; } = new ObservableCollection<Quantra.Models.PredictionModel>();

        private async void AnalyzeButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (AnalyzeButton != null)
                    AnalyzeButton.IsEnabled = false;

                if (StatusText != null)
                    StatusText.Text = "Starting analysis...";

                // Clear previous filter
                var view = CollectionViewSource.GetDefaultView(this.Predictions);
                if (view != null)
                {
                    view.Filter = null;
                }

                // RunAutomatedAnalysis is now awaited and will populate this.Predictions
                await RunAutomatedAnalysis();

                // After analysis completes, apply filters based on user selections
                string symbolFilter = "All"; // Default, implement actual filter logic if needed
                double minConfidence = 0.0; // Default to 0 to show all if parsing fails

                try
                {
                    if (SymbolFilterComboBox?.SelectedItem is ComboBoxItem symbolItem)
                        symbolFilter = symbolItem.Content.ToString();

                    if (ConfidenceComboBox?.SelectedItem is ComboBoxItem confidenceItem && confidenceItem.Tag != null)
                    {
                        double.TryParse(confidenceItem.Tag.ToString(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out minConfidence);
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Warning", "Error reading combo box values for filtering, using defaults", ex.ToString());
                }

                if (view != null)
                {
                    // Ensure predictions collection is not null, though it should be initialized.
                    if (this.Predictions == null)
                    {
                        DatabaseMonolith.Log("Error", "Predictions collection is null after analysis. Aborting filtering.");
                        if (StatusText != null)
                            StatusText.Text = "Error: No predictions available to filter.";
                        if (AnalyzeButton != null)
                            AnalyzeButton.IsEnabled = true;
                        return;
                    }

                    view.Filter = item =>
                    {
                        if (item is Quantra.Models.PredictionModel prediction)
                        {
                            bool confidenceMatch = prediction.Confidence >= minConfidence;
                            // TODO: Implement symbol filtering based on 'symbolFilter' variable
                            // Example: bool symbolMatch = (symbolFilter == "All" || prediction.Symbol.Equals(symbolFilter, StringComparison.OrdinalIgnoreCase));
                            // return confidenceMatch && symbolMatch;
                            return confidenceMatch;
                        }
                        return false;
                    };

                    // Count items after filtering
                    int displayedCount = this.Predictions.Count(p => view.Filter(p));
                    if (StatusText != null)
                        StatusText.Text = $"Analysis complete. Displaying {displayedCount} predictions meeting criteria.";
                }
                else
                {
                    // Fallback if CollectionView is not available (should not happen with ObservableCollection)
                    var filteredPredictions = this.Predictions.Where(p => p.Confidence >= minConfidence).ToList();
                    PredictionDataGrid.ItemsSource = filteredPredictions; // This would detach from the ObservableCollection, less ideal
                    if (StatusText != null)
                        StatusText.Text = $"Analysis complete. Found {filteredPredictions.Count} predictions meeting criteria (fallback filter).";
                }

                if (LastUpdatedText != null) // LastUpdatedText is updated by RunAutomatedAnalysis
                    LastUpdatedText.Text = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";

                SaveControlSettings();
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error during prediction analysis: {ex.Message}", ex.ToString());
                if (StatusText != null)
                    StatusText.Text = "Error during analysis.";
            }
            finally
            {
                if (AnalyzeButton != null)
                    AnalyzeButton.IsEnabled = true;
            }
        }

        private void PredictionDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            try
            {
                if (PredictionDataGrid?.SelectedItem is Quantra.Models.PredictionModel prediction)
                {
                    // Update chart module with selected symbol
                    if (_chartModule != null && !string.IsNullOrEmpty(prediction.Symbol))
                    {
                        _chartModule.Symbol = prediction.Symbol;
                    }
                    // Assign to a field if needed, or handle selection logic here
                    // selectedPrediction = prediction; // If you need to keep track, declare the field in a single partial class only
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error in PredictionDataGrid_SelectionChanged", ex.ToString());
            }
        }

        // Add a new method to batch UI updates for the sector charts
        private void UpdateSectorChartsBatched(string sector)
        {
            try
            {
                TopPerformerValues?.Clear();
                TopPerformerLabels?.Clear();
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to update sector charts in batched operation", ex.ToString());
            }
        }

        // Add new method at the end of the class
        private void SaveControlSettings()
        {
            try
            {
                var settings = new IndicatorSettingsModel
                {
                    // Assuming this control instance has a unique identifier property, e.g., ControlId.
                    // For demonstration, using GetHashCode() as a placeholder; replace with an appropriate unique identifier.
                    ControlId = this.GetHashCode(),
                    UseVwap = VwapCheckBox?.IsChecked ?? false,
                    UseMacd = MacdCheckBox?.IsChecked ?? false,
                    UseRsi = RsiCheckBox?.IsChecked ?? false,
                    UseBollinger = BollingerCheckBox?.IsChecked ?? false,
                    UseMa = MaCheckBox?.IsChecked ?? false,
                    UseVolume = VolumeCheckBox?.IsChecked ?? false,
                    UseBreadthThrust = BreadthThrustCheckBox?.IsChecked ?? false
                };

                var service = new IndicatorSettingsService();
                service.SaveOrUpdateSettingsForControl(settings);
                DatabaseMonolith.Log("Info", $"Indicator settings saved for control {settings.ControlId}");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error saving indicator settings", ex.ToString());
            }
        }

        private void PredictionAnalysisControl_Loaded(object sender, RoutedEventArgs e)
        {
            // Ensure trading components are initialized on load
            if (_tradingBot == null || _stockDataCache == null)
                InitializeTradingComponents();

            // ...existing code...
        }

        // If symbol selection logic is handled here, ensure "All Symbols" is supported.
        // For example, in the handler for symbol filter changes:
        private void OnSymbolFilterChanged(string filter)
        {
            if (filter == "All Symbols")
            {
                // Implement logic to select all symbols
                // Example:
                // DisplayedSymbols = SymbolService.GetAllSymbols();
                return;
            }

            // ...existing logic for other filters...
        }

        // Update the event handler for SymbolFilterComboBox selection change
        private void SymbolFilterComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var selectedItem = SymbolFilterComboBox.SelectedItem as ComboBoxItem;
            if (selectedItem == null)
                return;

            var selectedSymbol = selectedItem.Content?.ToString();
            if (selectedSymbol == "All Symbols")
            {
                SelectAllSymbols();
                return;
            }

            // Add or update the selected symbol in the PredictionDataGrid (StockDataGrid)
            AddOrUpdateSymbolInGrid(selectedSymbol);
        }

        // Add this helper method to add or update a symbol in the grid
        private void AddOrUpdateSymbolInGrid(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol) || symbol == "All")
                return;

            try
            {
                // Use the predictions collection as the data source
                if (predictions == null)
                    return;

                // Find the prediction for the selected symbol
                var prediction = predictions.FirstOrDefault(p =>
                    p.Symbol.Equals(symbol, StringComparison.InvariantCultureIgnoreCase));
                if (prediction == null)
                {
                    // Optionally, fetch or create a new prediction for the symbol if not present
                    // For now, do nothing if not found
                    return;
                }

                // Get the current items in the grid
                var currentList = (PredictionDataGrid.ItemsSource as IList<Quantra.Models.PredictionModel>)?.ToList() ?? new List<Quantra.Models.PredictionModel>();

                // Check if the symbol is already in the grid
                var existing = currentList.FirstOrDefault(p =>
                    p.Symbol.Equals(symbol, StringComparison.InvariantCultureIgnoreCase));
                if (existing != null)
                {
                    // Update the existing row
                    var idx = currentList.IndexOf(existing);
                    currentList[idx] = prediction;
                }
                else
                {
                    // Add the new symbol to the grid
                    currentList.Add(prediction);
                }

                // Refresh the grid
                PredictionDataGrid.ItemsSource = null;
                PredictionDataGrid.ItemsSource = currentList;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error adding/updating symbol in grid", ex.ToString());
            }
        }

        // Add this method to resolve the missing 'SelectAllSymbols' reference.
        private void SelectAllSymbols()
        {
            // Example logic: set the displayed symbols to all available symbols.
            // Replace 'DisplayedSymbols' and 'SymbolService.GetAllSymbols()' with your actual implementation.
            try
            {
                // If you have a property or field for the displayed symbols, update it here.
                // DisplayedSymbols = SymbolService.GetAllSymbols();

                // If you need to update a UI element, do so here.
                // For example, if PredictionDataGrid displays symbols:
                if (PredictionDataGrid != null)
                {
                    // Assuming you have a method or service to get all predictions/symbols.
                    var allSymbols = predictions ?? new List<Quantra.Models.PredictionModel>();
                    PredictionDataGrid.ItemsSource = allSymbols;
                }

                // Optionally update status text or other UI elements.
                if (StatusText != null)
                    StatusText.Text = "All symbols selected.";
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error selecting all symbols", ex.ToString());
            }
        }

        // Dummy analysis logic for illustration
        private List<PredictionAnalysisResult> RunPredictionAnalysis()
        {
            // Replace with actual analysis logic
            return new List<PredictionAnalysisResult>
            {
                new PredictionAnalysisResult
                {
                    Symbol = "AAPL",
                    PredictedAction = "BUY",
                    Confidence = 0.82,
                    CurrentPrice = 180.12,
                    TargetPrice = 195.00,
                    PotentialReturn = 0.0825,
                    TradingRule = "Breakout",
                    Indicators = new Dictionary<string, double> { { "RSI", 32.1 }, { "MACDHistogram", 0.12 } }
                }
            };
        }
    }
}
