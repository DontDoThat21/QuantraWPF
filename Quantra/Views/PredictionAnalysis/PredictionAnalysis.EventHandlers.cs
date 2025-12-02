using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Collections.ObjectModel; // Added for ObservableCollection
using System.Windows.Data;
using Quantra.DAL.Services; // Added for CollectionViewSource
using System.Windows.Input; // Added for KeyEventArgs

namespace Quantra.Controls
{
    public partial class PredictionAnalysis : UserControl
    {
        // Add this field to ensure selectedPrediction is available in this partial class
        // private Quantra.Models.PredictionModel selectedPrediction;

        // Changed from List<PredictionModel> to ObservableCollection<PredictionModel>
        // Renamed to Predictions for clarity and to avoid conflict if xaml.cs has 'predictions'
        public ObservableCollection<Quantra.Models.PredictionModel> Predictions { get; set; } = new ObservableCollection<Quantra.Models.PredictionModel>();

        // Track if user is manually interacting with tabs to prevent automatic tab switching
        private bool _isUserSelectingTab = false;
        private int _lastUserSelectedTabIndex = 0;

        // Symbol selection mode fields
        private ObservableCollection<CachedSymbolInfo> _cachedSymbols = new ObservableCollection<CachedSymbolInfo>();
        private SymbolSelectionMode _currentSymbolMode = SymbolSelectionMode.Individual;

        // Track if initial predictions have been loaded
        private bool _initialPredictionsLoaded = false;

        // Timer for debounced cached symbol search
        private System.Windows.Threading.DispatcherTimer _cachedSymbolSearchTimer;

        /// <summary>
        /// Enum to track symbol selection mode
        /// </summary>
        public enum SymbolSelectionMode
        {
            Individual,
            Category
        }

        /// <summary>
        /// Handles symbol selection mode changes
        /// </summary>
        private void SymbolModeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (SymbolModeComboBox?.SelectedItem is ComboBoxItem selectedItem)
            {
                var mode = selectedItem.Content?.ToString();

                if (mode == "Individual Symbol")
                {
                    _currentSymbolMode = SymbolSelectionMode.Individual;
                    if (IndividualSymbolPanel != null)
                        IndividualSymbolPanel.Visibility = Visibility.Visible;
                    if (CategoryFilterPanel != null)
                        CategoryFilterPanel.Visibility = Visibility.Collapsed;
                    if (StatusText != null)
                        StatusText.Text = "Individual symbol mode - Enter a symbol or select from cache";

                    // Load cached symbols when entering individual mode
                    LoadCachedSymbols();
                }
                else if (mode == "Category Filter")
                {
                    _currentSymbolMode = SymbolSelectionMode.Category;
                    if (IndividualSymbolPanel != null)
                        IndividualSymbolPanel.Visibility = Visibility.Collapsed;
                    if (CategoryFilterPanel != null)
                        CategoryFilterPanel.Visibility = Visibility.Visible;
                    if (StatusText != null)
                        StatusText.Text = "Category mode - Select a category to analyze multiple symbols";
                }
            }
        }

        /// <summary>
        /// Initialize the cached symbol search timer
        /// </summary>
        private void InitializeCachedSymbolSearchTimer()
        {
            _cachedSymbolSearchTimer = new System.Windows.Threading.DispatcherTimer();
            _cachedSymbolSearchTimer.Interval = TimeSpan.FromMilliseconds(300); // 300ms debounce delay for local database search
            _cachedSymbolSearchTimer.Tick += CachedSymbolSearchTimer_Tick;
        }

        /// <summary>
        /// Load all cached symbols and populate the search results listbox
        /// </summary>
        private void LoadCachedSymbols()
        {
            try
            {
                _cachedSymbols.Clear();

                if (_stockDataCacheService == null)
                {
                    if (StatusText != null)
                        StatusText.Text = "Cache service not available";
                    return;
                }

                var symbols = _stockDataCacheService.GetAllCachedSymbols();

                foreach (var symbol in symbols)
                {
                    _cachedSymbols.Add(new CachedSymbolInfo
                    {
                        Symbol = symbol,
                        CacheInfo = "cached in database"
                    });
                }

                // Populate the search results listbox with all cached symbols
                if (CachedSymbolSearchResultsListBox != null)
                {
                    CachedSymbolSearchResultsListBox.ItemsSource = _cachedSymbols;
                }

                _loggingService?.Log("Info", $"Loaded {_cachedSymbols.Count} cached symbols for prediction analysis");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to load cached symbols");
                if (StatusText != null)
                    StatusText.Text = "Error loading cached symbols";
            }
        }

        /// <summary>
        /// Search cached symbols from pre-loaded collection (debounced)
        /// </summary>
        private void CachedSymbolSearchTimer_Tick(object sender, EventArgs e)
        {
            _cachedSymbolSearchTimer?.Stop();

            try
            {
                var searchText = CachedSymbolSearchTextBox?.Text?.Trim() ?? "";

                // If search text is empty, show all cached symbols
                if (string.IsNullOrWhiteSpace(searchText))
                {
                    if (CachedSymbolSearchResultsListBox != null)
                    {
                        CachedSymbolSearchResultsListBox.ItemsSource = _cachedSymbols;
                        
                        if (_cachedSymbols.Count > 0 && CachedSymbolSearchPopup != null)
                        {
                            CachedSymbolSearchPopup.IsOpen = true;
                        }
                    }
                    return;
                }

                // Filter the pre-loaded cached symbols based on search text
                var matchingSymbols = _cachedSymbols
                    .Where(s => s.Symbol.Contains(searchText, StringComparison.OrdinalIgnoreCase))
                    .OrderBy(s => s.Symbol.StartsWith(searchText, StringComparison.OrdinalIgnoreCase) ? 0 : 1) // Prioritize symbols that start with search text
                    .ThenBy(s => s.Symbol.Length) // Then shorter symbols
                    .ThenBy(s => s.Symbol) // Then alphabetically
                    .Take(50) // Limit to 50 results for performance
                    .ToList();

                if (CachedSymbolSearchResultsListBox != null)
                {
                    CachedSymbolSearchResultsListBox.ItemsSource = matchingSymbols;

                    if (matchingSymbols.Count > 0 && CachedSymbolSearchPopup != null)
                    {
                        CachedSymbolSearchPopup.IsOpen = true;
                    }
                    else if (CachedSymbolSearchPopup != null)
                    {
                        CachedSymbolSearchPopup.IsOpen = false;
                    }
                }

                _loggingService?.Log("Debug", $"Cached symbol search: '{searchText}' returned {matchingSymbols.Count} results");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error performing cached symbol search");
                if (StatusText != null)
                    StatusText.Text = $"Error searching cached symbols: {ex.Message}";
            }
        }

        /// <summary>
        /// Handle text change in cached symbol search box
        /// </summary>
        private void CachedSymbolSearchTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            _cachedSymbolSearchTimer?.Stop();
            _cachedSymbolSearchTimer?.Start();
        }

        /// <summary>
        /// Handle keyboard navigation in cached symbol search box
        /// </summary>
        private void CachedSymbolSearchTextBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Down && CachedSymbolSearchResultsListBox?.Items.Count > 0)
            {
                CachedSymbolSearchResultsListBox.SelectedIndex = 0;
                CachedSymbolSearchResultsListBox.Focus();
                e.Handled = true;
            }
            else if (e.Key == Key.Enter && CachedSymbolSearchResultsListBox?.SelectedItem is CachedSymbolInfo selected)
            {
                SelectCachedSymbol(selected);
                e.Handled = true;
            }
            else if (e.Key == Key.Escape && CachedSymbolSearchPopup != null)
            {
                CachedSymbolSearchPopup.IsOpen = false;
                e.Handled = true;
            }
        }

        /// <summary>
        /// Show search results when focused
        /// </summary>
        private void CachedSymbolSearchTextBox_GotFocus(object sender, RoutedEventArgs e)
        {
            // If search text is empty, show all cached symbols
            if (string.IsNullOrWhiteSpace(CachedSymbolSearchTextBox?.Text))
            {
                if (CachedSymbolSearchResultsListBox != null)
                {
                    CachedSymbolSearchResultsListBox.ItemsSource = _cachedSymbols;
                }
            }
            
            if (CachedSymbolSearchResultsListBox?.Items.Count > 0 && CachedSymbolSearchPopup != null)
            {
                CachedSymbolSearchPopup.IsOpen = true;
            }
        }

        /// <summary>
        /// Hide search results when focus lost
        /// </summary>
        private void CachedSymbolSearchTextBox_LostFocus(object sender, RoutedEventArgs e)
        {
            // Delay closing to allow click on results
            Dispatcher.BeginInvoke(new Action(() =>
            {
                if (CachedSymbolSearchResultsListBox != null && CachedSymbolSearchTextBox != null && CachedSymbolSearchPopup != null)
                {
                    if (!CachedSymbolSearchResultsListBox.IsMouseOver && !CachedSymbolSearchTextBox.IsKeyboardFocusWithin)
                    {
                        CachedSymbolSearchPopup.IsOpen = false;
                    }
                }
            }), System.Windows.Threading.DispatcherPriority.Background);
        }

        /// <summary>
        /// Handle mouse click on search result
        /// </summary>
        private void CachedSymbolSearchResultsListBox_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (CachedSymbolSearchResultsListBox?.SelectedItem is CachedSymbolInfo selected)
            {
                SelectCachedSymbol(selected);
            }
        }

        /// <summary>
        /// Handle keyboard navigation in search results
        /// </summary>
        private void CachedSymbolSearchResultsListBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter && CachedSymbolSearchResultsListBox?.SelectedItem is CachedSymbolInfo selected)
            {
                SelectCachedSymbol(selected);
                e.Handled = true;
            }
            else if (e.Key == Key.Escape && CachedSymbolSearchPopup != null)
            {
                CachedSymbolSearchPopup.IsOpen = false;
                CachedSymbolSearchTextBox?.Focus();
                e.Handled = true;
            }
        }

        /// <summary>
        /// Select a cached symbol and populate the manual symbol textbox
        /// </summary>
        private void SelectCachedSymbol(CachedSymbolInfo symbol)
        {
            if (ManualSymbolTextBox != null)
                ManualSymbolTextBox.Text = symbol.Symbol;

            if (CachedSymbolSearchTextBox != null)
                CachedSymbolSearchTextBox.Text = symbol.Symbol;

            if (CachedSymbolSearchPopup != null)
                CachedSymbolSearchPopup.IsOpen = false;

            if (StatusText != null)
                StatusText.Text = $"Selected cached symbol: {symbol.Symbol}";

            _loggingService?.Log("Info", $"User selected cached symbol: {symbol.Symbol}");
        }

        /// <summary>
        /// Refresh the cached symbols list
        /// </summary>
        private void RefreshCacheButton_Click(object sender, RoutedEventArgs e)
        {
            LoadCachedSymbols();
            if (StatusText != null)
                StatusText.Text = $"Refreshed cached symbols. Found {_cachedSymbols.Count} symbols.";
        }

        /// <summary>
        /// Handle Enter key press in manual symbol textbox
        /// </summary>
        private async void ManualSymbolTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                var symbol = ManualSymbolTextBox?.Text?.Trim().ToUpper();
                if (!string.IsNullOrEmpty(symbol))
                {
                    await AnalyzeIndividualSymbol(symbol);
                }
            }
        }

        /// <summary>
        /// Analyze a single individual symbol
        /// </summary>
        private async Task AnalyzeIndividualSymbol(string symbol)
        {
            try
            {
                if (AnalyzeButton != null)
                    AnalyzeButton.IsEnabled = false;

                if (StatusText != null)
                    StatusText.Text = $"Analyzing {symbol}...";

                // Clear previous predictions
                Predictions.Clear();

                // Run prediction analysis for the individual symbol
                var prediction = await AnalyzeStockWithAllAlgorithms(symbol);

                if (prediction != null && prediction.PredictedAction != "ERROR")
                {
                    // Save prediction to database
                    await SavePredictionToDatabaseAsync(prediction);

                    // Add to UI collection on UI thread
                    await Dispatcher.InvokeAsync(() =>
                    {
                        Predictions.Add(prediction);

                        // Force DataGrid refresh
                        if (PredictionDataGrid != null)
                        {
                            PredictionDataGrid.ItemsSource = null;
                            PredictionDataGrid.ItemsSource = Predictions;
                            PredictionDataGrid.Items.Refresh();
                            PredictionDataGrid.UpdateLayout();
                        }

                        if (StatusText != null)
                            StatusText.Text = $"Analysis complete for {symbol}. Action: {prediction.PredictedAction}, Confidence: {prediction.Confidence:P0}";
                    });
                }
                else
                {
                    if (StatusText != null)
                        StatusText.Text = $"No prediction generated for {symbol}. Check if valid symbol.";
                }

                if (LastUpdatedText != null)
                    LastUpdatedText.Text = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Error analyzing {symbol}");
                if (StatusText != null)
                    StatusText.Text = $"Error analyzing {symbol}: {ex.Message}";
            }
            finally
            {
                if (AnalyzeButton != null)
                    AnalyzeButton.IsEnabled = true;
            }
        }

        private async void AnalyzeButton_Click(object sender, RoutedEventArgs e)
        {
            // Check if individual symbol mode is selected
            if (_currentSymbolMode == SymbolSelectionMode.Individual)
            {
                var symbol = ManualSymbolTextBox?.Text?.Trim().ToUpper();
                if (string.IsNullOrEmpty(symbol))
                {
                    if (StatusText != null)
                        StatusText.Text = "Please enter a valid symbol";
                    return;
                }
                await AnalyzeIndividualSymbol(symbol);
                return;
            }

            // Category mode - original implementation
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
                    //DatabaseMonolith.Log("Warning", "Error reading combo box values for filtering, using defaults", ex.ToString());
                }

                if (view != null)
                {
                    // Ensure predictions collection is not null, though it should be initialized.
                    if (this.Predictions == null)
                    {
                        //DatabaseMonolith.Log("Error", "Predictions collection is null after analysis. Aborting filtering.");
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
                //DatabaseMonolith.Log("Error", $"Error during prediction analysis: {ex.Message}", ex.ToString());
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
                //DatabaseMonolith.Log("Error", "Error in PredictionDataGrid_SelectionChanged", ex.ToString());
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
                //DatabaseMonolith.Log("Error", "Failed to update sector charts in batched operation", ex.ToString());
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

                _indicatorSettingsService.SaveOrUpdateSettingsForControl(settings);
                //DatabaseMonolith.Log("Info", $"Indicator settings saved for control {settings.ControlId}");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error saving indicator settings", ex.ToString());
            }
        }
        
        /// <summary>
        /// Load initial predictions from the database when the control loads
        /// </summary>
        private async Task LoadInitialPredictionsAsync()
        {
            if (_initialPredictionsLoaded)
                return;
                
            try
            {
                if (StatusText != null)
                    StatusText.Text = "Loading recent predictions...";
                
                // Get latest predictions from the database using PredictionAnalysisService
                var predictionService = new Quantra.DAL.Services.PredictionAnalysisService();
                var latestPredictions = await predictionService.GetLatestPredictionsAsync();
                
                if (latestPredictions != null && latestPredictions.Count > 0)
                {
                    // Add predictions to the collection on the UI thread
                    await Dispatcher.InvokeAsync(() =>
                    {
                        Predictions.Clear();
                        foreach (var prediction in latestPredictions)
                        {
                            Predictions.Add(prediction);
                        }
                        
                        // Force DataGrid refresh
                        if (PredictionDataGrid != null)
                        {
                            PredictionDataGrid.ItemsSource = null;
                            PredictionDataGrid.ItemsSource = Predictions;
                            PredictionDataGrid.Items.Refresh();
                        }
                        
                        if (StatusText != null)
                            StatusText.Text = $"Loaded {latestPredictions.Count} recent predictions";
                        
                        if (LastUpdatedText != null)
                            LastUpdatedText.Text = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";
                    });
                    
                    _initialPredictionsLoaded = true;
                    _loggingService?.Log("Info", $"Loaded {latestPredictions.Count} initial predictions");
                }
                else
                {
                    if (StatusText != null)
                        StatusText.Text = "No predictions found. Click Analyze to generate predictions.";
                    
                    _loggingService?.Log("Info", "No initial predictions found in database");
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to load initial predictions");
                if (StatusText != null)
                    StatusText.Text = "Error loading predictions. Click Analyze to generate new predictions.";
            }
        }

        private async void PredictionAnalysisControl_Loaded(object sender, RoutedEventArgs e)
        {
            // Ensure trading components are initialized on load
            if (_tradingBot == null || _stockDataCache == null)
                InitializeTradingComponents();

            // Initialize the cached symbol search timer
            InitializeCachedSymbolSearchTimer();

            // Load initial predictions from database
            await LoadInitialPredictionsAsync();
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
                // Use the Predictions ObservableCollection as the data source
                if (Predictions == null)
                    return;

                // Find the prediction for the selected symbol
                var prediction = Predictions.FirstOrDefault(p =>
                    p.Symbol.Equals(symbol, StringComparison.InvariantCultureIgnoreCase));
                if (prediction == null)
                {
                    // Optionally, fetch or create a new prediction for the symbol if not present
                    // For now, do nothing if not found
                    return;
                }

                // Check if the symbol is already in the grid
                var existing = Predictions.FirstOrDefault(p =>
                    p.Symbol.Equals(symbol, StringComparison.InvariantCultureIgnoreCase));
                if (existing != null)
                {
                    // Update the existing item in the ObservableCollection
                    var idx = Predictions.IndexOf(existing);
                    Predictions[idx] = prediction;
                }
                else
                {
                    // Add the new symbol to the collection
                    Predictions.Add(prediction);
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error adding/updating symbol in grid", ex.ToString());
            }
        }

        // Add this method to resolve the missing 'SelectAllSymbols' reference.
        private void SelectAllSymbols()
        {
            // Example logic: set the displayed symbols to all available symbols.
            try
            {
                // If you need to update a UI element, do so here.
                if (PredictionDataGrid != null)
                {
                    // Use the Predictions ObservableCollection
                    PredictionDataGrid.ItemsSource = Predictions;
                }

                // Optionally update status text or other UI elements.
                if (StatusText != null)
                    StatusText.Text = "All symbols selected.";
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error selecting all symbols", ex.ToString());
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

        // Handle tab selection changes to track user interaction
        private void ResultsTabControl_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            try
            {
                if (sender is TabControl tabControl)
                {
                    // Mark that the user is manually selecting a tab
                    _isUserSelectingTab = true;
                    _lastUserSelectedTabIndex = tabControl.SelectedIndex;

                    // Reset the flag after a short delay to allow the selection to complete
                    System.Windows.Threading.Dispatcher.CurrentDispatcher.BeginInvoke(
                        new Action(() => _isUserSelectingTab = false),
                        System.Windows.Threading.DispatcherPriority.Background);
                    
                    // Note: Removed any automatic tab navigation logic to allow user to freely select tabs
                    // The IndicatorsTabItem can now be selected and will remain selected
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error in ResultsTabControl_SelectionChanged", ex.ToString());
            }
        }

        // Handle radio button selection changes for view navigation
        private void ViewSelection_Changed(object sender, RoutedEventArgs e)
        {
            try
            {
                if (sender is RadioButton radioButton && radioButton.IsChecked == true)
                {
                    // Mark that the user is manually selecting a view
                    _isUserSelectingTab = true;

                    // Update the selected index based on which radio button was checked
                    if (radioButton.Name == "PredictionDetailsRadio")
                    {
                        _lastUserSelectedTabIndex = 0;
                    }
                    else if (radioButton.Name == "SentimentAnalysisRadio")
                    {
                        _lastUserSelectedTabIndex = 1;
                    }
                    else if (radioButton.Name == "TechnicalIndicatorsRadio")
                    {
                        _lastUserSelectedTabIndex = 2;
                    }

                    // Reset the flag after a short delay to allow the selection to complete
                    System.Windows.Threading.Dispatcher.CurrentDispatcher.BeginInvoke(
                        new Action(() => _isUserSelectingTab = false),
                        System.Windows.Threading.DispatcherPriority.Background);
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error in ViewSelection_Changed", ex.ToString());
            }
        }
    }
}
