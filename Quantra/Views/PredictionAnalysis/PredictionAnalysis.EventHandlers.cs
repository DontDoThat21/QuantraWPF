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
using Microsoft.EntityFrameworkCore;

namespace Quantra.Controls
{
    public partial class PredictionAnalysis : UserControl
    {
        // Add this field to ensure selectedPrediction is available in this partial class
        // private Quantra.Models.PredictionModel selectedPrediction;

        // Changed from List<PredictionModel> to ObservableCollection<PredictionModel>
        // Renamed to Predictions for clarity and to avoid conflict if xaml.cs has 'predictions'
        public ObservableCollection<Quantra.Models.PredictionModel> Predictions { get; set; } = new ObservableCollection<Quantra.Models.PredictionModel>();

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

                // Check if TFT architecture is selected
                string selectedArchitecture = GetSelectedArchitectureType();
                
                Quantra.Models.PredictionModel prediction = null;
                
                if (selectedArchitecture == TFT_ARCHITECTURE_TYPE && _realTimeInferenceService != null)
                {
                    // Use TFT-specific prediction path
                    if (StatusText != null)
                        StatusText.Text = $"Running TFT prediction for {symbol}...";
                    
                    var tftResult = await _realTimeInferenceService.GetTFTPredictionAsync(
                        symbol,
                        lookbackDays: TFT_DEFAULT_LOOKBACK_DAYS,
                        futureHorizon: TFT_DEFAULT_FUTURE_HORIZON,
                        progressCallback: (msg) => Dispatcher.InvokeAsync(() => 
                        {
                            if (StatusText != null)
                                StatusText.Text = msg;
                        })
                    );
                    
                    if (tftResult.Success && tftResult.Prediction != null)
                    {
                        // Convert PredictionResult to PredictionModel
                        prediction = new Quantra.Models.PredictionModel
                        {
                            Symbol = tftResult.Prediction.Symbol,
                            PredictedAction = tftResult.Prediction.Action,
                            Confidence = tftResult.Prediction.Confidence,
                            TargetPrice = tftResult.Prediction.TargetPrice,
                            CurrentPrice = tftResult.Prediction.CurrentPrice,
                            PredictionDate = tftResult.Prediction.PredictionDate,
                            ModelType = TFT_ARCHITECTURE_TYPE,
                            ArchitectureType = TFT_ARCHITECTURE_TYPE
                        };
                        
                        // Update TFT visualization in ViewModel if available
                        if (tftResult.TFTResult != null && _viewModel != null)
                        {
                            // Get historical prices for context visualization
                            var historicalPrices = await _stockDataCacheService?.GetRecentHistoricalSequenceAsync(symbol, TFT_HISTORICAL_VISUALIZATION_DAYS);
                            var priceList = historicalPrices?.Select(p => p.Close).ToList();
                            
                            await Dispatcher.InvokeAsync(() =>
                            {
                                _viewModel.UpdateTFTVisualization(tftResult.TFTResult, priceList);
                            });
                        }
                    }
                    else
                    {
                        // ENHANCED ERROR HANDLING: Parse and analyze the error message
                        var errorMsg = tftResult.ErrorMessage ?? "Unknown error";
                        var detailedError = $"TFT prediction failed for {symbol}. Success={tftResult.Success}, Prediction={tftResult.Prediction != null}, Error={errorMsg}";
                        _loggingService?.Log("Warning", detailedError);
                        
                        if (StatusText != null)
                            StatusText.Text = $"TFT prediction failed: {errorMsg}";
                        
                        // Parse feature dimension mismatch errors
                        string diagnosticInfo = "";
                        bool isFeatureMismatch = false;
                        
                        if (errorMsg.Contains("features") && errorMsg.Contains("expecting"))
                        {
                            isFeatureMismatch = true;
                            
                            // Extract feature counts from error message
                            // Example: "X has 9 features, but StandardScaler is expecting 15 features"
                            var match = System.Text.RegularExpressions.Regex.Match(
                                errorMsg, 
                                @"X has (\d+) features.*expecting (\d+) features"
                            );
                            
                            if (match.Success && match.Groups.Count >= 3)
                            {
                                string currentFeatures = match.Groups[1].Value;
                                string expectedFeatures = match.Groups[2].Value;
                                
                                diagnosticInfo = $"\n\n?? FEATURE DIMENSION MISMATCH DETECTED:\n" +
                                               $"??????????????????????????????????????\n" +
                                               $"Current data:     {currentFeatures} features\n" +
                                               $"Model expects:    {expectedFeatures} features\n" +
                                               $"Missing features: {int.Parse(expectedFeatures) - int.Parse(currentFeatures)}\n\n" +
                                               $"?? DIAGNOSIS:\n" +
                                               $"The TFT model was trained with {expectedFeatures} features,\n" +
                                               $"but the prediction data only has {currentFeatures} features.\n" +
                                               $"This usually means:\n" +
                                               $"  � Different feature engineering during training vs prediction\n" +
                                               $"  � Model was trained with advanced features\n" +
                                               $"  � Prediction is using basic OHLCV + simple indicators\n\n" +
                                               $"?? SOLUTION:\n" +
                                               $"You need to RETRAIN the TFT model to match current features.\n" +
                                               $"The training will use the same feature engineering as prediction.\n\n" +
                                               $"?? HOW TO FIX:\n" +
                                               $"1. Go to Model Training tab\n" +
                                               $"2. Select 'PyTorch' model type\n" +
                                               $"3. Select 'TFT' architecture\n" +
                                               $"4. Click 'Train Model'\n" +
                                               $"5. Wait for training to complete (~5-10 minutes)\n" +
                                               $"6. Try prediction again\n\n" +
                                               $"Or run from command line:\n" +
                                               $"python Quantra/python/train_from_database.py --model_type pytorch --architecture_type tft --epochs 50";
                            }
                        }
                        
                        // Show comprehensive error dialog
                        var errorTitle = isFeatureMismatch ? 
                            "TFT Feature Dimension Mismatch" : 
                            "TFT Prediction Error";
                        
                        var errorDetails = $"Symbol: {symbol}\n" +
                                         $"Success: {tftResult.Success}\n" +
                                         $"Prediction: {(tftResult.Prediction != null ? "Available" : "NULL")}\n" +
                                         $"\nError Message:\n{errorMsg}";
                        
                        if (isFeatureMismatch)
                        {
                            errorDetails += diagnosticInfo;
                        }
                        else
                        {
                            errorDetails += $"\n\n" +
                                          $"Common Issues:\n" +
                                          $"1. TFT model file missing in python/models/ directory\n" +
                                          $"2. Python environment missing packages (torch, darts)\n" +
                                          $"3. Model was trained with incorrect architecture\n" +
                                          $"4. Historical data insufficient (need 60+ days)\n" +
                                          $"5. Feature engineering pipeline mismatch";
                        }
                        
                        // Log detailed diagnostic info
                        _loggingService?.Log("Error", $"TFT Prediction Diagnostic Info:\n{errorDetails}");
                        
                        MessageBox.Show(
                            errorDetails,
                            errorTitle,
                            MessageBoxButton.OK,
                            isFeatureMismatch ? MessageBoxImage.Warning : MessageBoxImage.Error);
                    }
                }
                else
                {
                    // Use traditional prediction path (LSTM, Random Forest, etc.)
                    prediction = await AnalyzeStockWithAllAlgorithms(symbol);
                }

                if (prediction != null && prediction.PredictedAction != "ERROR")
                {
                    // Save prediction to database with expected fruition date and model info
                    await SavePredictionWithModelInfoAsync(prediction);

                    // Add to UI collection on UI thread
                    await Dispatcher.InvokeAsync(() =>
                    {
                        Predictions.Add(prediction);
                        
                        // Also add to the TopPredictions grid if ViewModel is available
                        // The count text is bound to ViewModel.TopPredictionsCountText
                        _viewModel?.AddToTopPredictions(prediction);

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
                
                // Update the candlestick chart with OHLCV data and predictions
                await UpdateChartWithPredictionsAsync(symbol);
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

        /// <summary>
        /// Saves a prediction to the database with model and expected fruition date info
        /// </summary>
        private async Task SavePredictionWithModelInfoAsync(Quantra.Models.PredictionModel prediction)
        {
            try
            {
                // Get selected model type, architecture, and expected fruition date
                string modelType = GetSelectedModelType();
                string architectureType = GetSelectedArchitectureType();
                DateTime? expectedFruitionDate = GetExpectedFruitionDate();

                // Get active training history ID
                int? trainingHistoryId = null;
                try
                {
                    if (_modelTrainingHistoryService != null)
                    {
                        var activeModel = await _modelTrainingHistoryService.GetActiveModelAsync(modelType, architectureType);
                        trainingHistoryId = activeModel?.Id;
                    }
                }
                catch (Exception ex)
                {
                    _loggingService?.Log("Warning", $"Could not get active model ID: {ex.Message}");
                }

                // Save with timeout protection (10 seconds)
                using (var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromSeconds(10)))
                {
                    if (_predictionService != null)
                    {
                        var id = await _predictionService.SavePredictionAsync(
                            prediction, 
                            cts.Token,
                            expectedFruitionDate,
                            modelType,
                            architectureType,
                            trainingHistoryId
                        ).ConfigureAwait(false);
                        
                        _loggingService?.Log("Info", $"Successfully saved prediction ID {id} for {prediction.Symbol} with expected fruition date {expectedFruitionDate:d}");
                    }
                    else
                    {
                        // Fallback to the old method if _predictionService is not available
                        await SavePredictionToDatabaseAsync(prediction);
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Error saving prediction with model info for {prediction.Symbol}");
                // Fall back to the old method
                await SavePredictionToDatabaseAsync(prediction);
            }
        }

        private async void AnalyzeButton_Click(object sender, RoutedEventArgs e)
        {
            // First, check if a trained model is available
            var modelAvailability = await CheckTrainedModelAvailabilityAsync();
            if (!modelAvailability.IsModelAvailable)
            {
                // If model file exists but no DB record, try to register it automatically
                if (modelAvailability.HasLocalModelFile && !modelAvailability.HasDatabaseRecord)
                {
                    if (StatusText != null)
                        StatusText.Text = "Found model file without database record. Registering...";

                    _loggingService?.Log("Info", "Attempting automatic registration of existing TFT model...");

                    bool registered = await RegisterExistingTFTModelAsync();
                    if (registered)
                    {
                        // Re-check availability after registration
                        modelAvailability = await CheckTrainedModelAvailabilityAsync();
                        
                        if (modelAvailability.IsModelAvailable)
                        {
                            _loggingService?.Log("Info", "Successfully registered and verified TFT model");
                            if (StatusText != null)
                                StatusText.Text = "TFT model registered successfully. Proceeding with analysis...";
                        }
                    }
                    else
                    {
                        _loggingService?.Log("Error", "Failed to register TFT model automatically");
                    }
                }

                // If still not available after registration attempt, show error
                if (!modelAvailability.IsModelAvailable)
                {
                    if (StatusText != null)
                        StatusText.Text = modelAvailability.StatusMessage;

                    MessageBox.Show(
                        $"{modelAvailability.StatusMessage}\n\nPlease use the 'Train Model' button to train a model before running analysis.",
                        "Model Not Available",
                        MessageBoxButton.OK,
                        MessageBoxImage.Warning);
                    return;
                }
            }

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
                    // Note: Without PredictionDataGrid, we just show the count
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

        /// <summary>
        /// Manually registers an existing TFT model file in the database
        /// This creates a database record for an existing model file
        /// </summary>
        private async Task<bool> RegisterExistingTFTModelAsync()
        {
            try
            {
                _loggingService?.Log("Info", "Attempting to register existing TFT model in database...");

                // Verify the model file exists
                string pythonModelsDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "models");
                string modelFilePath = System.IO.Path.Combine(pythonModelsDir, "tft_model.pt");

                if (!System.IO.File.Exists(modelFilePath))
                {
                    _loggingService?.Log("Error", $"TFT model file not found at: {modelFilePath}");
                    return false;
                }

                // Get file info
                var fileInfo = new System.IO.FileInfo(modelFilePath);
                _loggingService?.Log("Info", $"Found TFT model file: {modelFilePath}, Size: {fileInfo.Length} bytes");

                // Create a database record using EF Core
                var optionsBuilder = new DbContextOptionsBuilder<Quantra.DAL.Data.QuantraDbContext>();
                optionsBuilder.UseSqlServer(Quantra.DAL.Data.ConnectionHelper.ConnectionString);
                
                using (var dbContext = new Quantra.DAL.Data.QuantraDbContext(optionsBuilder.Options))
                {
                    // Check if a record already exists
                    var existingRecord = await dbContext.ModelTrainingHistory
                        .Where(m => m.ModelType == "pytorch" && m.ArchitectureType == "tft" && m.IsActive)
                        .FirstOrDefaultAsync();

                    if (existingRecord != null)
                    {
                        _loggingService?.Log("Info", "TFT model record already exists in database");
                        return true;
                    }

                    // Deactivate any existing active models for this type
                    var activeModels = await dbContext.ModelTrainingHistory
                        .Where(m => m.ModelType == "pytorch" && m.ArchitectureType == "tft" && m.IsActive)
                        .ToListAsync();

                    foreach (var model in activeModels)
                    {
                        model.IsActive = false;
                    }

                    // Create new training history record
                    var newRecord = new Quantra.DAL.Data.Entities.ModelTrainingHistory
                    {
                        ModelType = "pytorch",
                        ArchitectureType = "tft",
                        TrainingDate = fileInfo.CreationTime,
                        IsActive = true,
                        SymbolsCount = 0,
                        TrainingSamples = 0,
                        TestSamples = 0,
                        TrainingTimeSeconds = 0.0,
                        MAE = 0.0,
                        RMSE = 0.0,
                        R2Score = 0.0,
                        Notes = $"Manually registered existing TFT model file. Path: {modelFilePath}"
                    };

                    dbContext.ModelTrainingHistory.Add(newRecord);
                    await dbContext.SaveChangesAsync();

                    _loggingService?.Log("Info", $"Successfully registered TFT model in database with ID: {newRecord.Id}");
                    
                    if (StatusText != null)
                        StatusText.Text = $"Successfully registered TFT model (ID: {newRecord.Id})";
                    
                    return true;
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error registering existing TFT model");
                return false;
            }
        }

        /// <summary>
        /// Checks if a trained model is available (both DB record and local file)
        /// </summary>
        private async Task<TrainedModelAvailability> CheckTrainedModelAvailabilityAsync()
        {
            try
            {
                // Get selected model type and architecture from UI
                string modelType = GetSelectedModelType();
                string architectureType = GetSelectedArchitectureType();

                // Check for model availability using the service
                if (_modelTrainingHistoryService != null)
                {
                    var availability = await _modelTrainingHistoryService.CheckTrainedModelAvailabilityAsync(modelType, architectureType);
                    
                    // ENHANCED: Add feature dimension check for TFT models
                    if (availability.IsModelAvailable && architectureType.ToLower() == "tft")
                    {
                        await DiagnoseTFTModelFeaturesAsync(availability);
                    }
                    
                    return availability;
                }

                // Fallback: Create the service if not available (with proper disposal)
                var optionsBuilder = new Microsoft.EntityFrameworkCore.DbContextOptionsBuilder<Quantra.DAL.Data.QuantraDbContext>();
                optionsBuilder.UseSqlServer(Quantra.DAL.Data.ConnectionHelper.ConnectionString);
                using (var dbContext = new Quantra.DAL.Data.QuantraDbContext(optionsBuilder.Options))
                {
                    var historyService = new ModelTrainingHistoryService(dbContext, _loggingService);
                    var availability = await historyService.CheckTrainedModelAvailabilityAsync(modelType, architectureType);
                    
                    if (availability.IsModelAvailable && architectureType.ToLower() == "tft")
                    {
                        await DiagnoseTFTModelFeaturesAsync(availability);
                    }
                    
                    return availability;
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error checking trained model availability");
                return new TrainedModelAvailability
                {
                    IsModelAvailable = false,
                    StatusMessage = $"Error checking model availability: {ex.Message}"
                };
            }
        }
        
        /// <summary>
        /// Diagnose TFT model feature dimensions by checking the saved model metadata
        /// </summary>
        private async Task DiagnoseTFTModelFeaturesAsync(TrainedModelAvailability availability)
        {
            try
            {
                string pythonModelsDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "models");
                string modelFilePath = System.IO.Path.Combine(pythonModelsDir, "tft_model.pt");
                
                if (!System.IO.File.Exists(modelFilePath))
                {
                    _loggingService?.Log("Warning", "TFT model file not found for feature dimension check");
                    return;
                }
                
                // Create a simple Python script to check model dimensions
                string tempScriptPath = System.IO.Path.GetTempFileName() + ".py";
                string tempOutputPath = System.IO.Path.GetTempFileName();
                
                string pythonScript = @"
import sys
import json
import torch

try:
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    result = {
        'success': True,
        'input_dim': int(checkpoint.get('input_dim', 0)),
        'static_dim': int(checkpoint.get('static_dim', 0)),
        'hidden_dim': int(checkpoint.get('hidden_dim', 0)),
        'forecast_horizons': checkpoint.get('forecast_horizons', []),
        'is_trained': checkpoint.get('is_trained', False),
        'architecture_type': checkpoint.get('architecture_type', 'unknown')
    }
    
    # Try to get scaler info
    try:
        import joblib
        scaler_path = model_path.replace('tft_model.pt', 'tft_scaler.pkl')
        scalers = joblib.load(scaler_path)
        if 'scaler' in scalers:
            result['scaler_features'] = int(scalers['scaler'].n_features_in_)
    except:
        result['scaler_features'] = 0
    
    with open(output_path, 'w') as f:
        json.dump(result, f)
        
except Exception as e:
    with open(output_path, 'w') as f:
        json.dump({'success': False, 'error': str(e)}, f)
";
                
                await System.IO.File.WriteAllTextAsync(tempScriptPath, pythonScript);
                
                // Execute Python script
                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{tempScriptPath}\" \"{modelFilePath}\" \"{tempOutputPath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };
                
                using (var process = System.Diagnostics.Process.Start(psi))
                {
                    await process.WaitForExitAsync();
                    
                    if (System.IO.File.Exists(tempOutputPath))
                    {
                        string jsonResult = await System.IO.File.ReadAllTextAsync(tempOutputPath);
                        var result = System.Text.Json.JsonSerializer.Deserialize<System.Text.Json.JsonElement>(jsonResult);
                        
                        if (result.TryGetProperty("success", out var success) && success.GetBoolean())
                        {
                            int inputDim = result.GetProperty("input_dim").GetInt32();
                            int scalerFeatures = result.TryGetProperty("scaler_features", out var sf) ? sf.GetInt32() : 0;
                            
                            string diagnosticMsg = $"TFT Model Feature Dimensions:\n" +
                                                 $"  Input Dim: {inputDim}\n" +
                                                 $"  Scaler Expects: {scalerFeatures} features\n";
                            
                            _loggingService?.Log("Info", diagnosticMsg);
                            
                            // Add to availability status message
                            availability.StatusMessage += $"\n{diagnosticMsg}";
                            
                            // Warn if there's a mismatch indicator
                            if (scalerFeatures > 0 && scalerFeatures != inputDim)
                            {
                                string warningMsg = $"?? WARNING: Feature dimension inconsistency detected!\n" +
                                                  $"Model input_dim ({inputDim}) != Scaler features ({scalerFeatures})\n" +
                                                  $"This may cause prediction failures.";
                                _loggingService?.Log("Warning", warningMsg);
                                availability.StatusMessage += $"\n{warningMsg}";
                            }
                        }
                    }
                }
                
                // Cleanup
                try
                {
                    if (System.IO.File.Exists(tempScriptPath)) System.IO.File.Delete(tempScriptPath);
                    if (System.IO.File.Exists(tempOutputPath)) System.IO.File.Delete(tempOutputPath);
                }
                catch { }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Could not diagnose TFT model features: {ex.Message}");
            }
        }

        /// <summary>
        /// Calculates the expected fruition date based on the selected timeframe
        /// Uses DateTime.UtcNow for consistency across time zones
        /// </summary>
        private DateTime? GetExpectedFruitionDate()
        {
            try
            {
                var selectedTimeframe = TimeframeComboBox?.SelectedItem as ComboBoxItem;
                if (selectedTimeframe?.Tag is string timeframeTag)
                {
                    // Use UTC time for consistency across time zones
                    return timeframeTag.ToLower() switch
                    {
                        "1day" => DateTime.UtcNow.AddDays(1),
                        "1week" => DateTime.UtcNow.AddDays(7),
                        "1month" => DateTime.UtcNow.AddMonths(1),
                        "3month" => DateTime.UtcNow.AddMonths(3),
                        "1year" => DateTime.UtcNow.AddYears(1),
                        _ => DateTime.UtcNow.AddMonths(1) // Default to 1 month
                    };
                }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Error getting expected fruition date: {ex.Message}");
            }

            return DateTime.Now.AddMonths(1); // Default to 1 month
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

        /// <summary>
        /// Handles the delete prediction context menu click
        /// </summary>
        private async void DeletePrediction_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Get the selected prediction from the DataGrid
                if (TopPredictionsGrid?.SelectedItem is Quantra.Models.PredictionModel selectedPrediction)
                {
                    // Confirm deletion with user
                    var result = MessageBox.Show(
                        $"Are you sure you want to delete the prediction for {selectedPrediction.Symbol}?\n\n" +
                        $"Action: {selectedPrediction.PredictedAction}\n" +
                        $"Confidence: {selectedPrediction.Confidence:P1}\n" +
                        $"Created: {selectedPrediction.PredictionDate:yyyy-MM-dd HH:mm}",
                        "Delete Prediction",
                        MessageBoxButton.YesNo,
                        MessageBoxImage.Question);

                    if (result == MessageBoxResult.Yes)
                    {
                        // Delete from database using the service
                        if (_predictionService != null)
                        {
                            bool deleted = await _predictionService.DeletePredictionAsync(selectedPrediction.Id);

                            if (deleted)
                            {
                                // Remove from the ViewModel collection
                                if (_viewModel?.TopPredictions != null)
                                {
                                    _viewModel.TopPredictions.Remove(selectedPrediction);
                                }

                                // Update status
                                if (StatusText != null)
                                    StatusText.Text = $"Deleted prediction for {selectedPrediction.Symbol}";

                                _loggingService?.Log("Info", $"User deleted prediction ID {selectedPrediction.Id} for {selectedPrediction.Symbol}");
                            }
                            else
                            {
                                if (StatusText != null)
                                    StatusText.Text = $"Failed to delete prediction for {selectedPrediction.Symbol}";

                                MessageBox.Show(
                                    $"Failed to delete the prediction for {selectedPrediction.Symbol}. Please try again.",
                                    "Delete Failed",
                                    MessageBoxButton.OK,
                                    MessageBoxImage.Error);
                            }
                        }
                        else
                        {
                            MessageBox.Show(
                                "Prediction service is not available. Cannot delete prediction.",
                                "Service Unavailable",
                                MessageBoxButton.OK,
                                MessageBoxImage.Error);
                        }
                    }
                }
                else
                {
                    MessageBox.Show(
                        "Please select a prediction to delete.",
                        "No Selection",
                        MessageBoxButton.OK,
                        MessageBoxImage.Information);
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error deleting prediction");

                if (StatusText != null)
                    StatusText.Text = "Error deleting prediction";

                MessageBox.Show(
                    $"An error occurred while deleting the prediction: {ex.Message}",
                    "Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// Handles the delete all predictions button click
        /// </summary>
        private async void DeleteAllPredictionsButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Check if there are any predictions to delete
                if (_viewModel?.TopPredictions == null || _viewModel.TopPredictions.Count == 0)
                {
                    MessageBox.Show(
                        "There are no predictions to delete.",
                        "No Predictions",
                        MessageBoxButton.OK,
                        MessageBoxImage.Information);
                    return;
                }

                int predictionCount = _viewModel.TopPredictions.Count;

                // Confirm deletion with user
                var result = MessageBox.Show(
                    $"Are you sure you want to delete ALL {predictionCount} predictions?\n\n" +
                    $"This action cannot be undone.\n\n" +
                    $"All prediction data will be permanently removed from the database.",
                    "Delete All Predictions",
                    MessageBoxButton.YesNo,
                    MessageBoxImage.Warning);

                if (result == MessageBoxResult.Yes)
                {
                    // Update status
                    if (StatusText != null)
                        StatusText.Text = $"Deleting all {predictionCount} predictions...";

                    // Disable the button while deleting
                    if (DeleteAllPredictionsButton != null)
                        DeleteAllPredictionsButton.IsEnabled = false;

                    try
                    {
                        // Delete all predictions from database using the service
                        if (_predictionService != null)
                        {
                            // Get all prediction IDs
                            var predictionIds = _viewModel.TopPredictions.Select(p => p.Id).ToList();
                            int deletedCount = 0;
                            int failedCount = 0;

                            // Delete each prediction
                            foreach (var predictionId in predictionIds)
                            {
                                bool deleted = await _predictionService.DeletePredictionAsync(predictionId);
                                if (deleted)
                                    deletedCount++;
                                else
                                    failedCount++;
                            }

                            // Clear the ViewModel collection
                            if (_viewModel?.TopPredictions != null)
                            {
                                _viewModel.TopPredictions.Clear();
                            }

                            // Update status
                            if (StatusText != null)
                            {
                                if (failedCount == 0)
                                    StatusText.Text = $"Successfully deleted all {deletedCount} predictions";
                                else
                                    StatusText.Text = $"Deleted {deletedCount} predictions, {failedCount} failed";
                            }

                            _loggingService?.Log("Info", $"User deleted all predictions: {deletedCount} succeeded, {failedCount} failed");

                            // Show success message
                            if (failedCount == 0)
                            {
                                MessageBox.Show(
                                    $"Successfully deleted all {deletedCount} predictions.",
                                    "Delete Successful",
                                    MessageBoxButton.OK,
                                    MessageBoxImage.Information);
                            }
                            else
                            {
                                MessageBox.Show(
                                    $"Deleted {deletedCount} predictions successfully.\n{failedCount} predictions failed to delete.",
                                    "Partial Success",
                                    MessageBoxButton.OK,
                                    MessageBoxImage.Warning);
                            }
                        }
                        else
                        {
                            MessageBox.Show(
                                "Prediction service is not available. Cannot delete predictions.",
                                "Service Unavailable",
                                MessageBoxButton.OK,
                                MessageBoxImage.Error);

                            if (StatusText != null)
                                StatusText.Text = "Error: Prediction service unavailable";
                        }
                    }
                    finally
                    {
                        // Re-enable the button
                        if (DeleteAllPredictionsButton != null)
                            DeleteAllPredictionsButton.IsEnabled = true;
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error deleting all predictions");

                if (StatusText != null)
                    StatusText.Text = "Error deleting all predictions";

                MessageBox.Show(
                    $"An error occurred while deleting all predictions: {ex.Message}",
                    "Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);

                // Re-enable the button in case of error
                if (DeleteAllPredictionsButton != null)
                    DeleteAllPredictionsButton.IsEnabled = true;
            }
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
                // Predictions ObservableCollection is already available for binding
                // No need to manually set ItemsSource

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
    }
}
