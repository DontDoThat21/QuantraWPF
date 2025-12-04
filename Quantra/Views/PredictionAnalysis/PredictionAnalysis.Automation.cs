using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using Quantra.DAL.Services.Interfaces;
using System.Data.SQLite; // Added for SQLite commands
using Quantra;
using System.Windows.Threading;  // Add this explicit namespace reference
using Quantra.Adapters; // Add this to use the PredictionModelAdapter
using Quantra.Models;
using System.Collections.ObjectModel;
using Quantra.DAL.Services;
using Quantra.DAL.Data; // Added for ObservableCollection
using Quantra.Repositories; // Added for PredictionAnalysisRepository
using Microsoft.EntityFrameworkCore; // Added for DbContext

namespace Quantra.Controls
{
    public partial class PredictionAnalysis : UserControl, IDisposable
    {
     // Automated mode property
        private DispatcherTimer autoRefreshTimer;
        private CancellationTokenSource analysisTokenSource;
        private bool isDisposed = false;
     
        // Add repository for prediction management - will be initialized in OnControlLoaded
     private PredictionAnalysisRepository _predictionRepository;
  
        // Add service for prediction operations
        private PredictionAnalysisService _predictionService;
   
     // Constructor to initialize the repository and service (should be called from the OnControlLoaded)
  private void InitializePredictionRepository()
  {
            if (_predictionRepository != null)
       return; // Already initialized
     
            // Create the repository with parameterless constructor which handles DbContext internally
   _predictionRepository = new PredictionAnalysisRepository();
            
   // Get the prediction service from DI container or create with parameterless constructor as fallback
  _predictionService = App.ServiceProvider?.GetService(typeof(PredictionAnalysisService)) as PredictionAnalysisService
      ?? new PredictionAnalysisService();
        }

        public bool IsAutomatedMode
        {
            get => isAutomatedMode;
            set
            {
                if (isAutomatedMode != value)
                {
                    isAutomatedMode = value;
                    OnPropertyChanged(nameof(IsAutomatedMode));

                    // Save the auto mode state to user preferences
                    DatabaseMonolith.SaveUserPreference("PredictionAnalysisAutoMode", value.ToString());

                    // If turning on automated mode, trigger analysis and start timer
                    if (isAutomatedMode)
                    {
                        StartAutoMode();
                    }
                    else
                    {
                        StopAutoMode();
                    }
                }
            }
        }

        private void StartAutoMode()
        {
            // Cancel any existing analysis
            CancelAnalysisTask();

            // Create new cancellation token source
            analysisTokenSource = new CancellationTokenSource();
            
            // Run immediate analysis
            _ = GlobalLoadingStateService.WithLoadingState(RunAutomatedAnalysis());

            // Setup timer for periodic updates
            if (autoRefreshTimer == null)
            {
                autoRefreshTimer = new DispatcherTimer();
                autoRefreshTimer.Interval = TimeSpan.FromMinutes(10); // 10 minute refresh interval
                autoRefreshTimer.Tick += AutoRefreshTimer_Tick;
            }
            else if (autoRefreshTimer.IsEnabled)
            {
                // If timer is already running, stop it first to reset
                autoRefreshTimer.Stop();
            }
            
            autoRefreshTimer.Start();
            
            // Update status
            if (StatusText != null)
                StatusText.Text = "Auto Mode enabled. Analysis will refresh every 10 minutes.";

            //DatabaseMonolith.Log("Info", "Auto Mode enabled for PredictionAnalysisControl");
        }

        private void StopAutoMode()
        {
            // Stop timer with better error handling and more detailed logging
            try
            {
                if (autoRefreshTimer != null)
                {
                    if (autoRefreshTimer.IsEnabled)
                    {
                        autoRefreshTimer.Stop();
                        //DatabaseMonolith.Log("Info", "Auto refresh timer stopped successfully.");
                    }
                    else
                    {
                        //DatabaseMonolith.Log("Info", "Auto refresh timer was already stopped.");
                    }
                }
                
                // Cancel any running analysis (only if triggered by auto mode toggle)
                CancelAnalysisTask();
                
                // Update status
                if (StatusText != null)
                    StatusText.Text = "Auto Mode disabled. Manual refresh required.";
                    
                //DatabaseMonolith.Log("Info", "Auto Mode disabled for PredictionAnalysisControl");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error stopping auto mode", ex.ToString());
            }
        }
        
        private void AutoRefreshTimer_Tick(object sender, EventArgs e)
        {
            // First check if we're already disposed - safety guard
            if (isDisposed)
            {
                //DatabaseMonolith.Log("Warning", "Timer tick occurred after disposal. Stopping timer.");
                SafelyStopTimer();
                return;
            }
            
            // Check if we're in auto mode before running automated analysis
            if (isAutomatedMode)
            {
                try
                {
                    //DatabaseMonolith.Log("Info", "Auto refresh timer triggered analysis");
                    _ = GlobalLoadingStateService.WithLoadingState(RunAutomatedAnalysis());
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", "Error during auto refresh timer analysis", ex.ToString());
                }
            }
            else
            {
                // If auto mode was disabled but timer somehow still active, stop it
                //DatabaseMonolith.Log("Warning", "Auto refresh timer triggered but auto mode is disabled. Stopping timer.");
                SafelyStopTimer();
            }
        }

        // Cleanup resources
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        
        protected virtual void Dispose(bool disposing)
        {
            if (!isDisposed)
            {
                if (disposing)
                {
                    // Cleanup managed resources
                    StopAutoMode();
                    
                    if (autoRefreshTimer != null)
                    {
                        autoRefreshTimer.Tick -= AutoRefreshTimer_Tick;
                        autoRefreshTimer = null;
                    }
                    
                    if (analysisTokenSource != null)
                    {
                        analysisTokenSource.Dispose();
                        analysisTokenSource = null;
                    }
                }
                
                // Cleanup unmanaged resources
                
                isDisposed = true;
            }
        }
        
        // Method to cancel any running analysis task
        public void CancelAnalysisTask()
        {
            try
            {
                if (analysisTokenSource != null)
                {
                    if (!analysisTokenSource.IsCancellationRequested)
                    {
                        //DatabaseMonolith.Log("Info", "Cancelling ongoing analysis tasks");
                        analysisTokenSource.Cancel();
                    }
                    
                    // Dispose the token source
                    analysisTokenSource.Dispose();
                    analysisTokenSource = null;
                }
                
                // Only create a new token source if needed (when not disposing)
                // This prevents resource leaks from creating unnecessary token sources
                if (isAutomatedMode || !isDisposed)
                {
                    analysisTokenSource = new CancellationTokenSource();
                    //DatabaseMonolith.Log("Info", "Created new analysis token source");
                }
                
                //DatabaseMonolith.Log("Info", "Analysis tasks cancelled");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error cancelling analysis tasks", ex.ToString());
                
                // Ensure we have a valid token source even after an error
                if (analysisTokenSource == null && (isAutomatedMode || !isDisposed))
                {
                    analysisTokenSource = new CancellationTokenSource();
                }
            }
        }

        // Refactored method to load the latest predictions from the database using the service
        private async void LoadLatestPredictionsFromDatabase()
        {
     try
  {
      // Use the service to get latest predictions
  var latestPredictions = await _predictionService.GetLatestPredictionsAsync();
  
      if (latestPredictions != null && latestPredictions.Count > 0)
  {
        // Update UI on dispatcher thread
        await Dispatcher.InvokeAsync(() =>
   {
         // Clear existing predictions before loading new ones
       this.Predictions.Clear();
         
     foreach (var prediction in latestPredictions)
        {
      this.Predictions.Add(prediction);
      }

    if (StatusText != null)
  StatusText.Text = $"Loaded {this.Predictions.Count} cached predictions from database.";

if (LastUpdatedText != null)
         LastUpdatedText.Text = $"Last updated: {DateTime.Now.ToString("MM/dd/yyyy HH:mm")}";
         }, System.Windows.Threading.DispatcherPriority.Background);
    }
   }
    catch (Exception ex)
          {
 //_loggingService.Log("Error", "Failed to load cached predictions", ex.ToString());
  }
        }

        // Control loaded event handler to initiate automated analysis
        public void OnControlLoaded(object sender, RoutedEventArgs e)
        {
            // Initialize the prediction repository if not already initialized
            InitializePredictionRepository();
     
            // Load the auto mode state from user preferences
            string savedState = DatabaseMonolith.GetUserPreference("PredictionAnalysisAutoMode", "False");
            bool autoMode = bool.TryParse(savedState, out bool result) && result;
            
            // Update auto mode without triggering the property changed event
            if (autoMode != isAutomatedMode)
            {
                isAutomatedMode = autoMode;
                
                // Update the toggle button without triggering event
                if (AutoModeToggle != null)
                    AutoModeToggle.IsChecked = autoMode;
                
                // Start auto mode if enabled
                if (autoMode)
                {
                    StartAutoMode();
                    // Still load top predictions
                    _ = LoadTopPredictionsGridAsync();
                    return;
                }
            }

            // If not in auto mode, load cached predictions from database
            LoadLatestPredictionsFromDatabase();
            
            // Load top predictions grid
            _ = LoadTopPredictionsGridAsync();
        }

        /// <summary>
        /// Loads all predictions from the database into the TopPredictions grid
        /// </summary>
        private async Task LoadTopPredictionsGridAsync()
        {
            try
            {
                // Use the ViewModel if available - count text is bound via TopPredictionsCountText property
                if (_viewModel != null)
                {
                    await _viewModel.LoadTopPredictionsAsync();
                }
                else
                {
                    // Fallback: Load directly using service
                    if (_predictionService == null)
                    {
                        _predictionService = new PredictionAnalysisService();
                    }
                    
                    var predictions = await _predictionService.GetAllPredictionsAsync(1000);
                    
                    // Update the DataGrid directly
                    var grid = this.FindName("TopPredictionsGrid") as DataGrid;
                    if (grid != null)
                    {
                        await Dispatcher.InvokeAsync(() =>
                        {
                            grid.ItemsSource = predictions;
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to load top predictions grid");
            }
        }
        
        public void OnControlUnloaded(object sender, RoutedEventArgs e)
        {
            // Ensure we clean up when the control is unloaded
            StopAutoMode();
        }

        private async Task RunAutomatedAnalysis() // Changed from void to async Task
        {
            bool isManualTrigger = (autoRefreshTimer == null || !autoRefreshTimer.IsEnabled);
            
            if (!isAutomatedMode && !isManualTrigger)
            {
                //DatabaseMonolith.Log("Warning", "Automated analysis attempted while auto mode is disabled. Skipping.");
                SafelyStopTimer();
                return;
            }

            CancellationToken token = CancellationToken.None; // Default token
            try
            {
                if (analysisTokenSource == null || analysisTokenSource.IsCancellationRequested)
                {
                    analysisTokenSource = new CancellationTokenSource();
                    //DatabaseMonolith.Log("Info", "Created new analysis token source for analysis run");
                }
                token = analysisTokenSource.Token;
                
                await Dispatcher.InvokeAsync(() =>
                {
                    if (StatusText != null) StatusText.Text = "Running automated stock analysis...";
                    // The Predictions collection is Quantra.Models.PredictionModel
                    this.Predictions.Clear();
                });

                //EnsureDatabaseTablesExist();
                List<string> majorStocks = await FetchMajorUSStocks(token);

                token.ThrowIfCancellationRequested();
                
                if (majorStocks == null || majorStocks.Count == 0)
                {
                    //DatabaseMonolith.Log("Warning", "No stock symbols retrieved for automated analysis");
                    await Dispatcher.InvokeAsync(() =>
                    {
                        if (StatusText != null) StatusText.Text = "Failed to retrieve stock symbols for analysis";
                    });
                    return;
                }

                //DatabaseMonolith.Log("Info", $"Beginning automated analysis on {majorStocks.Count} stocks");

                int processedCount = 0;
                var batchedResults = new List<Quantra.Models.PredictionModel>();
                var lastStatusUpdate = DateTime.Now;
                
                foreach (var symbol in majorStocks)
                {
                    token.ThrowIfCancellationRequested();

                    try
                    {
                        // AnalyzeStockWithAllAlgorithms returns Quantra.Models.PredictionModel
                        Quantra.Models.PredictionModel analysisResult = await AnalyzeStockWithAllAlgorithms(symbol); 
                        if (analysisResult != null)
                        {
                            // Collect results for batched UI update
                            batchedResults.Add(analysisResult);
                            // Save to database immediately for data persistence (async, non-blocking)
                            await SavePredictionToDatabaseAsync(analysisResult).ConfigureAwait(false);
                        }
                    }
                    catch (OperationCanceledException) {
                        //DatabaseMonolith.Log("Info", $"Analysis for {symbol} was cancelled.");
                        break; 
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Error", $"Failed to analyze {symbol}", ex.ToString());
                    }
                    finally
                    {
                        processedCount++;
                        
                        // Throttle status updates - only update every 2 seconds or every 10 items
                        if (StatusText != null && !token.IsCancellationRequested && 
                            ((DateTime.Now - lastStatusUpdate).TotalSeconds >= 2.0 || processedCount % 10 == 0))
                        {
                            lastStatusUpdate = DateTime.Now;
                            int count = processedCount;
                            int total = majorStocks.Count;
                            await Dispatcher.InvokeAsync(() =>
                            {
                                if (StatusText != null)
                                    StatusText.Text = $"Analyzing stocks... ({count}/{total})";
                            });
                        }
                        
                        await Task.Delay(250, token); 
                    }
                }
                
                token.ThrowIfCancellationRequested();

                // Batch UI updates using dispatcher to minimize UI thread work
                await Dispatcher.InvokeAsync(() =>
                {
                    // Clear existing predictions
                    this.Predictions.Clear();
                    
                    // Sort and add all results in batch
                    var sortedResults = batchedResults.OrderByDescending(p => p.Confidence);
                    foreach (var prediction in sortedResults)
                    {
                        this.Predictions.Add(prediction);
                    }
                    
                    // Update status and timestamp
                    if (StatusText != null)
                        StatusText.Text = $"Automated analysis complete. Found {this.Predictions.Count} predictions";
                    if (LastUpdatedText != null)
                        LastUpdatedText.Text = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";
                }, System.Windows.Threading.DispatcherPriority.Background);

                //DatabaseMonolith.Log("Info", $"Automated analysis complete. Generated {this.Predictions.Count} predictions");

                if (isAutomatedMode && this.Predictions.Any())
                {
                    await Dispatcher.InvokeAsync(() =>
                    {
                        if (StatusText != null) StatusText.Text = "Executing trades based on predictions...";
                    });
                    // ExecuteTradesFromPredictionsAsync expects List<Quantra.Models.PredictionModel>
                    await ExecuteTradesFromPredictionsAsync(this.Predictions.ToList());
                    await Dispatcher.InvokeAsync(() =>
                    {
                        if (StatusText != null) StatusText.Text = "Automated trading complete.";
                    });
                }
            }
            catch (OperationCanceledException)
            {
                //DatabaseMonolith.Log("Info", "Automated analysis cancelled");
                await Dispatcher.InvokeAsync(() =>
                {
                    if (StatusText != null) StatusText.Text = "Analysis cancelled";
                });
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error during automated analysis", ex.ToString());
                await Dispatcher.InvokeAsync(() =>
                {
                    if (StatusText != null) StatusText.Text = "Error during automated analysis";
                });
            }
        }

        // Helper method to safely stop the timer
        private void SafelyStopTimer()
        {
            try
            {
                if (autoRefreshTimer != null)
                {
                    if (autoRefreshTimer.IsEnabled)
                    {
                        autoRefreshTimer.Stop();
                        //DatabaseMonolith.Log("Info", "Auto refresh timer stopped successfully.");
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error stopping timer", ex.ToString());
            }
        }

        private async Task SavePredictionToDatabaseAsync(Models.PredictionModel prediction)
        {
            try
            {
                // Validate required fields before attempting to insert
                if (string.IsNullOrWhiteSpace(prediction.Symbol) ||
                    string.IsNullOrWhiteSpace(prediction.PredictedAction) ||
                    double.IsNaN(prediction.Confidence) || double.IsInfinity(prediction.Confidence) ||
                    double.IsNaN(prediction.CurrentPrice) || double.IsInfinity(prediction.CurrentPrice) ||
                    double.IsNaN(prediction.TargetPrice) || double.IsInfinity(prediction.TargetPrice) ||
                    double.IsNaN(prediction.PotentialReturn) || double.IsInfinity(prediction.PotentialReturn))
                {
                    _loggingService?.Log("Error", $"Invalid prediction data for {prediction?.Symbol ?? "<null>"}. Skipping insert.");
                    string symbol = prediction?.Symbol;
                    await Dispatcher.InvokeAsync(() =>
                    {
                        if (StatusText != null)
                            StatusText.Text = $"Skipped invalid prediction for {symbol}";
                    });
                    return;
                }

                // Log the save attempt
                _loggingService?.Log("Info", $"Attempting to save prediction for {prediction.Symbol}: {prediction.PredictedAction} @ {prediction.Confidence:P0}");

                // Save with timeout protection (10 seconds)
                using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10)))
                {
                    if (_predictionService != null)
                    {
                        // ConfigureAwait(false) prevents deadlock by not capturing synchronization context
                        var id = await _predictionService.SavePredictionAsync(prediction, cts.Token).ConfigureAwait(false);
                        _loggingService?.Log("Info", $"Successfully saved prediction ID {id} for {prediction.Symbol}");
                    }
                    else
                    {
                        _loggingService?.Log("Error", "PredictionService is null - cannot save prediction");
                    }
                }
            }
            catch (OperationCanceledException)
            {
                _loggingService?.Log("Warning", $"Timeout saving prediction for {prediction?.Symbol ?? "<null>"}");
                string symbol = prediction?.Symbol;
                await Dispatcher.InvokeAsync(() =>
                {
                    if (StatusText != null)
                        StatusText.Text = $"Timeout saving {symbol}";
                });
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to save prediction for {prediction?.Symbol ?? "<null>"}");
                string symbol = prediction?.Symbol;
                string message = ex.Message;
                await Dispatcher.InvokeAsync(() =>
                {
                    if (StatusText != null)
                        StatusText.Text = $"Error saving {symbol}: {message}";
                });
            }
        }

        private async Task<List<string>> FetchMajorUSStocks(CancellationToken token)
        {
     try
            {
         // Use the default fallback list - in production this would come from a proper stock symbol service
    var defaultSymbols = new List<string> { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA" };
          //_loggingService.Log("Info", $"Using default list of {defaultSymbols.Count} symbols.");
   return defaultSymbols;
            }
      catch (OperationCanceledException)
      {
      //_loggingService.Log("Info", "Fetching major US stocks was cancelled.");
         return new List<string>();
   }
         catch (Exception ex)
            {
 //_loggingService.Log("Error", "Failed to fetch US stock symbols", ex.ToString());
    var defaultSymbols = new List<string> { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA" };
      return defaultSymbols;
         }
        }

        // Add this public method to allow external/manual triggering of prediction algorithms
        public void RunPredictionAlgorithms()
   {
       _ = GlobalLoadingStateService.WithLoadingState(RunAutomatedAnalysis());
        }

        // Add this public async method to run prediction algorithms for a specific symbol
public async Task<Quantra.Models.PredictionModel> RunPredictionAlgorithms(string symbol)
   {
            // This method runs the same logic as AnalyzeStockWithAllAlgorithms for the given symbol
          // and returns a PredictionModel result.
         return await AnalyzeStockWithAllAlgorithms(symbol);
        }
    }
}