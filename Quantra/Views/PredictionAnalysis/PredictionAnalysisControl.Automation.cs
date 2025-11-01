using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using Quantra.DAL.Services.Interfaces;
//using System.Data.SQLite;
using Quantra;
using System.Windows.Threading;  // Add this explicit namespace reference
using Quantra.Adapters; // Add this to use the PredictionModelAdapter
using Quantra.Models;
using System.Collections.ObjectModel;
using Quantra.DAL.Services; // Added for ObservableCollection

namespace Quantra.Controls
{
    public partial class PredictionAnalysisControl : UserControl, IDisposable
    {
        // Automated mode property
        private DispatcherTimer autoRefreshTimer;
        private CancellationTokenSource analysisTokenSource;
        private bool isDisposed = false;
        
        // Add this field for AlphaVantageService
        private readonly AlphaVantageService alphaVantageService = new AlphaVantageService();

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

        // Add this method to load the latest predictions from the database and display them in the UI
        private void LoadLatestPredictionsFromDatabase()
        {
            try
            {
                var latestDataPredictions = GetLatestPredictionsFromDatabase(); // This returns List<Quantra.Models.PredictionModel>
                if (latestDataPredictions != null && latestDataPredictions.Count > 0)
                {
                    // Clear existing predictions before loading new ones
                    this.Predictions.Clear(); 
                    foreach (Quantra.Models.PredictionModel dataModel in latestDataPredictions)
                    {
                        // No conversion needed, just add the model
                        this.Predictions.Add(dataModel);
                    }

                    if (PredictionDataGrid != null)
                        PredictionDataGrid.ItemsSource = this.Predictions;

                    if (StatusText != null)
                        StatusText.Text = $"Loaded {this.Predictions.Count} cached predictions from database.";

                    if (LastUpdatedText != null)
                        LastUpdatedText.Text = $"Last updated: {DateTime.Now.ToString("MM/dd/yyyy HH:mm")}";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to load cached predictions", ex.ToString());
            }
        }

        // Add this method to retrieve the latest predictions from the database
        private List<Quantra.Models.PredictionModel> GetLatestPredictionsFromDatabase()
        {
            var result = new List<Quantra.Models.PredictionModel>();
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    // Get the most recent prediction for each symbol
                    string sql = @"
                        SELECT p1.*
                        FROM StockPredictions p1
                        INNER JOIN (
                            SELECT Symbol, MAX(CreatedDate) AS MaxDate
                            FROM StockPredictions
                            GROUP BY Symbol
                        ) p2 ON p1.Symbol = p2.Symbol AND p1.CreatedDate = p2.MaxDate
                        ORDER BY p1.Confidence DESC
                    ";
                    using (var command = new SQLiteCommand(sql, connection))
                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            var model = new Quantra.Models.PredictionModel
                            {
                                Symbol = reader["Symbol"].ToString(),
                                PredictedAction = reader["PredictedAction"].ToString(),
                                Confidence = Convert.ToDouble(reader["Confidence"]),
                                CurrentPrice = Convert.ToDouble(reader["CurrentPrice"]),
                                TargetPrice = Convert.ToDouble(reader["TargetPrice"]),
                                PotentialReturn = Convert.ToDouble(reader["PotentialReturn"]),
                                PredictionDate = Convert.ToDateTime(reader["CreatedDate"]),
                                TradingRule = reader["TradingRule"]?.ToString(),
                                Indicators = new Dictionary<string, double>()
                            };

                            // Load indicators for this prediction
                            long predictionId = Convert.ToInt64(reader["Id"]);
                            using (var indCmd = new SQLiteCommand("SELECT IndicatorName, IndicatorValue FROM PredictionIndicators WHERE PredictionId = @PredictionId", connection))
                            {
                                indCmd.Parameters.AddWithValue("@PredictionId", predictionId);
                                using (var indReader = indCmd.ExecuteReader())
                                {
                                    while (indReader.Read())
                                    {
                                        string name = indReader["IndicatorName"].ToString();
                                        double value = Convert.ToDouble(indReader["IndicatorValue"]);
                                        model.Indicators[name] = value;
                                    }
                                }
                            }

                            result.Add(model);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error retrieving latest predictions from database", ex.ToString());
            }
            return result;
        }

        // Control loaded event handler to initiate automated analysis
        public void OnControlLoaded(object sender, RoutedEventArgs e)
        {
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
                    return;
                }
            }

            // If not in auto mode, load cached predictions from database
            LoadLatestPredictionsFromDatabase();
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
                
                if (StatusText != null) StatusText.Text = "Running automated stock analysis...";
                // The Predictions collection is Quantra.Models.PredictionModel
                this.Predictions.Clear(); 
                if (PredictionDataGrid != null) PredictionDataGrid.ItemsSource = this.Predictions;

                EnsureDatabaseTablesExist();
                List<string> majorStocks = await FetchMajorUSStocks(token);

                token.ThrowIfCancellationRequested();
                
                if (majorStocks == null || majorStocks.Count == 0)
                {
                    //DatabaseMonolith.Log("Warning", "No stock symbols retrieved for automated analysis");
                    if (StatusText != null) StatusText.Text = "Failed to retrieve stock symbols for analysis";
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
                            // Save to database immediately for data persistence
                            SavePredictionToDatabase(analysisResult);
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
                            StatusText.Text = $"Analyzing stocks... ({processedCount}/{majorStocks.Count})";
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
                    
                    // Update status and timestamp in same dispatcher call
                    if (StatusText != null)
                        StatusText.Text = $"Automated analysis complete. Found {this.Predictions.Count} predictions";
                    if (LastUpdatedText != null)
                        LastUpdatedText.Text = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";
                }, System.Windows.Threading.DispatcherPriority.Background);

                //DatabaseMonolith.Log("Info", $"Automated analysis complete. Generated {this.Predictions.Count} predictions");

                if (isAutomatedMode && this.Predictions.Any())
                {
                     if (StatusText != null) StatusText.Text = "Executing trades based on predictions...";
                    // ExecuteTradesFromPredictionsAsync expects List<Quantra.Models.PredictionModel>
                    await ExecuteTradesFromPredictionsAsync(this.Predictions.ToList()); 
                     if (StatusText != null) StatusText.Text = "Automated trading complete.";
                }
            }
            catch (OperationCanceledException)
            {
                //DatabaseMonolith.Log("Info", "Automated analysis cancelled");
                 if (StatusText != null) StatusText.Text = "Analysis cancelled";
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error during automated analysis", ex.ToString());
                 if (StatusText != null) StatusText.Text = "Error during automated analysis";
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

        private void EnsureDatabaseTablesExist()
        {
            try
            {
                // Create stock symbols table
                DatabaseMonolith.ExecuteNonQuery(@"
                    CREATE TABLE IF NOT EXISTS StockSymbols (
                        Symbol TEXT PRIMARY KEY,
                        Name TEXT,
                        Sector TEXT,
                        Industry TEXT,
                        LastUpdated DATETIME
                    )");

                // Create predictions table
                DatabaseMonolith.ExecuteNonQuery(@"
                    CREATE TABLE IF NOT EXISTS StockPredictions (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Symbol TEXT NOT NULL,
                        PredictedAction TEXT NOT NULL,
                        Confidence REAL NOT NULL,
                        CurrentPrice REAL NOT NULL,
                        TargetPrice REAL NOT NULL,
                        PotentialReturn REAL NOT NULL,
                        CreatedDate DATETIME NOT NULL,
                        TradingRule TEXT,
                        FOREIGN KEY(Symbol) REFERENCES StockSymbols(Symbol)
                    )");

                // Create indicators table
                DatabaseMonolith.ExecuteNonQuery(@"
                    CREATE TABLE IF NOT EXISTS PredictionIndicators (
                        PredictionId INTEGER NOT NULL,
                        IndicatorName TEXT NOT NULL,
                        IndicatorValue REAL NOT NULL,
                        PRIMARY KEY(PredictionId, IndicatorName),
                        FOREIGN KEY(PredictionId) REFERENCES StockPredictions(Id)
                    )");
                
                // Create StockDataCache table
                DatabaseMonolith.ExecuteNonQuery(@"
                    CREATE TABLE IF NOT EXISTS StockDataCache (
                        Symbol TEXT NOT NULL,
                        TimeRange TEXT NOT NULL,
                        Interval TEXT NOT NULL,
                        Data TEXT NOT NULL,
                        CacheTime DATETIME NOT NULL,
                        PRIMARY KEY (Symbol, TimeRange, Interval)
                    )");
                
                // Create OrderHistory table
                DatabaseMonolith.ExecuteNonQuery(@"
                    CREATE TABLE IF NOT EXISTS OrderHistory (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Symbol TEXT NOT NULL,
                        OrderType TEXT NOT NULL,
                        Quantity INTEGER NOT NULL,
                        Price REAL NOT NULL,
                        StopLoss REAL,
                        TakeProfit REAL,
                        IsPaperTrade INTEGER NOT NULL,
                        Status TEXT NOT NULL,
                        PredictionSource TEXT,
                        Timestamp DATETIME NOT NULL
                    )");

                //DatabaseMonolith.Log("Info", "Ensured all required database tables exist");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error ensuring database tables exist", ex.ToString());
            }
        }

        private void SavePredictionToDatabase(Models.PredictionModel prediction)
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
                    //DatabaseMonolith.Log("Error", $"Invalid prediction data for {prediction?.Symbol ?? "<null>"}. Skipping insert.");
                    return;
                }

                // First ensure the symbol exists in the symbols table
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    using (var transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            using (var command = new SQLiteCommand(connection))
                            {
                                // Insert or update symbol
                                command.CommandText = @"
                                    INSERT OR IGNORE INTO StockSymbols (Symbol, LastUpdated) 
                                    VALUES (@Symbol, @LastUpdated)";
                                command.Parameters.AddWithValue("@Symbol", prediction.Symbol);
                                command.Parameters.AddWithValue("@LastUpdated", DateTime.Now);
                                command.ExecuteNonQuery();

                                // Save prediction
                                command.CommandText = @"
                                    INSERT INTO StockPredictions 
                                    (Symbol, PredictedAction, Confidence, CurrentPrice, TargetPrice, PotentialReturn, CreatedDate) 
                                    VALUES (@Symbol, @PredictedAction, @Confidence, @CurrentPrice, @TargetPrice, @PotentialReturn, @CreatedDate)";
                                command.Parameters.Clear();
                                command.Parameters.AddWithValue("@Symbol", prediction.Symbol);
                                command.Parameters.AddWithValue("@PredictedAction", prediction.PredictedAction);
                                command.Parameters.AddWithValue("@Confidence", prediction.Confidence);
                                command.Parameters.AddWithValue("@CurrentPrice", prediction.CurrentPrice);
                                command.Parameters.AddWithValue("@TargetPrice", prediction.TargetPrice);
                                command.Parameters.AddWithValue("@PotentialReturn", prediction.PotentialReturn);
                                command.Parameters.AddWithValue("@CreatedDate", DateTime.Now);
                                command.ExecuteNonQuery();

                                // Get last inserted prediction ID
                                command.CommandText = "SELECT last_insert_rowid()";
                                long predictionId = (long)command.ExecuteScalar();

                                // Save indicators
                                foreach (var indicator in prediction.Indicators)
                                {
                                    command.CommandText = @"
                                        INSERT INTO PredictionIndicators (PredictionId, IndicatorName, IndicatorValue)
                                        VALUES (@PredictionId, @IndicatorName, @IndicatorValue)";
                                    command.Parameters.Clear();
                                    command.Parameters.AddWithValue("@PredictionId", predictionId);
                                    command.Parameters.AddWithValue("@IndicatorName", indicator.Key);
                                    command.Parameters.AddWithValue("@IndicatorValue", indicator.Value);
                                    command.ExecuteNonQuery();
                                }
                            }

                            transaction.Commit();
                        }
                        catch (Exception ex)
                        {
                            transaction.Rollback();
                            throw new Exception($"Failed to save prediction: {ex.Message}", ex);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to save prediction for {prediction?.Symbol ?? "<null>"}", ex.ToString());
            }
        }

        private async Task<List<string>> FetchMajorUSStocks(CancellationToken token)
        {
            try
            {
                // First, try to get symbols from the database
                List<string> symbols = GetSymbolsFromDatabase();

                if (symbols != null && symbols.Count > 0)
                {
                    //DatabaseMonolith.Log("Info", $"Fetched {symbols.Count} US stock symbols from database");
                    return symbols;
                }
                else
                {
                    //DatabaseMonolith.Log("Warning", "No symbols found in database, falling back to default list. API call for symbols has been removed.");
                    // API call removed: var symbols = await alphaVantageService.GetAllUsStockSymbolsAsync(token); 
                    // Fallback to a default list if database is empty and API call is removed
                    var defaultSymbols = new List<string> { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA" }; 
                    // Optionally, save these default symbols to the database if that's desired behavior
                    // SaveSymbolsToDatabase(defaultSymbols); // This line can be uncommented if defaults should be saved
                    //DatabaseMonolith.Log("Info", $"Using default hardcoded list of {defaultSymbols.Count} symbols.");
                    return defaultSymbols;
                }
            }
            catch (OperationCanceledException)
            {
                //DatabaseMonolith.Log("Info", "Fetching major US stocks was cancelled.");
                return new List<string>();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to fetch US stock symbols", ex.ToString());
                //DatabaseMonolith.Log("Warning", "Falling back to default list due to error.");
                var defaultSymbols = new List<string> { "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA" };
                // SaveSymbolsToDatabase(defaultSymbols); // This line can be uncommented if defaults should be saved
                return defaultSymbols;
            }
        }

        private List<string> GetSymbolsFromDatabase()
        {
            List<string> symbols = new List<string>();
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    using (var command = new SQLiteCommand("SELECT Symbol FROM StockSymbols", connection))
                    {
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                symbols.Add(reader["Symbol"].ToString());
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error retrieving symbols from database", ex.ToString());
            }
            return symbols;
        }

        private void SaveSymbolsToDatabase(List<string> symbols)
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    using (var transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            using (var command = new SQLiteCommand(connection))
                            {
                                foreach (var symbol in symbols)
                                {
                                    command.CommandText = @"
                                        INSERT OR IGNORE INTO StockSymbols (Symbol, LastUpdated) 
                                        VALUES (@Symbol, @LastUpdated)";
                                    command.Parameters.Clear();
                                    command.Parameters.AddWithValue("@Symbol", symbol);
                                    command.Parameters.AddWithValue("@LastUpdated", DateTime.Now);
                                    command.ExecuteNonQuery();
                                }
                            }
                            transaction.Commit();
                        }
                        catch (Exception)
                        {
                            transaction.Rollback();
                            throw;
                        }
                    }
                }
                //DatabaseMonolith.Log("Info", $"Saved {symbols.Count} symbols to database");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error saving symbols to database", ex.ToString());
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