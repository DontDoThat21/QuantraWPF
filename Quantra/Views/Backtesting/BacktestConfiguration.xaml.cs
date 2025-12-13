using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Quantra.Models;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using Quantra.ViewModels;

namespace Quantra.Views.Backtesting
{
    /// <summary>
    /// Interaction logic for BacktestConfiguration.xaml
    /// Main UI for configuring and running backtests
    /// </summary>
    public partial class BacktestConfiguration : UserControl
    {
        private readonly UserSettingsService _userSettingsService;
        private readonly LoggingService _loggingService;
        private readonly IAlphaVantageService _alphaVantageService;
        private HistoricalDataService _historicalDataService;
        private StockDataCacheService _stockDataCacheService;
        private BacktestResultService _backtestResultService;
        private List<TradingStrategyProfile> _availableStrategies;
        private TradingStrategyProfile _selectedStrategy;
        private BacktestResults _resultsControl;
        private BacktestingEngine.BacktestResult _currentBacktestResult;
        private List<HistoricalPrice> _currentHistoricalData;
        private double _currentInitialCapital;
        private ObservableCollection<CachedSymbolInfo> _cachedSymbols;

        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public BacktestConfiguration()
        {
            InitializeComponent();
            _cachedSymbols = new ObservableCollection<CachedSymbolInfo>();
        }

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public BacktestConfiguration(UserSettingsService userSettingsService,
            LoggingService loggingService,
            IAlphaVantageService alphaVantageService,
            StockSymbolCacheService stockSymbolCacheService)
        {
            InitializeComponent();

            _userSettingsService = userSettingsService ?? throw new ArgumentNullException(nameof(userSettingsService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));

            _historicalDataService = new HistoricalDataService(_userSettingsService, _loggingService, stockSymbolCacheService);
            _stockDataCacheService = new StockDataCacheService(_userSettingsService, _loggingService, stockSymbolCacheService);
            _backtestResultService = new BacktestResultService(_loggingService);
            _cachedSymbols = new ObservableCollection<CachedSymbolInfo>();

            // Initialize UI
            InitializeStrategies();
            InitializeDatePickers();
            
            // Load cached symbols asynchronously after UI is loaded
            this.Loaded += BacktestConfiguration_Loaded;
        }

        /// <summary>
        /// Handle control loaded event to initialize async operations
        /// </summary>
        private async void BacktestConfiguration_Loaded(object sender, RoutedEventArgs e)
        {
            // Only load once
            this.Loaded -= BacktestConfiguration_Loaded;
            await LoadCachedSymbolsAsync();
        }

        /// <summary>
        /// Load available strategies from StrategyProfileManager
        /// </summary>
        private void InitializeStrategies()
        {
            try
            {
                // Get all available strategies from the manager
                _availableStrategies = StrategyProfileManager.Instance.GetProfiles().ToList();

                // Add built-in strategies if none exist
                if (_availableStrategies.Count == 0)
                {
                    _availableStrategies = new List<TradingStrategyProfile>
                    {
                        new SmaCrossoverStrategy { Name = "SMA Crossover (20/50)" },
                        new MacdCrossoverStrategy { Name = "MACD Crossover" },
                        new RsiDivergenceStrategy { Name = "RSI Divergence" },
                        new BollingerBandsMeanReversionStrategy { Name = "Bollinger Bands Mean Reversion" }
                    };
                }

                StrategyComboBox.ItemsSource = _availableStrategies;
                if (_availableStrategies.Count > 0)
                {
                    StrategyComboBox.SelectedIndex = 0;
                }
            }
            catch (Exception ex)
            {
                ShowError($"Failed to load strategies: {ex.Message}");
            }
        }

        /// <summary>
        /// Initialize date pickers with default values
        /// </summary>
        private void InitializeDatePickers()
        {
            EndDatePicker.SelectedDate = DateTime.Today;
            StartDatePicker.SelectedDate = DateTime.Today.AddYears(-1);
            
            // Add event handlers for real-time validation
            EndDatePicker.SelectedDateChanged += EndDatePicker_SelectedDateChanged;
            StartDatePicker.SelectedDateChanged += StartDatePicker_SelectedDateChanged;
        }
        
        /// <summary>
        /// Handle end date selection changes to provide real-time validation feedback
        /// </summary>
        private void EndDatePicker_SelectedDateChanged(object sender, SelectionChangedEventArgs e)
        {
            if (EndDatePicker.SelectedDate == null)
                return;
                
            var endDate = EndDatePicker.SelectedDate.Value;
            
            // Check if end date is in the future
            if (endDate > DateTime.Today)
            {
                // Apply error styling
                EndDatePicker.BorderBrush = Brushes.Red;
                EndDatePicker.BorderThickness = new Thickness(2);
                ShowStatus($"⚠️ End date cannot be in the future! Please select {DateTime.Today:d} or earlier.");
                StatusText.Foreground = Brushes.Orange;
            }
            else if (StartDatePicker.SelectedDate != null && endDate <= StartDatePicker.SelectedDate.Value)
            {
                // End date must be after start date
                EndDatePicker.BorderBrush = Brushes.Orange;
                EndDatePicker.BorderThickness = new Thickness(2);
                ShowStatus("⚠️ End date must be after start date.");
                StatusText.Foreground = Brushes.Orange;
            }
            else
            {
                // Valid date - remove error styling
                EndDatePicker.BorderBrush = Brushes.Green;
                EndDatePicker.BorderThickness = new Thickness(1);
                
                // Calculate and show the date range
                if (StartDatePicker.SelectedDate != null)
                {
                    var days = (endDate - StartDatePicker.SelectedDate.Value).Days;
                    ShowStatus($"✓ Valid date range: {days} days from {StartDatePicker.SelectedDate.Value:d} to {endDate:d}");
                }
            }
        }
        
        /// <summary>
        /// Handle start date selection changes to provide real-time validation feedback
        /// </summary>
        private void StartDatePicker_SelectedDateChanged(object sender, SelectionChangedEventArgs e)
        {
            if (StartDatePicker.SelectedDate == null)
                return;
                
            var startDate = StartDatePicker.SelectedDate.Value;
            
            // Check if start date is too far in the past
            if (startDate < DateTime.Today.AddYears(-20))
            {
                // Apply warning styling
                StartDatePicker.BorderBrush = Brushes.Orange;
                StartDatePicker.BorderThickness = new Thickness(2);
                ShowStatus($"⚠️ Start date is very old. Historical data may not be available before {DateTime.Today.AddYears(-20):d}");
                StatusText.Foreground = Brushes.Orange;
            }
            else if (EndDatePicker.SelectedDate != null && startDate >= EndDatePicker.SelectedDate.Value)
            {
                // Start date must be before end date
                StartDatePicker.BorderBrush = Brushes.Orange;
                StartDatePicker.BorderThickness = new Thickness(2);
                ShowStatus("⚠️ Start date must be before end date.");
                StatusText.Foreground = Brushes.Orange;
            }
            else
            {
                // Valid date - remove error styling
                StartDatePicker.BorderBrush = Brushes.Green;
                StartDatePicker.BorderThickness = new Thickness(1);
                
                // Calculate and show the date range
                if (EndDatePicker.SelectedDate != null)
                {
                    var days = (EndDatePicker.SelectedDate.Value - startDate).Days;
                    ShowStatus($"✓ Valid date range: {days} days from {startDate:d} to {EndDatePicker.SelectedDate.Value:d}");
                }
            }
            
            // Also revalidate end date to update its styling
            if (EndDatePicker.SelectedDate != null)
            {
                EndDatePicker_SelectedDateChanged(EndDatePicker, null);
            }
        }

        /// <summary>
        /// Handle strategy selection change
        /// </summary>
        private void StrategyComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (StrategyComboBox.SelectedItem is TradingStrategyProfile strategy)
            {
                _selectedStrategy = strategy;
                UpdateStrategyInfo();

                // Update strategy parameters panel
                UpdateStrategyParametersPanel(strategy);
            }
        }

        /// <summary>
        /// Handle cost model selection change
        /// </summary>
        private void CostModelComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            UpdateStrategyInfo();
        }

        /// <summary>
        /// Update the strategy info text with strategy description and cost model details
        /// </summary>
        private void UpdateStrategyInfo()
        {
            // Guard against control not being initialized yet
            if (StrategyDescriptionText == null)
                return;

            if (_selectedStrategy == null)
            {
                StrategyDescriptionText.Text = "Select a strategy to see its description";
                return;
            }

            string strategyInfo = _selectedStrategy.Description ?? "No description available";
            
            // Add cost model information
            string costInfo = GetCostModelDescription();
            if (!string.IsNullOrEmpty(costInfo))
            {
                strategyInfo += $"\n\nCost Model: {costInfo}";
            }

            StrategyDescriptionText.Text = strategyInfo;
        }

        /// <summary>
        /// Get description of the selected cost model
        /// </summary>
        private string GetCostModelDescription()
        {
            if (CostModelComboBox == null || CostModelComboBox.SelectedIndex < 0)
                return string.Empty;

            return CostModelComboBox.SelectedIndex switch
            {
                0 => "Zero Cost - No commissions or slippage applied. Ideal for testing pure strategy performance.",
                1 => "Retail Brokerage - $1 commission per trade plus bid-ask spread (0.05%) and slippage (0.1%). Realistic for most retail investors.",
                2 => "Fixed Commission - $10 per trade. Suitable for traditional brokerage accounts.",
                3 => "Percentage Commission - 0.1% of trade value per trade. Common for some institutional accounts.",
                _ => string.Empty
            };
        }

        /// <summary>
        /// Dynamically update the strategy parameters panel based on selected strategy
        /// </summary>
        private void UpdateStrategyParametersPanel(TradingStrategyProfile strategy)
        {
            StrategyParametersPanel.Children.Clear();

            // Add specific parameter controls based on strategy type
            if (strategy is SmaCrossoverStrategy smaStrategy)
            {
                AddParameterControl("Fast Period:", smaStrategy.FastPeriod.ToString(), "FastPeriod");
                AddParameterControl("Slow Period:", smaStrategy.SlowPeriod.ToString(), "SlowPeriod");
            }
            else if (strategy is MacdCrossoverStrategy macdStrategy)
            {
                AddParameterControl("Fast Period:", macdStrategy.FastPeriod.ToString(), "FastPeriod");
                AddParameterControl("Slow Period:", macdStrategy.SlowPeriod.ToString(), "SlowPeriod");
                AddParameterControl("Signal Period:", macdStrategy.SignalPeriod.ToString(), "SignalPeriod");
            }
            else if (strategy is RsiDivergenceStrategy rsiStrategy)
            {
                AddParameterControl("RSI Period:", rsiStrategy.RsiPeriod.ToString(), "RsiPeriod");
                AddParameterControl("Overbought Level:", rsiStrategy.OverboughtLevel.ToString(), "OverboughtLevel");
                AddParameterControl("Oversold Level:", rsiStrategy.OversoldLevel.ToString(), "OversoldLevel");
            }

            // Add common parameters
            AddParameterControl("Min Confidence:", strategy.MinConfidence.ToString("F2"), "MinConfidence");
            AddParameterControl("Risk Level:", strategy.RiskLevel.ToString("F2"), "RiskLevel");
        }

        /// <summary>
        /// Add a parameter control to the parameters panel
        /// </summary>
        private void AddParameterControl(string label, string value, string parameterName)
        {
            var grid = new Grid { Margin = new Thickness(0, 5, 0, 0) };
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(150) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

            var labelBlock = new TextBlock
            {
                Text = label,
                VerticalAlignment = VerticalAlignment.Center,
                Foreground = Brushes.White
            };
            Grid.SetColumn(labelBlock, 0);

            var textBox = new TextBox
            {
                Text = value,
                Tag = parameterName,
                Margin = new Thickness(0, 0, 10, 0)
            };
            textBox.SetResourceReference(StyleProperty, "EnhancedTextBoxStyle");
            Grid.SetColumn(textBox, 1);

            grid.Children.Add(labelBlock);
            grid.Children.Add(textBox);
            StrategyParametersPanel.Children.Add(grid);
        }

        /// <summary>
        /// Handle Run Backtest button click
        /// </summary>
        private async void RunBacktestButton_Click(object sender, RoutedEventArgs e)
        {
            await RunBacktestAsync(useQuickTest: false);
        }

        /// <summary>
        /// Handle Quick Test button click (last 6 months)
        /// </summary>
        private async void QuickTestButton_Click(object sender, RoutedEventArgs e)
        {
            await RunBacktestAsync(useQuickTest: true);
        }

        /// <summary>
        /// Main backtest execution method
        /// </summary>
        private async Task RunBacktestAsync(bool useQuickTest)
        {
            try
            {
                // Validate inputs
                if (!ValidateInputs())
                    return;

                // Show progress
                ShowProgress("Initializing backtest...");

                // Parse inputs
                string symbol = SymbolTextBox.Text.Trim().ToUpper();
                double initialCapital = double.Parse(InitialCapitalTextBox.Text);
                int tradeSize = int.Parse(TradeSizeTextBox.Text);
                DateTime startDate = useQuickTest ? DateTime.Today.AddMonths(-6) : (StartDatePicker.SelectedDate ?? DateTime.Today.AddYears(-1));
                DateTime endDate = EndDatePicker.SelectedDate ?? DateTime.Today;

                // Store initial capital for saving
                _currentInitialCapital = initialCapital;

                // Update progress
                ShowProgress($"Fetching historical data for {symbol}...");

                // Get historical data
                var historicalData = await _historicalDataService.GetComprehensiveHistoricalData(symbol);

                if (historicalData == null || historicalData.Count == 0)
                {
                    ShowError($"No historical data found for {symbol}");
                    return;
                }

                // Filter by date range
                historicalData = historicalData
                    .Where(h => h.Date >= startDate && h.Date <= endDate)
                    .OrderBy(h => h.Date)
                    .ToList();

                if (historicalData.Count < 30)
                {
                    ShowError($"Insufficient data. Need at least 30 days, got {historicalData.Count}");
                    return;
                }

                ShowProgress($"Running backtest with {_selectedStrategy.Name}...");

                // Create cost model
                var costModel = GetSelectedCostModel();

                // Update strategy parameters from UI
                UpdateStrategyFromUI(_selectedStrategy);

                // Create backtesting engine
                var engine = new BacktestingEngine(_historicalDataService);

                // Run backtest
                var result = await engine.RunBacktestAsync(
                    symbol,
                    historicalData,
                    _selectedStrategy,
                    initialCapital,
                    tradeSize,
                    costModel
                );

                // Debug: Log backtest results
                System.Diagnostics.Debug.WriteLine($"Backtest completed for {symbol}:");
                System.Diagnostics.Debug.WriteLine($"  Total Return: {result.TotalReturn:P2}");
                System.Diagnostics.Debug.WriteLine($"  Max Drawdown: {result.MaxDrawdown:P2}");
                System.Diagnostics.Debug.WriteLine($"  Win Rate: {result.WinRate:P2}");
                System.Diagnostics.Debug.WriteLine($"  Total Trades: {result.TotalTrades}");
                System.Diagnostics.Debug.WriteLine($"  Sharpe Ratio: {result.SharpeRatio:F2}");

                ShowProgress("Rendering results...");

                // Display results
                DisplayResults(result, historicalData);

                ShowStatus($"Backtest completed! Total Return: {result.TotalReturn:P2}, Win Rate: {result.WinRate:P2}");
            }
            catch (Exception ex)
            {
                ShowError($"Backtest failed: {ex.Message}");
            }
            finally
            {
                HideProgress();
            }
        }

        /// <summary>
        /// Update strategy parameters from the UI parameter controls
        /// </summary>
        private void UpdateStrategyFromUI(TradingStrategyProfile strategy)
        {
            foreach (var child in StrategyParametersPanel.Children)
            {
                if (child is Grid grid && grid.Children.Count >= 2)
                {
                    if (grid.Children[1] is TextBox textBox && textBox.Tag is string paramName)
                    {
                        var property = strategy.GetType().GetProperty(paramName);
                        if (property != null && property.CanWrite)
                        {
                            try
                            {
                                if (property.PropertyType == typeof(int))
                                {
                                    property.SetValue(strategy, int.Parse(textBox.Text));
                                }
                                else if (property.PropertyType == typeof(double))
                                {
                                    property.SetValue(strategy, double.Parse(textBox.Text));
                                }
                            }
                            catch
                            {
                                // Ignore parse errors, keep default value
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Get the selected transaction cost model
        /// </summary>
        private TransactionCostModel GetSelectedCostModel()
        {
            return CostModelComboBox.SelectedIndex switch
            {
                0 => TransactionCostModel.CreateZeroCostModel(),
                1 => TransactionCostModel.CreateRetailBrokerageModel(),
                2 => TransactionCostModel.CreateFixedCommissionModel(10),
                3 => TransactionCostModel.CreatePercentageCommissionModel(0.001),
                _ => TransactionCostModel.CreateRetailBrokerageModel()
            };
        }

        /// <summary>
        /// Validate user inputs
        /// </summary>
        private bool ValidateInputs()
        {
            if (string.IsNullOrWhiteSpace(SymbolTextBox.Text))
            {
                ShowError("Please enter a stock symbol");
                return false;
            }

            if (_selectedStrategy == null)
            {
                ShowError("Please select a strategy");
                return false;
            }

            if (!double.TryParse(InitialCapitalTextBox.Text, out double capital) || capital <= 0)
            {
                ShowError("Initial capital must be a positive number");
                return false;
            }

            if (!int.TryParse(TradeSizeTextBox.Text, out int tradeSize) || tradeSize <= 0)
            {
                ShowError("Trade size must be a positive integer");
                return false;
            }

            if (StartDatePicker.SelectedDate == null || EndDatePicker.SelectedDate == null)
            {
                ShowError("Please select start and end dates");
                return false;
            }

            if (StartDatePicker.SelectedDate >= EndDatePicker.SelectedDate)
            {
                ShowError("Start date must be before end date");
                return false;
            }

            // Validate that end date is not in the future
            if (EndDatePicker.SelectedDate > DateTime.Today)
            {
                ShowError($"End date cannot be in the future. Please select a date on or before {DateTime.Today:d}.\n\n" +
                         $"Current end date: {EndDatePicker.SelectedDate.Value:d}\n" +
                         $"Today's date: {DateTime.Today:d}");
                return false;
            }

            // Validate that dates are not too far in the past (reasonable data availability)
            if (StartDatePicker.SelectedDate < DateTime.Today.AddYears(-20))
            {
                ShowError("Start date is too far in the past. Historical data may not be available before " +
                         DateTime.Today.AddYears(-20).ToShortDateString());
                return false;
            }

            return true;
        }

        /// <summary>
        /// Display backtest results
        /// </summary>
        private void DisplayResults(BacktestingEngine.BacktestResult result, List<HistoricalPrice> historical)
        {
            // Store current results for potential saving
            _currentBacktestResult = result;
            _currentHistoricalData = historical;

            // Create or reuse results control
            if (_resultsControl == null)
            {
                var customBenchmarkService = new CustomBenchmarkService(_historicalDataService);
                var viewModel = new BacktestResultsViewModel(
                    _historicalDataService,
                    customBenchmarkService,
                    _userSettingsService,
                    _alphaVantageService);

                _resultsControl = new BacktestResults(viewModel);
            }

            // Load results into the control
            _resultsControl.LoadResults(result, historical);

            // Display the control
            ResultsContainer.Content = _resultsControl;
            EmptyStatePanel.Visibility = Visibility.Collapsed;

            // Enable the save button
            SaveResultButton.IsEnabled = true;
        }

        /// <summary>
        /// Clear results
        /// </summary>
        private void ClearButton_Click(object sender, RoutedEventArgs e)
        {
            ResultsContainer.Content = null;
            EmptyStatePanel.Visibility = Visibility.Visible;
            _currentBacktestResult = null;
            _currentHistoricalData = null;
            SaveResultButton.IsEnabled = false;
            ShowStatus("Results cleared. Ready to run backtest");
        }

        /// <summary>
        /// Load cached symbols from StockDataCache for dropdown selection (async version)
        /// </summary>
        private async Task LoadCachedSymbolsAsync()
        {
            try
            {
                // Show loading indicator
                RefreshCacheButton.IsEnabled = false;
                RefreshCacheButton.Content = "Loading...";

                _cachedSymbols.Clear();

                if (_stockDataCacheService == null)
                {
                    return;
                }

                // Get all cached symbols asynchronously
                var symbols = await Task.Run(() => _stockDataCacheService.GetAllCachedSymbolsAsync());

                foreach (var symbol in symbols)
                {
                    _cachedSymbols.Add(new CachedSymbolInfo
                    {
                        Symbol = symbol,
                        CacheInfo = "cached"
                    });
                }

                // Set the ItemsSource for the ComboBox
                CachedSymbolsComboBox.ItemsSource = _cachedSymbols;

                _loggingService?.Log("Info", $"Loaded {_cachedSymbols.Count} cached symbols for backtest selection");
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Failed to load cached symbols: {ex.Message}", ex.ToString());
            }
            finally
            {
                // Restore button state
                RefreshCacheButton.IsEnabled = true;
                RefreshCacheButton.Content = "↻";
            }
        }

        /// <summary>
        /// Handle cached symbol selection from dropdown
        /// </summary>
        private void CachedSymbolsComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (CachedSymbolsComboBox.SelectedItem is CachedSymbolInfo selectedItem)
            {
                // Update the symbol textbox with the selected cached symbol
                SymbolTextBox.Text = selectedItem.Symbol;
                ShowStatus($"Selected cached symbol: {selectedItem.Symbol}");
            }
        }

        /// <summary>
        /// Refresh the cached symbols list
        /// </summary>
        private async void RefreshCacheButton_Click(object sender, RoutedEventArgs e)
        {
            await LoadCachedSymbolsAsync();
            ShowStatus($"Refreshed cached symbols. Found {_cachedSymbols.Count} symbols.");
        }

        /// <summary>
        /// Handle Run All Symbols button click - batch backtest all cached symbols
        /// </summary>
        private async void RunAllSymbolsButton_Click(object sender, RoutedEventArgs e)
        {
            await RunBatchBacktestsAsync();
        }

        /// <summary>
        /// Run backtests for all cached symbols and automatically save results
        /// </summary>
        private async Task RunBatchBacktestsAsync()
        {
            var originalCursor = this.Cursor;

            try
            {
                // Set waiting cursor
                this.Cursor = System.Windows.Input.Cursors.Wait;

                // Validate that we have a strategy selected
                if (_selectedStrategy == null)
                {
                    ShowError("Please select a strategy before running batch backtests");
                    return;
                }

                // Validate other required inputs
                if (!double.TryParse(InitialCapitalTextBox.Text, out double initialCapital) || initialCapital <= 0)
                {
                    ShowError("Initial capital must be a positive number");
                    return;
                }

                if (!int.TryParse(TradeSizeTextBox.Text, out int tradeSize) || tradeSize <= 0)
                {
                    ShowError("Trade size must be a positive integer");
                    return;
                }

                if (StartDatePicker.SelectedDate == null || EndDatePicker.SelectedDate == null)
                {
                    ShowError("Please select start and end dates");
                    return;
                }

                if (StartDatePicker.SelectedDate >= EndDatePicker.SelectedDate)
                {
                    ShowError("Start date must be before end date");
                    return;
                }

                // Get all cached symbols
                ShowProgress("Loading cached symbols...");
                var symbols = await Task.Run(() => _stockDataCacheService.GetAllCachedSymbolsAsync());

                if (symbols == null || !symbols.Any())
                {
                    ShowError("No cached symbols found. Please cache some symbols in Stock Explorer first.");
                    return;
                }

                var symbolList = symbols.ToList();
                int totalSymbols = symbolList.Count;
                int successCount = 0;
                int failCount = 0;

                // Configure progress bar for batch processing
                ProgressBar.Visibility = Visibility.Visible;
                ProgressBar.IsIndeterminate = false;
                ProgressBar.Minimum = 0;
                ProgressBar.Maximum = totalSymbols;
                ProgressBar.Value = 0;

                // Disable buttons during batch processing
                RunBacktestButton.IsEnabled = false;
                QuickTestButton.IsEnabled = false;
                RunAllSymbolsButton.IsEnabled = false;
                SaveResultButton.IsEnabled = false;

                DateTime startDate = StartDatePicker.SelectedDate.Value;
                DateTime endDate = EndDatePicker.SelectedDate.Value;

                // Get cost model
                var costModel = GetSelectedCostModel();

                // Update strategy parameters from UI
                UpdateStrategyFromUI(_selectedStrategy);

                // Process each symbol
                for (int i = 0; i < totalSymbols; i++)
                {
                    string symbol = symbolList[i];

                    try
                    {
                        // Update progress
                        ProgressBar.Value = i;
                        ShowStatus($"Processing {i + 1}/{totalSymbols}: {symbol}...");
                        StatusText.Foreground = Brushes.Yellow;

                        // Get historical data
                        var historicalData = await _historicalDataService.GetComprehensiveHistoricalData(symbol);

                        if (historicalData == null || historicalData.Count == 0)
                        {
                            _loggingService?.Log("Warning", $"No historical data found for {symbol}, skipping...");
                            failCount++;
                            continue;
                        }

                        // Filter by date range
                        historicalData = historicalData
                            .Where(h => h.Date >= startDate && h.Date <= endDate)
                            .OrderBy(h => h.Date)
                            .ToList();

                        if (historicalData.Count < 30)
                        {
                            _loggingService?.Log("Warning", $"Insufficient data for {symbol} ({historicalData.Count} days), skipping...");
                            failCount++;
                            continue;
                        }

                        // Create backtesting engine
                        var engine = new BacktestingEngine(_historicalDataService);

                        // Run backtest
                        var result = await engine.RunBacktestAsync(
                            symbol,
                            historicalData,
                            _selectedStrategy,
                            initialCapital,
                            tradeSize,
                            costModel
                        );

                        // Save result to database
                        var entity = _backtestResultService.ConvertFromEngineResult(
                            result,
                            _selectedStrategy.Name,
                            initialCapital,
                            $"Batch Run - {DateTime.Now:yyyy-MM-dd HH:mm}");

                        await _backtestResultService.SaveResultAsync(entity);

                        successCount++;
                        _loggingService?.Log("Info", $"Successfully backtested and saved {symbol}: Return={result.TotalReturn:P2}, WinRate={result.WinRate:P2}");
                    }
                    catch (Exception ex)
                    {
                        failCount++;
                        _loggingService?.Log("Error", $"Failed to backtest {symbol}: {ex.Message}", ex.ToString());
                    }
                }

                // Update final progress
                ProgressBar.Value = totalSymbols;
                ShowStatus($"Batch backtest completed! Successful: {successCount}, Failed: {failCount}, Total: {totalSymbols}");
                StatusText.Foreground = Brushes.LightGreen;

                MessageBox.Show(
                    $"Batch backtest completed!\n\n" +
                    $"Total Symbols: {totalSymbols}\n" +
                    $"Successful: {successCount}\n" +
                    $"Failed: {failCount}\n\n" +
                    $"Strategy: {_selectedStrategy.Name}\n" +
                    $"Date Range: {startDate:d} to {endDate:d}",
                    "Batch Backtest Complete",
                    MessageBoxButton.OK,
                    MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                ShowError($"Batch backtest failed: {ex.Message}");
                _loggingService?.Log("Error", $"Batch backtest error: {ex.Message}", ex.ToString());
            }
            finally
            {
                // Restore UI state
                this.Cursor = originalCursor;
                ProgressBar.Visibility = Visibility.Collapsed;
                ProgressBar.IsIndeterminate = false;
                RunBacktestButton.IsEnabled = true;
                QuickTestButton.IsEnabled = true;
                RunAllSymbolsButton.IsEnabled = true;
            }
        }

        /// <summary>
        /// Save the current backtest result to the database
        /// </summary>
        private async void SaveResultButton_Click(object sender, RoutedEventArgs e)
        {
            if (_currentBacktestResult == null)
            {
                ShowError("No backtest result to save. Please run a backtest first.");
                return;
            }

            if (_backtestResultService == null)
            {
                ShowError("Backtest result service not available.");
                return;
            }

            try
            {
                ShowProgress("Saving backtest result...");

                // Get optional run name from user
                string runName = null;
                var inputDialog = new InputDialog("Save Backtest Result", "Enter a name for this backtest run (optional):");
                if (inputDialog.ShowDialog() == true)
                {
                    runName = inputDialog.ResponseText;
                }

                // Convert engine result to entity
                var entity = _backtestResultService.ConvertFromEngineResult(
                    _currentBacktestResult,
                    _selectedStrategy?.Name ?? "Unknown Strategy",
                    _currentInitialCapital,
                    runName);

                // Save to database
                var savedEntity = await _backtestResultService.SaveResultAsync(entity);

                ShowStatus($"Backtest result saved successfully with ID {savedEntity.Id}");
                MessageBox.Show($"Backtest result saved successfully!\n\nSymbol: {savedEntity.Symbol}\nStrategy: {savedEntity.StrategyName}\nTotal Return: {savedEntity.TotalReturn:P2}",
                    "Result Saved", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                ShowError($"Failed to save backtest result: {ex.Message}");
            }
            finally
            {
                HideProgress();
            }
        }

        #region UI Helpers

        private void ShowProgress(string message)
        {
            StatusText.Text = message;
            StatusText.Foreground = Brushes.Yellow;
            ProgressBar.Visibility = Visibility.Visible;
            ProgressBar.IsIndeterminate = true;
            RunBacktestButton.IsEnabled = false;
            QuickTestButton.IsEnabled = false;
        }

        private void HideProgress()
        {
            ProgressBar.Visibility = Visibility.Collapsed;
            ProgressBar.IsIndeterminate = false;
            RunBacktestButton.IsEnabled = true;
            QuickTestButton.IsEnabled = true;
        }

        private void ShowStatus(string message)
        {
            StatusText.Text = message;
            StatusText.Foreground = Brushes.LightGreen;
        }

        private void ShowError(string message)
        {
            StatusText.Text = $"Error: {message}";
            StatusText.Foreground = Brushes.Red;
            MessageBox.Show(message, "Backtest Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        #endregion
    }

    /// <summary>
    /// Simple input dialog for getting user text input
    /// </summary>
    public class InputDialog : Window
    {
        private TextBox _responseTextBox;

        public string ResponseText => _responseTextBox.Text;

        public InputDialog(string title, string prompt)
        {
            Title = title;
            Width = 400;
            Height = 150;
            WindowStartupLocation = WindowStartupLocation.CenterOwner;
            Background = new SolidColorBrush(Color.FromRgb(35, 35, 58));

            var stackPanel = new StackPanel { Margin = new Thickness(15) };

            var promptLabel = new TextBlock
            {
                Text = prompt,
                Foreground = Brushes.White,
                Margin = new Thickness(0, 0, 0, 10)
            };
            stackPanel.Children.Add(promptLabel);

            _responseTextBox = new TextBox
            {
                Height = 25,
                Margin = new Thickness(0, 0, 0, 15)
            };
            stackPanel.Children.Add(_responseTextBox);

            var buttonPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right
            };

            var okButton = new Button
            {
                Content = "OK",
                Width = 75,
                Height = 30,
                Margin = new Thickness(0, 0, 10, 0),
                IsDefault = true
            };
            okButton.Click += (s, e) => { DialogResult = true; };
            buttonPanel.Children.Add(okButton);

            var cancelButton = new Button
            {
                Content = "Cancel",
                Width = 75,
                Height = 30,
                IsCancel = true
            };
            cancelButton.Click += (s, e) => { DialogResult = false; };
            buttonPanel.Children.Add(cancelButton);

            stackPanel.Children.Add(buttonPanel);
            Content = stackPanel;
        }
    }
}
