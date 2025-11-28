using System;
using System.Collections.Generic;
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
        private List<TradingStrategyProfile> _availableStrategies;
        private TradingStrategyProfile _selectedStrategy;
        private BacktestResults _resultsControl;

        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public BacktestConfiguration()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public BacktestConfiguration(UserSettingsService userSettingsService,
            LoggingService loggingService,
            IAlphaVantageService alphaVantageService)
        {
            InitializeComponent();

            _userSettingsService = userSettingsService ?? throw new ArgumentNullException(nameof(userSettingsService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));

            _historicalDataService = new HistoricalDataService(_userSettingsService, _loggingService);

            // Initialize UI
            InitializeStrategies();
            InitializeDatePickers();
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

            return true;
        }

        /// <summary>
        /// Display backtest results
        /// </summary>
        private void DisplayResults(BacktestingEngine.BacktestResult result, List<HistoricalPrice> historical)
        {
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
        }

        /// <summary>
        /// Clear results
        /// </summary>
        private void ClearButton_Click(object sender, RoutedEventArgs e)
        {
            ResultsContainer.Content = null;
            EmptyStatePanel.Visibility = Visibility.Visible;
            ShowStatus("Results cleared. Ready to run backtest");
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
}
