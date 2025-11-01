using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;
using Quantra.ViewModels;
using Quantra.Models;
using Quantra.Controls.Components;
using Quantra.Views.PredictionAnalysis.Components;
using PredictionChartModuleType = Quantra.Views.PredictionAnalysis.Components.PredictionChartModule;
using System.Windows.Markup;
using Quantra.DAL.Services.Interfaces;
using System.Threading.Tasks;
using System.Collections.ObjectModel;
using Quantra.Repositories;
using Quantra.DAL.Services;

namespace Quantra.Controls
{
    public partial class PredictionAnalysisControl : UserControl, INotifyPropertyChanged
    {
        private readonly PredictionAnalysisViewModel _viewModel;
        private readonly INotificationService _notificationService;
        private readonly IStockDataCacheService _stockDataCacheService;
        private readonly ITechnicalIndicatorService _indicatorService;
        private readonly IEmailService _emailService;
        private readonly IndicatorDisplayModule _indicatorModule;
        private readonly PredictionChartModuleType _chartModule;
        private string _pacId; // Unique identifier for this PAC instance
        private ComboBox _strategyProfileComboBox;
        private Dictionary<string, DateTime> _lastAnalysisTime = new(); // Track last analysis time per symbol
        
        // Indicators dictionary (not duplicated in other files)
        private Dictionary<string, double> indicators;
        // Confidence value (not duplicated in other files)
        private double confidence;

        public PredictionAnalysisControl(
            PredictionAnalysisViewModel viewModel,
            INotificationService notificationService,
            ITechnicalIndicatorService indicatorService,
            PredictionAnalysisRepository analysisRepository,
            StockDataCacheService stockDataCacheService,
            ITradingService tradingService,
            ISettingsService settingsService,
            IAlphaVantageService alphaVantageService,
            IEmailService emailService)
        {
            InitializeComponent();

            var repo = analysisRepository ?? new PredictionAnalysisRepository();
            var indicatorSvc = indicatorService ?? new TechnicalIndicatorService();
            var emailSvc = emailService ?? new EmailService();
            var audioSvc = new AudioService(DatabaseMonolith.GetUserSettings());
            var smsSvc = new SmsService();
            var settingsSvc = settingsService ?? new SettingsService();
            var notificationSvc = notificationService ?? new NotificationService(DatabaseMonolith.GetUserSettings(), audioSvc, settingsSvc);
            var stockDataCacheSvc = stockDataCacheService ?? new StockDataCacheService();
            var tradingSvc = tradingService ?? new TradingService(emailSvc, notificationSvc, smsSvc);
            var alphaSvc = alphaVantageService ?? new AlphaVantageService();

            _viewModel = viewModel ?? new PredictionAnalysisViewModel(indicatorSvc, repo, tradingSvc, settingsSvc, alphaSvc, emailSvc);
            _notificationService = notificationSvc;
            _stockDataCacheService = stockDataCacheSvc;
            _indicatorService = indicatorSvc;
            _emailService = emailSvc;
        

            _indicatorModule = new IndicatorDisplayModule(_settingsService, _indicatorService, _notificationService, _emailService);
            _chartModule = new PredictionChartModuleType(_indicatorService, _notificationService, stockDataCacheService);

            DataContext = _viewModel;

            // Ensure the PredictionDataGrid is bound to the Predictions collection
            if (PredictionDataGrid != null)
            {
                PredictionDataGrid.ItemsSource = Predictions;
            }

            // Attach modules to containers if they exist
            var indicatorContainer = this.FindName("indicatorContainer") as Panel;
            if (indicatorContainer != null)
                indicatorContainer.Children.Add(_indicatorModule);

            var chartContainer = this.FindName("chartContainer") as Panel;
            if (chartContainer != null)
                chartContainer.Children.Add(_chartModule);

            // Register for Loaded event - use OnPredictionAnalysisControlLoaded to avoid duplicate
            Loaded += OnPredictionAnalysisControlLoaded;
        }

        // Initialize strategy profile selection in this separate method
        private void InitializeStrategyProfileSelection()
        {
            // Generate a unique PAC ID
            _pacId = this.GetHashCode().ToString();

            // Find or create the ComboBox for strategy selection
            if (_strategyProfileComboBox == null)
            {
                // Create the list of strategies with proper explicit typing to avoid ambiguity
                var strategies = new List<object>();
                
                // Add only the strategies that are available in your project
                // First, try with TradingStrategyProfile type if it exists
                try
                {
                    // Try to create and add a SmaCrossoverStrategy
                    var smaCrossover = Activator.CreateInstance(Type.GetType("Quantra.Models.SmaCrossoverStrategy, Quantra"));
                    if (smaCrossover != null)
                        strategies.Add(smaCrossover);
                }
                catch
                {
                    // If that fails, try with SmaCrossoverStrategy directly
                    try
                    {
                        // Fallback to basic implementation if needed
                        strategies.Add(new object()); // Placeholder for SmaCrossoverStrategy
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Could not create strategy: {ex.Message}");
                    }
                }

                // Add other strategies if they exist in your project
                try {
                    // Add EMA/SMA cross strategy if it exists
                    var emaSmaStrategy = Activator.CreateInstance(Type.GetType("Quantra.Models.EmaSmaCrossStrategy, Quantra"));
                    if (emaSmaStrategy != null)
                        strategies.Add(emaSmaStrategy);
                } catch (Exception ex) {
                    // Silently ignore missing strategy classes
                    System.Diagnostics.Debug.WriteLine($"Some strategy classes could not be loaded: {ex.Message}");
                }

                _strategyProfileComboBox = new ComboBox
                {
                    Margin = new Thickness(4),
                    Width = 180,
                    ItemsSource = strategies,
                    DisplayMemberPath = "Name",
                    SelectedIndex = strategies.Count > 0 ? 0 : -1
                };
                _strategyProfileComboBox.SelectionChanged += StrategyProfileComboBox_SelectionChanged;
                
                // Add to UI - find an appropriate panel in the XAML
                var tradingStrategyCombo = this.FindName("TradingStrategy") as ComboBox;
                if (tradingStrategyCombo != null)
                {
                    // Get parent grid or panel
                    var parent = tradingStrategyCombo.Parent as FrameworkElement;
                    while (parent != null && !(parent is Panel))
                    {
                        parent = parent.Parent as FrameworkElement;
                    }
                    
                    if (parent is Panel panel)
                    {
                        panel.Children.Add(_strategyProfileComboBox);
                    }
                }
            }
            
            // Set the initial strategy in the ViewModel if possible
            if (_strategyProfileComboBox.SelectedItem != null)
            {
                var selectedItem = _strategyProfileComboBox.SelectedItem;
                try
                {
                    // Use dynamic to avoid type casting issues
                    dynamic selectedStrategy = selectedItem;
                    _viewModel.SelectedStrategyProfile = selectedStrategy;
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error setting strategy: {ex.Message}");
                }
            }
        }

        // Helper to get selected strategy for this PAC
        public object GetSelectedStrategyProfile()
        {
            if (_strategyProfileComboBox == null || _strategyProfileComboBox.SelectedItem == null)
                return null; // Default strategy
                
            return _strategyProfileComboBox.SelectedItem;
        }

        private async void StrategyProfileComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_strategyProfileComboBox.SelectedItem != null)
            {
                var strategy = _strategyProfileComboBox.SelectedItem;
                
                try
                {
                    // Set the selected strategy profile in the ViewModel
                    // Use dynamic to avoid type casting issues
                    dynamic dynamicStrategy = strategy;
                    _viewModel.SelectedStrategyProfile = dynamicStrategy;
                    
                    // Save this strategy as associated with this PAC instance
                    SaveStrategyProfileForPac(_pacId, dynamicStrategy.Name);
                    
                    // Re-analyze with new strategy
                    await _viewModel.AnalyzeAsync();
                    
                    ShowNotification($"Applied {dynamicStrategy.Name} strategy profile", PackIconKind.ChartLineVariant, Colors.Blue);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error applying strategy: {ex.Message}");
                    ShowNotification("Error applying strategy", PackIconKind.Alert, Colors.Red);
                }
            }
        }

        // Method to save a strategy profile for a PAC instance
        private void SaveStrategyProfileForPac(string pacId, string strategyName)
        {
            try
            {
                // Store in local dictionary if needed
                var pacMappings = new Dictionary<string, string>();
                pacMappings[pacId] = strategyName;
                
                // Persist to settings if needed
                // DatabaseMonolith.SaveUserPreference($"PAC_Strategy_{pacId}", strategyName);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error saving strategy profile: {ex.Message}");
            }
        }

        private void ShowNotification(string message, PackIconKind icon, Color iconColor)
        {
            var snackbar = this.FindName("snackbar") as MaterialDesignThemes.Wpf.Snackbar;
            snackbar?.MessageQueue?.Enqueue(message, null, null, null, false, true, TimeSpan.FromSeconds(3));
        }

        // Changed name to avoid duplication with the method in other partial classes
        private void OnPredictionAnalysisControlLoaded(object sender, RoutedEventArgs e)
        {
            InitializeStrategyProfileSelection();
            InitializeComponents();
            CheckInitializeSentimentVisualizationOnLoad(sender, e);
            
            // Initialize the control components
            OnControlLoaded(sender, e);
            
            // Initialize status text if available
            var statusText = this.FindName("StatusText") as TextBlock;
            if (statusText != null)
                statusText.Text = "Ready";
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private void TradingStrategy_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var comboBox = sender as ComboBox;
            if (comboBox?.SelectedItem is TradingStrategyProfile selectedStrategy)
            {
                try
                {
                    // Update the strategy in the ViewModel
                    _viewModel.SelectedStrategyProfile = selectedStrategy;
                    
                    // Trigger a refresh of the indicators
                    RefreshIndicators();

                    _notificationService.ShowInfo($"Trading strategy changed to: {selectedStrategy.Name}");
                }
                catch (Exception ex)
                {
                    _notificationService.ShowError($"Error changing trading strategy: {ex.Message}");
                }
            }
        }

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var button = sender as Button;
                if (button != null)
                {
                    button.IsEnabled = false;
                }

                await RefreshAnalysis();
                _notificationService.ShowInfo("Analysis refreshed successfully");
            }
            catch (Exception ex)
            {
                _notificationService.ShowError($"Error refreshing analysis: {ex.Message}");
            }
            finally
            {
                if (sender is Button button)
                {
                    button.IsEnabled = true;
                }
            }
        }

        private void TradingRule_Click(object sender, RoutedEventArgs e)
        {
            if (_indicatorModule?.SelectedTradingRule != null)
            {
                try
                {
                    // Show the trading rule details
                    var rule = _indicatorModule.SelectedTradingRule;
                    var confirmationMessage = $"Trading Rule Details:\nSymbol: {rule.Symbol}\n" +
                                           $"Order Type: {rule.OrderType}\n" +
                                           $"Entry Price: ${rule.EntryPrice:F2}\n" +
                                           $"Exit Price: ${rule.ExitPrice:F2}\n" +
                                           $"Stop Loss: ${rule.StopLoss:F2}\n" +
                                           $"Quantity: {rule.Quantity}";

                    MessageBox.Show(confirmationMessage, "Trading Rule Details", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    _notificationService.ShowError($"Error displaying trading rule: {ex.Message}");
                }
            }
            else
            {
                _notificationService.ShowWarning("No trading rule selected");
            }
        }

        private void CreateRule_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Create new trading rule based on current analysis
                var rule = new TradingRule
                {
                    Symbol = _viewModel.Symbol,
                    Name = $"Rule_{DateTime.Now:yyyyMMddHHmmss}",
                    OrderType = "MARKET",
                    MinConfidence = 0.7,
                    EntryPrice = _viewModel.CurrentPrice,
                    ExitPrice = _viewModel.CurrentPrice * 1.02, // 2% profit target
                    StopLoss = _viewModel.CurrentPrice * 0.98,  // 2% stop loss
                    Quantity = 1,
                    IsActive = true,
                    CreatedDate = DateTime.Now
                };

                // Save the rule
                SaveTradingRule(rule);

                // Refresh the rules list
                _indicatorModule?.LoadTradingRules();

                _notificationService.ShowInfo("New trading rule created successfully");
            }
            catch (Exception ex)
            {
                _notificationService.ShowError($"Error creating trading rule: {ex.Message}");
            }
        }

        private async void ExecuteTradeButton_Click(object sender, RoutedEventArgs e)
        {
            if (_indicatorModule?.SelectedTradingRule == null)
            {
                _notificationService.ShowWarning("No trading rule selected");
                return;
            }

            try
            {
                var rule = _indicatorModule.SelectedTradingRule;
                var confirmResult = MessageBox.Show(
                    $"Execute {rule.OrderType} order for {rule.Quantity} shares of {rule.Symbol}?",
                    "Confirm Trade",
                    MessageBoxButton.YesNo,
                    MessageBoxImage.Question
                );

                if (confirmResult == MessageBoxResult.Yes)
                {
                    await ExecuteTrade(rule);
                }
            }
            catch (Exception ex)
            {
                _notificationService.ShowError($"Error executing trade: {ex.Message}");
            }
        }

        private void SaveTradingRule(TradingRule rule)
        {
            using (var connection = ConnectionHelper.GetConnection())
            {
                connection.Open();
                string query = @"
                    INSERT INTO TradingRules 
                    (Name, Symbol, OrderType, MinConfidence, EntryPrice, ExitPrice, StopLoss, Quantity, IsActive, CreatedDate) 
                    VALUES 
                    (@Name, @Symbol, @OrderType, @MinConfidence, @EntryPrice, @ExitPrice, @StopLoss, @Quantity, @IsActive, @CreatedDate)";

                using var cmd = connection.CreateCommand();
                cmd.CommandText = query;
                cmd.Parameters.AddWithValue("@Name", rule.Name);
                cmd.Parameters.AddWithValue("@Symbol", rule.Symbol);
                cmd.Parameters.AddWithValue("@OrderType", rule.OrderType);
                cmd.Parameters.AddWithValue("@MinConfidence", rule.MinConfidence);
                cmd.Parameters.AddWithValue("@EntryPrice", rule.EntryPrice);
                cmd.Parameters.AddWithValue("@ExitPrice", rule.ExitPrice);
                cmd.Parameters.AddWithValue("@StopLoss", rule.StopLoss);
                cmd.Parameters.AddWithValue("@Quantity", rule.Quantity);
                cmd.Parameters.AddWithValue("@IsActive", rule.IsActive);
                cmd.Parameters.AddWithValue("@CreatedDate", rule.CreatedDate);

                cmd.ExecuteNonQuery();
            }
        }

        private async Task ExecuteTrade(TradingRule rule)
        {
            var trade = new TradeRecord
            {
                Symbol = rule.Symbol,
                Action = rule.OrderType,
                Quantity = rule.Quantity,
                Price = rule.EntryPrice,
                TimeStamp = DateTime.Now,
                Notes = $"Executed from trading rule: {rule.Name}"
            };

            await _viewModel.TradingService.ExecuteTradeAsync(trade.Symbol, trade.Action, trade.Price, rule.ExitPrice);
            _notificationService.ShowInfo($"Trade executed successfully: {trade.Action} {trade.Quantity} {trade.Symbol}");
        }

        private async Task RefreshAnalysis()
        {
            try
            {
                await _viewModel.AnalyzeAsync();
                await RefreshIndicators();
            }
            catch (Exception ex)
            {
                _notificationService.ShowError($"Error refreshing analysis: {ex.Message}");
            }
        }

        private async Task RefreshIndicators()
        {
            if (_viewModel.SelectedStrategyProfile == null || string.IsNullOrEmpty(_viewModel.Symbol))
                return;

            try
            {
                var indicators = await _indicatorService.GetIndicatorsForPrediction(_viewModel.Symbol, "5min");
                _indicatorModule?.UpdateIndicatorValues(indicators);
            }
            catch (Exception ex)
            {
                _notificationService.ShowError($"Error refreshing indicators: {ex.Message}");
            }
        }

        /// <summary>
        /// Helper method to initialize all control components
        /// </summary>
        private void InitializeComponents()
        {
            try
            {
                // Initialize dictionary for indicators
                indicators = new Dictionary<string, double>();
                // Set default values as needed
                confidence = 0.7; // Default confidence value
                // Initialize other components as needed
                // ...
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to initialize components", ex.ToString());
            }
        }

        private void CheckInitializeSentimentVisualizationOnLoad(object sender, RoutedEventArgs e)
        {
            try
            {
                // Find the sentiment visualization container
                var sentimentContainer = this.FindName("sentimentVisualizationContainer") as Panel;
                if (sentimentContainer != null && sentimentContainer.Children.Count == 0)
                {
                    // Reinitialize sentiment visualization if needed
                    if (_sentimentVisualizationControl == null)
                    {
                        _sentimentVisualizationControl = new SentimentVisualizationControl();
                    }
                    // Add to container if not already added
                    if (!sentimentContainer.Children.Contains(_sentimentVisualizationControl))
                    {
                        sentimentContainer.Children.Add(_sentimentVisualizationControl);
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to initialize sentiment visualization on load", ex.ToString());
            }
        }
    }
}
