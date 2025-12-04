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
using Quantra.DAL.Data;
using Microsoft.EntityFrameworkCore;

namespace Quantra.Controls
{
    public partial class PredictionAnalysis : UserControl, INotifyPropertyChanged
    {
        private readonly PredictionAnalysisViewModel _viewModel;
        private readonly NotificationService _notificationService;
        private readonly StockDataCacheService _stockDataCacheService;
        private readonly TechnicalIndicatorService _indicatorService;
        private readonly EmailService _emailService;
        private readonly TradingService _tradingService;
        private readonly SettingsService _settingsService;
        private readonly PredictionAnalysisRepository _analysisRepository;
        private readonly AlphaVantageService _alphaVantageService;
        private readonly HistoricalDataService _historicalDataService;
        private readonly IndicatorSettingsService _indicatorSettingsService;
        private readonly TradingRuleService _tradingRuleService;
        private readonly UserSettingsService _userSettingsService;
        private readonly LoggingService _loggingService;
        private readonly OrderHistoryService _orderHistoryService;
        private string _pacId; // Unique identifier for this PAC instance
        private Dictionary<string, DateTime> _lastAnalysisTime = new(); // Track last analysis time per symbol
        
        // NOTE: Sentiment analysis service fields are declared in PredictionAnalysis.Analysis.cs
        // Do not redeclare here to avoid ambiguity errors
        
        // Indicators dictionary (not duplicated in other files)
        private Dictionary<string, double> indicators;
        // Confidence value (not duplicated in other files)
        private double confidence;

        // Parameterless constructor for XAML designer support
        public PredictionAnalysis()
        {
            InitializeComponent();
            indicators = new Dictionary<string, double>();
            _lastAnalysisTime = new Dictionary<string, DateTime>();
        }

     public PredictionAnalysis(
        PredictionAnalysisViewModel viewModel,
        NotificationService notificationService,
        TechnicalIndicatorService indicatorService,
        PredictionAnalysisRepository analysisRepository,
        StockDataCacheService stockDataCacheService,
        TradingService tradingService,
        HistoricalDataService historicalDataService,
        SettingsService settingsService,
        AlphaVantageService alphaVantageService,
        EmailService emailService,
        IndicatorSettingsService indicatorSettingsService,
        TradingRuleService tradingRuleService,
        UserSettingsService userSettingsService,
        LoggingService loggingService,
        OrderHistoryService orderHistoryService,
        TwitterSentimentService twitterSentimentService,
        FinancialNewsSentimentService financialNewsSentimentService,
        IEarningsTranscriptService earningsTranscriptService,
        IAnalystRatingService analystRatingService,
        IInsiderTradingService insiderTradingService)
     {
  InitializeComponent();

    _viewModel = viewModel;
    _notificationService = notificationService;
    _stockDataCacheService = stockDataCacheService;
    _indicatorService = indicatorService;
    _emailService = emailService;
    _tradingService = tradingService;
    _analysisRepository = analysisRepository;
    _settingsService = settingsService;
    _historicalDataService = historicalDataService;
    _alphaVantageService = alphaVantageService;
    _indicatorSettingsService = indicatorSettingsService ?? throw new ArgumentNullException(nameof(indicatorSettingsService));
    _tradingRuleService = tradingRuleService ?? throw new ArgumentNullException(nameof(tradingRuleService));
    _userSettingsService = userSettingsService ?? throw new ArgumentNullException(nameof(userSettingsService));
    _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
    _orderHistoryService = orderHistoryService ?? throw new ArgumentNullException(nameof(orderHistoryService));

    // Initialize sentiment analysis services via DI (MVVM pattern)
    _twitterSentimentService = twitterSentimentService ?? throw new ArgumentNullException(nameof(twitterSentimentService));
    _financialNewsSentimentService = financialNewsSentimentService ?? throw new ArgumentNullException(nameof(financialNewsSentimentService));
    _earningsTranscriptService = earningsTranscriptService ?? throw new ArgumentNullException(nameof(earningsTranscriptService));
    _analystRatingService = analystRatingService ?? throw new ArgumentNullException(nameof(analystRatingService));
    _insiderTradingService = insiderTradingService ?? throw new ArgumentNullException(nameof(insiderTradingService));
    _userSettings = _userSettingsService.GetUserSettings();

    // Initialize model training service
    InitializeTrainingService();

         DataContext = _viewModel;

    // Predictions collection is now available for any other UI binding

        // Register for Loaded event - use OnPredictionAnalysisControlLoaded to avoid duplicate
    Loaded += OnPredictionAnalysisControlLoaded;
  }


        private void ShowNotification(string message, PackIconKind icon, Color iconColor)
        {
            var snackbar = this.FindName("snackbar") as MaterialDesignThemes.Wpf.Snackbar;
            snackbar?.MessageQueue?.Enqueue(message, null, null, null, false, true, TimeSpan.FromSeconds(3));
        }

        // Changed name to avoid duplication with the method in other partial classes
        private void OnPredictionAnalysisControlLoaded(object sender, RoutedEventArgs e)
        {
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

                _notificationService.ShowInfo("New trading rule created successfully");
            }
            catch (Exception ex)
            {
                _notificationService.ShowError($"Error creating trading rule: {ex.Message}");
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
                // Indicators can be displayed in another way if needed
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
