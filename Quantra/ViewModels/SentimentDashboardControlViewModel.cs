using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Input;
using LiveCharts;
using Quantra.Commands;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.Modules;
using Quantra.Repositories;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Sentiment Dashboard Control
    /// </summary>
    public class SentimentDashboardControlViewModel : ViewModelBase
    {
        private readonly Modules.SentimentPriceCorrelationAnalysis _sentimentCorrelationAnalysis;
        private readonly IAnalystRatingService _analystRatingService;
        private readonly IInsiderTradingService _insiderTradingService;

        private string _symbol;
        private int _currentLookbackDays;
        private SentimentPriceVisualData _currentSentimentData;
        private AnalystRatingAggregate _currentAnalystData;
        private List<InsiderTransaction> _currentInsiderData;
        private double _overallSentiment;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public SentimentDashboardControlViewModel(
            UserSettings userSettings,
            UserSettingsService userSettingsService,
            LoggingService loggingService,
            FinancialNewsSentimentService financialNewsSentimentService,
            ISocialMediaSentimentService socialMediaSentimentService,
            IAnalystRatingService analystRatingService,
            IInsiderTradingService insiderTradingService,
            SectorSentimentAnalysisService sectorSentimentService,
            PredictionAnalysisRepository predictionAnalysisRepository,
            SectorMomentumService sectorMomentumService)
        {
            if (userSettings == null) throw new ArgumentNullException(nameof(userSettings));
            if (userSettingsService == null) throw new ArgumentNullException(nameof(userSettingsService));
            if (loggingService == null) throw new ArgumentNullException(nameof(loggingService));

            _analystRatingService = analystRatingService ?? throw new ArgumentNullException(nameof(analystRatingService));
            _insiderTradingService = insiderTradingService ?? throw new ArgumentNullException(nameof(insiderTradingService));

            _sentimentCorrelationAnalysis = new Modules.SentimentPriceCorrelationAnalysis(
                userSettings, 
                userSettingsService, 
                loggingService,
                financialNewsSentimentService,
                socialMediaSentimentService,
                analystRatingService,
                insiderTradingService,
                sectorSentimentService,
                predictionAnalysisRepository,
                sectorMomentumService);

            // Initialize chart collections
            SentimentSeries = new SeriesCollection();
            RatingDistributionSeries = new SeriesCollection();
            PriceTargetSeries = new SeriesCollection();
            InsiderActivitySeries = new SeriesCollection();
            SentimentShiftSeries = new SeriesCollection();

            _currentLookbackDays = 30;
            _symbol = "--";

            InitializeCommands();
        }

        #region Properties

        /// <summary>
        /// Current stock symbol
        /// </summary>
        public string Symbol
        {
            get => _symbol;
            set => SetProperty(ref _symbol, value?.ToUpper());
        }

        /// <summary>
        /// Lookback period in days
        /// </summary>
        public int CurrentLookbackDays
        {
            get => _currentLookbackDays;
            set => SetProperty(ref _currentLookbackDays, value);
        }

        /// <summary>
        /// Overall sentiment score (0-100)
        /// </summary>
        public double OverallSentiment
        {
            get => _overallSentiment;
            set => SetProperty(ref _overallSentiment, value);
        }

        /// <summary>
        /// Sentiment trend chart series
        /// </summary>
        public SeriesCollection SentimentSeries { get; }

        /// <summary>
        /// Rating distribution chart series
        /// </summary>
        public SeriesCollection RatingDistributionSeries { get; }

        /// <summary>
        /// Price target chart series
        /// </summary>
        public SeriesCollection PriceTargetSeries { get; }

        /// <summary>
        /// Insider activity chart series
        /// </summary>
        public SeriesCollection InsiderActivitySeries { get; }

        /// <summary>
        /// Sentiment shift chart series
        /// </summary>
        public SeriesCollection SentimentShiftSeries { get; }

        #endregion

        #region Commands

        public ICommand UpdateDashboardCommand { get; private set; }
        public ICommand ChangeTimeframeCommand { get; private set; }

        #endregion

        #region Events

        /// <summary>
        /// Event fired when dashboard update completes
        /// </summary>
        public event EventHandler<string> DashboardUpdated;

        /// <summary>
        /// Event fired when an error occurs
        /// </summary>
        public event EventHandler<string> ErrorOccurred;

        #endregion

        #region Public Methods

        /// <summary>
        /// Update the entire dashboard with data for the specified symbol
        /// </summary>
        public async Task UpdateDashboardAsync(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return;

            Symbol = symbol;

            try
            {
                // Get sentiment data
                // TODO: Replace with actual implementation when available
                _currentSentimentData = null;

                // Get analyst data
                _currentAnalystData = await _analystRatingService.GetAggregatedRatingsAsync(symbol);

                // Get insider trading data
                _currentInsiderData = await _insiderTradingService.GetInsiderTransactionsAsync(symbol);

                // Update overall sentiment
                CalculateOverallSentiment();

                DashboardUpdated?.Invoke(this, symbol);
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(this, $"Error updating sentiment dashboard: {ex.Message}");
            }
        }

        /// <summary>
        /// Get current sentiment data
        /// </summary>
        public SentimentPriceVisualData GetCurrentSentimentData() => _currentSentimentData;

        /// <summary>
        /// Get current analyst data
        /// </summary>
        public AnalystRatingAggregate GetCurrentAnalystData() => _currentAnalystData;

        /// <summary>
        /// Get current insider data
        /// </summary>
        public List<InsiderTransaction> GetCurrentInsiderData() => _currentInsiderData;

        #endregion

        #region Private Methods

        private void InitializeCommands()
        {
            UpdateDashboardCommand = new RelayCommand(async param => await ExecuteUpdateDashboardAsync(param));
            ChangeTimeframeCommand = new RelayCommand(ExecuteChangeTimeframe);
        }

        private void CalculateOverallSentiment()
        {
            double sentiment = 50.0; // Default neutral

            // Calculate based on available data
            if (_currentAnalystData != null)
            {
                // Weight ratings: Buy=100, Hold=50, Sell=0
                int totalRatings = _currentAnalystData.BuyCount +
                                   _currentAnalystData.HoldCount +
                                   _currentAnalystData.SellCount;

                if (totalRatings > 0)
                {
                    double weightedScore =
                        (_currentAnalystData.BuyCount * 100) +
                        (_currentAnalystData.HoldCount * 50) +
                        (_currentAnalystData.SellCount * 0);

                    sentiment = weightedScore / totalRatings;
                }
            }

            OverallSentiment = sentiment;
        }

        #endregion

        #region Command Implementations

        private async Task ExecuteUpdateDashboardAsync(object parameter)
        {
            if (parameter is string symbol)
            {
                await UpdateDashboardAsync(symbol);
            }
        }

        private void ExecuteChangeTimeframe(object parameter)
        {
            if (parameter is int days)
            {
                CurrentLookbackDays = days;
                // Re-update dashboard with new timeframe with proper exception handling
                Task.Run(async () =>
                {
                    try
                    {
                        await UpdateDashboardAsync(Symbol);
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Error updating dashboard on timeframe change: {ex.Message}");
                    }
                });
            }
        }

        #endregion
    }
}
