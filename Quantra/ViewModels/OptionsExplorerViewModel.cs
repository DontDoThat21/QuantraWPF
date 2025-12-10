using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Input;
using Quantra.Commands;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Options Explorer view
    /// Handles options chain display, Greeks visualization, and multi-leg strategy analysis
    /// </summary>
    public class OptionsExplorerViewModel : INotifyPropertyChanged
    {
        private readonly OptionsDataService _optionsDataService;
        private readonly IVSurfaceService _ivSurfaceService;
        private readonly OptionsPricingService _pricingService;
        private readonly GreekCalculationEngine _greekCalculator;
        private readonly IAlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;

        #region Private Fields

        private string _symbol;
        private double _currentPrice;
        private bool _isLoading;
        private string _statusMessage;
        private DateTime? _selectedExpiration;
        private OptionData _selectedOption;
        private IVSurfaceData _ivSurface;
        private IVSkewMetrics _ivSkew;
        private GreekMetrics _portfolioGreeks;

        #endregion

        #region Public Properties

        /// <summary>
        /// Current symbol being analyzed
        /// </summary>
        public string Symbol
        {
            get => _symbol;
            set
            {
                if (_symbol != value)
                {
                    _symbol = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// Current underlying price
        /// </summary>
        public double CurrentPrice
        {
            get => _currentPrice;
            set
            {
                if (_currentPrice != value)
                {
                    _currentPrice = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// Loading indicator
        /// </summary>
        public bool IsLoading
        {
            get => _isLoading;
            set
            {
                if (_isLoading != value)
                {
                    _isLoading = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// Status message
        /// </summary>
        public string StatusMessage
        {
            get => _statusMessage;
            set
            {
                if (_statusMessage != value)
                {
                    _statusMessage = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// Selected expiration date
        /// </summary>
        public DateTime? SelectedExpiration
        {
            get => _selectedExpiration;
            set
            {
                if (_selectedExpiration != value)
                {
                    _selectedExpiration = value;
                    OnPropertyChanged();
                    LoadOptionsChainCommand.Execute(null);
                }
            }
        }

        /// <summary>
        /// Selected option for detailed view
        /// </summary>
        public OptionData SelectedOption
        {
            get => _selectedOption;
            set
            {
                if (_selectedOption != value)
                {
                    _selectedOption = value;
                    OnPropertyChanged();
                    _ = LoadOptionDetailsAsync();
                }
            }
        }

        /// <summary>
        /// IV surface data
        /// </summary>
        public IVSurfaceData IVSurface
        {
            get => _ivSurface;
            set
            {
                if (_ivSurface != value)
                {
                    _ivSurface = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// IV skew metrics
        /// </summary>
        public IVSkewMetrics IVSkew
        {
            get => _ivSkew;
            set
            {
                if (_ivSkew != value)
                {
                    _ivSkew = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// Portfolio-level Greeks
        /// </summary>
        public GreekMetrics PortfolioGreeks
        {
            get => _portfolioGreeks;
            set
            {
                if (_portfolioGreeks != value)
                {
                    _portfolioGreeks = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// Collection of available expiration dates
        /// </summary>
        public ObservableCollection<DateTime> ExpirationDates { get; } = new ObservableCollection<DateTime>();

        /// <summary>
        /// Collection of call options
        /// </summary>
        public ObservableCollection<OptionData> CallOptions { get; } = new ObservableCollection<OptionData>();

        /// <summary>
        /// Collection of put options
        /// </summary>
        public ObservableCollection<OptionData> PutOptions { get; } = new ObservableCollection<OptionData>();

        /// <summary>
        /// Collection of options in portfolio/watchlist
        /// </summary>
        public ObservableCollection<OptionData> PortfolioOptions { get; } = new ObservableCollection<OptionData>();

        #endregion

        #region Commands

        public ICommand LoadSymbolCommand { get; }
        public ICommand LoadOptionsChainCommand { get; }
        public ICommand RefreshDataCommand { get; }
        public ICommand AddToPortfolioCommand { get; }
        public ICommand RemoveFromPortfolioCommand { get; }
        public ICommand CalculateGreeksCommand { get; }
        public ICommand BuildIVSurfaceCommand { get; }

        #endregion

        #region Constructor

        public OptionsExplorerViewModel(
            OptionsDataService optionsDataService,
            IVSurfaceService ivSurfaceService,
            OptionsPricingService pricingService,
            GreekCalculationEngine greekCalculator,
            IAlphaVantageService alphaVantageService,
            LoggingService loggingService)
        {
            _optionsDataService = optionsDataService ?? throw new ArgumentNullException(nameof(optionsDataService));
            _ivSurfaceService = ivSurfaceService ?? throw new ArgumentNullException(nameof(ivSurfaceService));
            _pricingService = pricingService ?? throw new ArgumentNullException(nameof(pricingService));
            _greekCalculator = greekCalculator ?? throw new ArgumentNullException(nameof(greekCalculator));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));

            // Initialize commands
            LoadSymbolCommand = new RelayCommand(async _ => await LoadSymbolAsync(), _ => !IsLoading && !string.IsNullOrWhiteSpace(Symbol));
            LoadOptionsChainCommand = new RelayCommand(async _ => await LoadOptionsChainAsync(), _ => !IsLoading && !string.IsNullOrWhiteSpace(Symbol));
            RefreshDataCommand = new RelayCommand(async _ => await RefreshDataAsync(), _ => !IsLoading);
            AddToPortfolioCommand = new RelayCommand(param => AddToPortfolio(param as OptionData), param => param is OptionData);
            RemoveFromPortfolioCommand = new RelayCommand(param => RemoveFromPortfolio(param as OptionData), param => param is OptionData);
            CalculateGreeksCommand = new RelayCommand(async _ => await CalculatePortfolioGreeksAsync(), _ => PortfolioOptions.Count > 0);
            BuildIVSurfaceCommand = new RelayCommand(async _ => await BuildIVSurfaceAsync(), _ => !IsLoading && !string.IsNullOrWhiteSpace(Symbol));
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Loads symbol and retrieves current price and expiration dates
        /// </summary>
        public async Task LoadSymbolAsync()
        {
            if (string.IsNullOrWhiteSpace(Symbol))
                return;

            try
            {
                IsLoading = true;
                StatusMessage = $"Loading {Symbol}...";

                // Get current price
                var quote = await _alphaVantageService.GetQuoteDataAsync(Symbol);
                if (quote != null)
                {
                    CurrentPrice = quote.Price;
                }

                // Get expiration dates
                var expirations = await _optionsDataService.GetExpirationDatesAsync(Symbol);
                ExpirationDates.Clear();
                foreach (var exp in expirations)
                {
                    ExpirationDates.Add(exp);
                }

                // Select nearest expiration
                if (ExpirationDates.Count > 0)
                {
                    SelectedExpiration = ExpirationDates.First();
                }

                StatusMessage = $"Loaded {Symbol} - ${CurrentPrice:F2}";
                _loggingService.Log("Info", $"Loaded symbol {Symbol} with {ExpirationDates.Count} expirations");
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error loading {Symbol}";
                _loggingService.LogErrorWithContext(ex, $"Error loading symbol {Symbol}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Loads options chain for selected expiration
        /// </summary>
        public async Task LoadOptionsChainAsync()
        {
            if (string.IsNullOrWhiteSpace(Symbol) || !SelectedExpiration.HasValue)
                return;

            try
            {
                IsLoading = true;
                StatusMessage = "Loading options chain...";

                var chain = await _optionsDataService.GetOptionsChainAsync(Symbol, SelectedExpiration, true);

                CallOptions.Clear();
                PutOptions.Clear();

                foreach (var option in chain.OrderBy(o => o.StrikePrice))
                {
                    if (option.OptionType?.ToUpper() == "CALL")
                    {
                        CallOptions.Add(option);
                    }
                    else
                    {
                        PutOptions.Add(option);
                    }
                }

                StatusMessage = $"Loaded {CallOptions.Count} calls and {PutOptions.Count} puts";
                _loggingService.Log("Info", $"Loaded options chain for {Symbol} exp {SelectedExpiration:yyyy-MM-dd}");

                // Automatically analyze IV skew
                await AnalyzeIVSkewAsync(chain);
            }
            catch (Exception ex)
            {
                StatusMessage = "Error loading options chain";
                _loggingService.LogErrorWithContext(ex, $"Error loading options chain for {Symbol}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Refreshes all data
        /// </summary>
        public async Task RefreshDataAsync()
        {
            await LoadSymbolAsync();
            if (SelectedExpiration.HasValue)
            {
                await LoadOptionsChainAsync();
            }
        }

        /// <summary>
        /// Adds an option to the portfolio
        /// </summary>
        public void AddToPortfolio(OptionData option)
        {
            if (option == null || PortfolioOptions.Contains(option))
                return;

            PortfolioOptions.Add(option);
            StatusMessage = $"Added {option.OptionType} ${option.StrikePrice} to portfolio";
            
            // Recalculate portfolio Greeks
            _ = CalculatePortfolioGreeksAsync();
        }

        /// <summary>
        /// Removes an option from the portfolio
        /// </summary>
        public void RemoveFromPortfolio(OptionData option)
        {
            if (option == null)
                return;

            PortfolioOptions.Remove(option);
            StatusMessage = $"Removed {option.OptionType} ${option.StrikePrice} from portfolio";
            
            // Recalculate portfolio Greeks
            _ = CalculatePortfolioGreeksAsync();
        }

        /// <summary>
        /// Builds the IV surface visualization
        /// </summary>
        public async Task BuildIVSurfaceAsync()
        {
            try
            {
                IsLoading = true;
                StatusMessage = "Building IV surface...";

                // Get full options chain (all expirations)
                var fullChain = await _optionsDataService.GetOptionsChainAsync(Symbol, null, false);
                
                IVSurface = await _ivSurfaceService.BuildIVSurfaceAsync(Symbol, fullChain);

                StatusMessage = $"Built IV surface with {IVSurface?.DataPoints?.Count ?? 0} points";
            }
            catch (Exception ex)
            {
                StatusMessage = "Error building IV surface";
                _loggingService.LogErrorWithContext(ex, $"Error building IV surface for {Symbol}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Loads detailed information for a selected option
        /// </summary>
        private async Task LoadOptionDetailsAsync()
        {
            if (SelectedOption == null)
                return;

            try
            {
                // Perform pricing analysis
                var analysis = await _pricingService.AnalyzePricingAsync(
                    SelectedOption,
                    CurrentPrice,
                    0.05 // TODO: Get actual risk-free rate
                );

                if (analysis != null)
                {
                    StatusMessage = $"Theoretical: ${analysis.TheoreticalPrice:F2}, Market: ${analysis.MarketPrice:F2}";
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error loading option details");
            }
        }

        /// <summary>
        /// Analyzes IV skew for the options chain
        /// </summary>
        private async Task AnalyzeIVSkewAsync(List<OptionData> chain)
        {
            try
            {
                IVSkew = await _ivSurfaceService.AnalyzeIVSkewAsync(chain);
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error analyzing IV skew");
            }
        }

        /// <summary>
        /// Calculates portfolio-level Greeks
        /// </summary>
        private async Task CalculatePortfolioGreeksAsync()
        {
            if (PortfolioOptions.Count == 0)
            {
                PortfolioGreeks = null;
                return;
            }

            try
            {
                await Task.Run(() =>
                {
                    var positions = PortfolioOptions.Select(opt => new Position
                    {
                        UnderlyingPrice = CurrentPrice,
                        StrikePrice = opt.StrikePrice,
                        TimeToExpiration = opt.TimeToExpiration,
                        Volatility = opt.ImpliedVolatility,
                        RiskFreeRate = 0.05, // TODO: Get actual risk-free rate
                        IsCall = opt.OptionType?.ToUpper() == "CALL",
                        Quantity = 1 // TODO: Allow user to specify quantity
                    }).ToList();

                    var market = new MarketConditions
                    {
                        InterestRate = 0.05 // TODO: Get actual risk-free rate
                    };

                    PortfolioGreeks = _greekCalculator.CalculatePortfolioGreeks(positions, market);
                    
                    StatusMessage = $"Portfolio Delta: {PortfolioGreeks.Delta:F2}";
                });
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error calculating portfolio Greeks");
            }
        }

        #endregion

        #region INotifyPropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
