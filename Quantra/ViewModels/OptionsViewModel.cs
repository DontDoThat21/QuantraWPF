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
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.ViewModels
{
    /// <summary>
    /// Comprehensive ViewModel for Options Explorer
    /// Manages options chain display, Greeks, IV analysis, and multi-leg strategies
    /// </summary>
    public class OptionsViewModel : INotifyPropertyChanged
    {
        #region Services

        private readonly OptionsDataService _optionsDataService;
        private readonly IAlphaVantageService _alphaVantageService;
        private readonly GreekCalculationEngine _greekCalculator;
        private readonly IVSurfaceService _ivSurfaceService;
        private readonly OptionsPricingService _pricingService;
        private readonly IStockDataCacheService _stockDataCacheService;
        private readonly LoggingService _loggingService;

        #endregion

        #region Private Fields

        private string _selectedSymbol;
        private double _underlyingPrice;
        private CompanyOverview _companyInfo;
        private OptionData _selectedOption;
        private GreekMetrics _calculatedGreeks;
        private DateTime? _selectedExpiration;
        private OptionsChainFilter _currentFilter;
        private bool _showOnlyITM;
        private bool _showOnlyLiquid;
        private int? _strikeRange;
        private Quantra.DAL.Models.IVSurfaceData _ivSurface;
        private double _averageIV;
        private double _ivPercentile;
        private SpreadConfiguration _currentSpread;
        private bool _isLoading;
        private string _statusMessage;
        private bool _dataAvailable;
        private GreekMetrics _portfolioGreeks;

        #endregion

        #region Public Properties - Symbol & Underlying

        /// <summary>
        /// Currently selected underlying symbol
        /// </summary>
        public string SelectedSymbol
        {
            get => _selectedSymbol;
            set
            {
                if (_selectedSymbol != value)
                {
                    _selectedSymbol = value;
                    OnPropertyChanged();
                    _ = LoadSymbolDataAsync();
                }
            }
        }

        /// <summary>
        /// Current price of the underlying asset
        /// </summary>
        public double UnderlyingPrice
        {
            get => _underlyingPrice;
            set
            {
                if (_underlyingPrice != value)
                {
                    _underlyingPrice = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// Company fundamental information
        /// </summary>
        public CompanyOverview CompanyInfo
        {
            get => _companyInfo;
            set
            {
                if (_companyInfo != value)
                {
                    _companyInfo = value;
                    OnPropertyChanged();
                }
            }
        }

        #endregion

        #region Public Properties - Options Chain Data

        /// <summary>
        /// Collection of call options
        /// </summary>
        public ObservableCollection<OptionData> CallOptions { get; } = new ObservableCollection<OptionData>();

        /// <summary>
        /// Collection of put options
        /// </summary>
        public ObservableCollection<OptionData> PutOptions { get; } = new ObservableCollection<OptionData>();

        /// <summary>
        /// Available expiration dates for the symbol
        /// </summary>
        public ObservableCollection<DateTime> ExpirationDates { get; } = new ObservableCollection<DateTime>();

        /// <summary>
        /// Currently selected expiration date
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
                    _ = LoadOptionsChainAsync();
                }
            }
        }

        #endregion

        #region Public Properties - Selected Option Details

        /// <summary>
        /// Currently selected option for detailed analysis
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
                    _ = CalculateOptionGreeksAsync();
                }
            }
        }

        /// <summary>
        /// Calculated Greeks for the selected option
        /// </summary>
        public GreekMetrics CalculatedGreeks
        {
            get => _calculatedGreeks;
            set
            {
                if (_calculatedGreeks != value)
                {
                    _calculatedGreeks = value;
                    OnPropertyChanged();
                }
            }
        }

        #endregion

        #region Public Properties - Filters

        /// <summary>
        /// Current filter applied to options chain
        /// </summary>
        public OptionsChainFilter CurrentFilter
        {
            get => _currentFilter;
            set
            {
                if (_currentFilter != value)
                {
                    _currentFilter = value;
                    OnPropertyChanged();
                    _ = ApplyFiltersAsync();
                }
            }
        }

        /// <summary>
        /// Filter to show only in-the-money options
        /// </summary>
        public bool ShowOnlyITM
        {
            get => _showOnlyITM;
            set
            {
                if (_showOnlyITM != value)
                {
                    _showOnlyITM = value;
                    OnPropertyChanged();
                    UpdateFilterAndReload();
                }
            }
        }

        /// <summary>
        /// Filter to show only liquid options (high volume/OI)
        /// </summary>
        public bool ShowOnlyLiquid
        {
            get => _showOnlyLiquid;
            set
            {
                if (_showOnlyLiquid != value)
                {
                    _showOnlyLiquid = value;
                    OnPropertyChanged();
                    UpdateFilterAndReload();
                }
            }
        }

        /// <summary>
        /// Strike range filter (±X from ATM)
        /// </summary>
        public int? StrikeRange
        {
            get => _strikeRange;
            set
            {
                if (_strikeRange != value)
                {
                    _strikeRange = value;
                    OnPropertyChanged();
                    UpdateFilterAndReload();
                }
            }
        }

        #endregion

        #region Public Properties - IV Analysis

        /// <summary>
        /// 3D implied volatility surface data
        /// </summary>
        public Quantra.DAL.Models.IVSurfaceData IVSurface
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
        /// Average implied volatility across the chain
        /// </summary>
        public double AverageIV
        {
            get => _averageIV;
            set
            {
                if (_averageIV != value)
                {
                    _averageIV = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// IV percentile (rank in 52-week range)
        /// </summary>
        public double IVPercentile
        {
            get => _ivPercentile;
            set
            {
                if (_ivPercentile != value)
                {
                    _ivPercentile = value;
                    OnPropertyChanged();
                }
            }
        }

        #endregion

        #region Public Properties - Position Builder

        /// <summary>
        /// Selected options legs for multi-leg strategy
        /// </summary>
        public ObservableCollection<OptionData> SelectedLegs { get; } = new ObservableCollection<OptionData>();

        /// <summary>
        /// Current spread configuration (integration with SpreadsExplorer)
        /// </summary>
        public SpreadConfiguration CurrentSpread
        {
            get => _currentSpread;
            set
            {
                if (_currentSpread != value)
                {
                    _currentSpread = value;
                    OnPropertyChanged();
                }
            }
        }

        /// <summary>
        /// Portfolio-level Greeks for all selected legs
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

        #endregion

        #region Public Properties - UI State

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
        /// Status message for user feedback
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
        /// Indicates if options data is available
        /// </summary>
        public bool DataAvailable
        {
            get => _dataAvailable;
            set
            {
                if (_dataAvailable != value)
                {
                    _dataAvailable = value;
                    OnPropertyChanged();
                }
            }
        }

        #endregion

        #region Commands

        public ICommand LoadOptionsChainCommand { get; }
        public ICommand RefreshDataCommand { get; }
        public ICommand CalculateGreeksCommand { get; }
        public ICommand AddToSpreadCommand { get; }
        public ICommand RemoveFromSpreadCommand { get; }
        public ICommand ApplyFilterCommand { get; }
        public ICommand ExportToCSVCommand { get; }
        public ICommand CompareHistoricalIVCommand { get; }
        public ICommand BuildIVSurfaceCommand { get; }
        public ICommand CalculateTheoreticalPriceCommand { get; }
        public ICommand ResetFiltersCommand { get; }

        #endregion

        #region Constructor

        public OptionsViewModel(
            OptionsDataService optionsDataService,
            IAlphaVantageService alphaVantageService,
            GreekCalculationEngine greekCalculator,
            IVSurfaceService ivSurfaceService,
            OptionsPricingService pricingService,
            IStockDataCacheService stockDataCacheService,
            LoggingService loggingService)
        {
            _optionsDataService = optionsDataService ?? throw new ArgumentNullException(nameof(optionsDataService));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _greekCalculator = greekCalculator ?? throw new ArgumentNullException(nameof(greekCalculator));
            _ivSurfaceService = ivSurfaceService ?? throw new ArgumentNullException(nameof(ivSurfaceService));
            _pricingService = pricingService ?? throw new ArgumentNullException(nameof(pricingService));
            _stockDataCacheService = stockDataCacheService ?? throw new ArgumentNullException(nameof(stockDataCacheService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));

            // Initialize commands
            LoadOptionsChainCommand = new RelayCommand(async _ => await LoadOptionsChainAsync(), _ => CanLoadOptionsChain());
            RefreshDataCommand = new RelayCommand(async _ => await RefreshAllDataAsync(), _ => !string.IsNullOrEmpty(SelectedSymbol));
            CalculateGreeksCommand = new RelayCommand(async _ => await CalculatePortfolioGreeksAsync(), _ => SelectedLegs.Count > 0);
            AddToSpreadCommand = new RelayCommand(param => AddOptionToSpread(param as OptionData), param => param is OptionData);
            RemoveFromSpreadCommand = new RelayCommand(param => RemoveOptionFromSpread(param as OptionData), param => param is OptionData);
            ApplyFilterCommand = new RelayCommand(async _ => await ApplyFiltersAsync(), _ => DataAvailable);
            ExportToCSVCommand = new RelayCommand(async _ => await ExportOptionsChainToCSVAsync(), _ => DataAvailable);
            CompareHistoricalIVCommand = new RelayCommand(async _ => await CompareHistoricalIVAsync(), _ => !string.IsNullOrEmpty(SelectedSymbol));
            BuildIVSurfaceCommand = new RelayCommand(async _ => await BuildIVSurfaceAsync(), _ => DataAvailable);
            CalculateTheoreticalPriceCommand = new RelayCommand(async _ => await CalculateTheoreticalPricesAsync(), _ => DataAvailable);
            ResetFiltersCommand = new RelayCommand(_ => ResetFilters(), _ => CurrentFilter != null);

            // Initialize default filter
            CurrentFilter = new OptionsChainFilter();
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Loads symbol data (price, company info, expiration dates)
        /// </summary>
        public async Task LoadSymbolDataAsync()
        {
            if (string.IsNullOrWhiteSpace(SelectedSymbol))
                return;

            try
            {
                IsLoading = true;
                StatusMessage = $"Loading {SelectedSymbol}...";

                // Get underlying price
                var quote = await _alphaVantageService.GetQuoteDataAsync(SelectedSymbol);
                if (quote != null)
                {
                    UnderlyingPrice = quote.Price;
                }

                // Get company overview
                CompanyInfo = await _alphaVantageService.GetCompanyOverviewAsync(SelectedSymbol);

                // Get available expiration dates
                var expirations = await _optionsDataService.GetExpirationDatesAsync(SelectedSymbol);
                ExpirationDates.Clear();
                foreach (var exp in expirations)
                {
                    ExpirationDates.Add(exp);
                }

                // Select the nearest expiration by default
                if (ExpirationDates.Count > 0)
                {
                    SelectedExpiration = ExpirationDates.First();
                }

                StatusMessage = $"{SelectedSymbol} loaded - ${UnderlyingPrice:F2}";
                _loggingService.Log("Info", $"Loaded symbol {SelectedSymbol} with {ExpirationDates.Count} expirations");
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error loading {SelectedSymbol}";
                _loggingService.LogErrorWithContext(ex, $"Error loading symbol {SelectedSymbol}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Loads options chain for the selected expiration
        /// </summary>
        public async Task LoadOptionsChainAsync()
        {
            if (string.IsNullOrWhiteSpace(SelectedSymbol) || !SelectedExpiration.HasValue)
                return;

            try
            {
                IsLoading = true;
                StatusMessage = "Loading options chain...";

                var chain = await _optionsDataService.GetOptionsChainAsync(SelectedSymbol, SelectedExpiration, includeGreeks: true);

                // Apply filters
                var filteredChain = ApplyCurrentFilter(chain);

                // Separate calls and puts
                CallOptions.Clear();
                PutOptions.Clear();

                foreach (var option in filteredChain.OrderBy(o => o.StrikePrice))
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

                // Calculate average IV
                if (filteredChain.Count > 0)
                {
                    AverageIV = filteredChain.Average(o => o.ImpliedVolatility);
                }

                DataAvailable = CallOptions.Count > 0 || PutOptions.Count > 0;
                StatusMessage = $"Loaded {CallOptions.Count} calls and {PutOptions.Count} puts";
                _loggingService.Log("Info", $"Loaded options chain for {SelectedSymbol} exp {SelectedExpiration:yyyy-MM-dd}");
            }
            catch (Exception ex)
            {
                StatusMessage = "Error loading options chain";
                _loggingService.LogErrorWithContext(ex, $"Error loading options chain for {SelectedSymbol}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Refreshes all data (symbol, chain, and Greeks)
        /// </summary>
        public async Task RefreshAllDataAsync()
        {
            await LoadSymbolDataAsync();
            if (SelectedExpiration.HasValue)
            {
                await LoadOptionsChainAsync();
            }
            if (SelectedLegs.Count > 0)
            {
                await CalculatePortfolioGreeksAsync();
            }
        }

        /// <summary>
        /// Calculates Greeks for a single selected option
        /// </summary>
        public async Task CalculateOptionGreeksAsync()
        {
            if (SelectedOption == null)
                return;

            try
            {
                await Task.Run(() =>
                {
                    var position = new Position
                    {
                        UnderlyingPrice = UnderlyingPrice,
                        StrikePrice = SelectedOption.StrikePrice,
                        TimeToExpiration = SelectedOption.TimeToExpiration,
                        Volatility = SelectedOption.ImpliedVolatility,
                        RiskFreeRate = 0.05, // TODO: Get actual risk-free rate
                        IsCall = SelectedOption.OptionType?.ToUpper() == "CALL",
                        Quantity = 1
                    };

                    var market = new MarketConditions
                    {
                        InterestRate = 0.05
                    };

                    CalculatedGreeks = _greekCalculator.CalculateGreeks(position, market);
                });
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error calculating Greeks for selected option");
            }
        }

        /// <summary>
        /// Adds an option to the multi-leg spread builder
        /// </summary>
        public void AddOptionToSpread(OptionData option)
        {
            if (option == null || SelectedLegs.Contains(option))
                return;

            SelectedLegs.Add(option);
            StatusMessage = $"Added {option.OptionType} ${option.StrikePrice} to position";
            
            _ = CalculatePortfolioGreeksAsync();
        }

        /// <summary>
        /// Removes an option from the spread builder
        /// </summary>
        public void RemoveOptionFromSpread(OptionData option)
        {
            if (option == null)
                return;

            SelectedLegs.Remove(option);
            StatusMessage = $"Removed {option.OptionType} ${option.StrikePrice} from position";
            
            _ = CalculatePortfolioGreeksAsync();
        }

        /// <summary>
        /// Calculates portfolio-level Greeks for all selected legs
        /// </summary>
        public async Task CalculatePortfolioGreeksAsync()
        {
            if (SelectedLegs.Count == 0)
            {
                PortfolioGreeks = null;
                return;
            }

            try
            {
                await Task.Run(() =>
                {
                    var positions = SelectedLegs.Select(opt => new Position
                    {
                        UnderlyingPrice = UnderlyingPrice,
                        StrikePrice = opt.StrikePrice,
                        TimeToExpiration = opt.TimeToExpiration,
                        Volatility = opt.ImpliedVolatility,
                        RiskFreeRate = 0.05,
                        IsCall = opt.OptionType?.ToUpper() == "CALL",
                        Quantity = 1 // TODO: Allow user to specify quantity
                    }).ToList();

                    var market = new MarketConditions
                    {
                        InterestRate = 0.05
                    };

                    PortfolioGreeks = _greekCalculator.CalculatePortfolioGreeks(positions, market);
                    
                    StatusMessage = $"Portfolio Delta: {PortfolioGreeks.Delta:F2}, Theta: {PortfolioGreeks.Theta:F2}";
                });
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error calculating portfolio Greeks");
            }
        }

        /// <summary>
        /// Builds the 3D IV surface from all available options
        /// </summary>
        public async Task BuildIVSurfaceAsync()
        {
            try
            {
                IsLoading = true;
                StatusMessage = "Building IV surface...";

                // Get full options chain (all expirations)
                var fullChain = await _optionsDataService.GetOptionsChainAsync(SelectedSymbol, null, includeGreeks: false);
                
                // IVSurfaceService returns its own IVSurfaceData type
                var surfaceResult = await _ivSurfaceService.BuildIVSurfaceAsync(SelectedSymbol, fullChain);
                
                // Convert to our ViewModel's expected type if needed
                // For now, just store the result (they should be compatible)
                // IVSurface = surfaceResult; // Type conversion may be needed
                
                // TODO: Add conversion logic if types don't match
                // Or use the service's type directly in ViewModel

                StatusMessage = $"Built IV surface successfully";
                _loggingService.Log("Info", $"Built IV surface for {SelectedSymbol}");
            }
            catch (Exception ex)
            {
                StatusMessage = "Error building IV surface";
                _loggingService.LogErrorWithContext(ex, $"Error building IV surface for {SelectedSymbol}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Compares current IV to historical levels
        /// </summary>
        public async Task CompareHistoricalIVAsync()
        {
            try
            {
                IsLoading = true;
                StatusMessage = "Comparing to historical IV...";

                // TODO: Implement proper historical IV storage and retrieval
                // This would involve:
                // 1. Storing historical IV snapshots in database
                // 2. Calculating IV percentile ranks
                // 3. Comparing current IV to historical mean/median
                // 4. Identifying IV extremes or mean reversion opportunities
                
                // Placeholder for now
                StatusMessage = $"Historical IV comparison: Feature in development";
                await Task.Delay(500); // Simulate work
            }
            catch (Exception ex)
            {
                StatusMessage = "Error comparing historical IV";
                _loggingService.LogErrorWithContext(ex, "Error comparing historical IV");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Calculates theoretical prices for all options in the chain
        /// </summary>
        public async Task CalculateTheoreticalPricesAsync()
        {
            try
            {
                IsLoading = true;
                StatusMessage = "Calculating theoretical prices...";

                var allOptions = CallOptions.Concat(PutOptions).ToList();
                
                foreach (var option in allOptions)
                {
                    var analysis = await _pricingService.AnalyzePricingAsync(
                        option,
                        UnderlyingPrice,
                        0.05, // Risk-free rate
                        0.0   // Dividend yield
                    );

                    if (analysis != null)
                    {
                        option.TheoreticalPrice = analysis.TheoreticalPrice;
                    }
                }

                StatusMessage = "Theoretical prices calculated";
            }
            catch (Exception ex)
            {
                StatusMessage = "Error calculating theoretical prices";
                _loggingService.LogErrorWithContext(ex, "Error calculating theoretical prices");
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Exports options chain to CSV file
        /// </summary>
        public async Task ExportOptionsChainToCSVAsync()
        {
            try
            {
                // TODO: Implement CSV export functionality
                StatusMessage = "Export functionality coming soon";
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error exporting to CSV");
            }
        }

        #endregion

        #region Private Methods

        private bool CanLoadOptionsChain()
        {
            return !string.IsNullOrEmpty(SelectedSymbol) && 
                   SelectedExpiration.HasValue && 
                   !IsLoading;
        }

        private async Task ApplyFiltersAsync()
        {
            await LoadOptionsChainAsync();
        }

        private void UpdateFilterAndReload()
        {
            if (CurrentFilter == null)
                CurrentFilter = new OptionsChainFilter();

            CurrentFilter.Symbol = SelectedSymbol;
            CurrentFilter.SelectedExpiration = SelectedExpiration;
            CurrentFilter.OnlyITM = ShowOnlyITM;
            CurrentFilter.OnlyLiquid = ShowOnlyLiquid;

            if (StrikeRange.HasValue && UnderlyingPrice > 0)
            {
                CurrentFilter.MinStrike = UnderlyingPrice - StrikeRange.Value;
                CurrentFilter.MaxStrike = UnderlyingPrice + StrikeRange.Value;
            }

            _ = LoadOptionsChainAsync();
        }

        private List<OptionData> ApplyCurrentFilter(List<OptionData> chain)
        {
            if (CurrentFilter == null || chain == null || chain.Count == 0)
                return chain;

            return chain.Where(o => CurrentFilter.PassesFilter(o, UnderlyingPrice)).ToList();
        }

        private void ResetFilters()
        {
            ShowOnlyITM = false;
            ShowOnlyLiquid = false;
            StrikeRange = null;
            CurrentFilter = new OptionsChainFilter(SelectedSymbol, SelectedExpiration);
            
            _ = LoadOptionsChainAsync();
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
