using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Input;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for Company Overview Control
    /// </summary>
    public class CompanyOverviewViewModel : INotifyPropertyChanged
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;

        private string _symbol;
        private CompanyOverview _companyOverview;
        private bool _isLoading;
        private string _statusMessage;
        private string _errorMessage;

        public event PropertyChangedEventHandler PropertyChanged;

        public CompanyOverviewViewModel(AlphaVantageService alphaVantageService, LoggingService loggingService)
        {
            _alphaVantageService = alphaVantageService;
            _loggingService = loggingService;
            StatusMessage = "Enter a symbol to load company data.";
        }

        #region Properties

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

        public CompanyOverview CompanyOverview
        {
            get => _companyOverview;
            set
            {
                if (_companyOverview != value)
                {
                    _companyOverview = value;
                    OnPropertyChanged();
                    OnPropertyChanged(nameof(HasData));
                }
            }
        }

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

        public string ErrorMessage
        {
            get => _errorMessage;
            set
            {
                if (_errorMessage != value)
                {
                    _errorMessage = value;
                    OnPropertyChanged();
                    OnPropertyChanged(nameof(HasError));
                }
            }
        }

        public bool HasData => CompanyOverview != null;
        public bool HasError => !string.IsNullOrEmpty(ErrorMessage);

        #endregion

        #region Methods

        public async Task LoadCompanyOverviewAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                ErrorMessage = "Please enter a valid symbol.";
                return;
            }

            try
            {
                IsLoading = true;
                ErrorMessage = null;
                StatusMessage = $"Loading data for {symbol.ToUpper()}...";
                Symbol = symbol.ToUpper();

                var overview = await _alphaVantageService.GetCompanyOverviewAsync(Symbol);

                if (overview != null)
                {
                    CompanyOverview = overview;
                    StatusMessage = $"Last updated: {overview.LastUpdated:g}";
                    _loggingService?.Log("Info", $"Loaded company overview for {Symbol}");
                }
                else
                {
                    ErrorMessage = $"No data found for symbol '{Symbol}'. Please check the symbol and try again.";
                    StatusMessage = "No data found.";
                }
            }
            catch (Exception ex)
            {
                ErrorMessage = $"Error loading data: {ex.Message}";
                StatusMessage = "Error occurred.";
                _loggingService?.LogErrorWithContext(ex, $"Error loading company overview for {symbol}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        #endregion

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    /// <summary>
    /// ViewModel for Cash Flow Control
    /// </summary>
    public class CashFlowViewModel : INotifyPropertyChanged
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;

        private string _symbol;
        private CashFlowStatement _cashFlowStatement;
        private bool _isLoading;
        private string _statusMessage;
        private string _errorMessage;
        private bool _showQuarterly = true;
        private ObservableCollection<CashFlowReport> _displayedReports;

        public event PropertyChangedEventHandler PropertyChanged;

        public CashFlowViewModel(AlphaVantageService alphaVantageService, LoggingService loggingService)
        {
            _alphaVantageService = alphaVantageService;
            _loggingService = loggingService;
            DisplayedReports = new ObservableCollection<CashFlowReport>();
            StatusMessage = "Enter a symbol to load cash flow data.";
        }

        #region Properties

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

        public CashFlowStatement CashFlowStatement
        {
            get => _cashFlowStatement;
            set
            {
                if (_cashFlowStatement != value)
                {
                    _cashFlowStatement = value;
                    OnPropertyChanged();
                    OnPropertyChanged(nameof(HasData));
                    UpdateDisplayedReports();
                }
            }
        }

        public ObservableCollection<CashFlowReport> DisplayedReports
        {
            get => _displayedReports;
            set
            {
                if (_displayedReports != value)
                {
                    _displayedReports = value;
                    OnPropertyChanged();
                }
            }
        }

        public bool ShowQuarterly
        {
            get => _showQuarterly;
            set
            {
                if (_showQuarterly != value)
                {
                    _showQuarterly = value;
                    OnPropertyChanged();
                    UpdateDisplayedReports();
                }
            }
        }

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

        public string ErrorMessage
        {
            get => _errorMessage;
            set
            {
                if (_errorMessage != value)
                {
                    _errorMessage = value;
                    OnPropertyChanged();
                    OnPropertyChanged(nameof(HasError));
                }
            }
        }

        public bool HasData => CashFlowStatement != null;
        public bool HasError => !string.IsNullOrEmpty(ErrorMessage);

        #endregion

        #region Methods

        public async Task LoadCashFlowAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                ErrorMessage = "Please enter a valid symbol.";
                return;
            }

            try
            {
                IsLoading = true;
                ErrorMessage = null;
                StatusMessage = $"Loading cash flow data for {symbol.ToUpper()}...";
                Symbol = symbol.ToUpper();

                var cashFlow = await _alphaVantageService.GetCashFlowAsync(Symbol);

                if (cashFlow != null)
                {
                    CashFlowStatement = cashFlow;
                    StatusMessage = $"Last updated: {cashFlow.LastUpdated:g}";
                    _loggingService?.Log("Info", $"Loaded cash flow for {Symbol}");
                }
                else
                {
                    ErrorMessage = $"No data found for symbol '{Symbol}'. Please check the symbol and try again.";
                    StatusMessage = "No data found.";
                }
            }
            catch (Exception ex)
            {
                ErrorMessage = $"Error loading data: {ex.Message}";
                StatusMessage = "Error occurred.";
                _loggingService?.LogErrorWithContext(ex, $"Error loading cash flow for {symbol}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        private void UpdateDisplayedReports()
        {
            DisplayedReports.Clear();
            if (CashFlowStatement == null) return;

            var reports = ShowQuarterly ? CashFlowStatement.QuarterlyReports : CashFlowStatement.AnnualReports;
            foreach (var report in reports)
            {
                DisplayedReports.Add(report);
            }
        }

        #endregion

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    /// <summary>
    /// ViewModel for Earnings Control
    /// </summary>
    public class EarningsViewModel : INotifyPropertyChanged
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;

        private string _symbol;
        private EarningsData _earningsData;
        private bool _isLoading;
        private string _statusMessage;
        private string _errorMessage;
        private bool _showQuarterly = true;
        private ObservableCollection<QuarterlyEarningsReport> _quarterlyReports;
        private ObservableCollection<AnnualEarningsReport> _annualReports;

        public event PropertyChangedEventHandler PropertyChanged;

        public EarningsViewModel(AlphaVantageService alphaVantageService, LoggingService loggingService)
        {
            _alphaVantageService = alphaVantageService;
            _loggingService = loggingService;
            QuarterlyReports = new ObservableCollection<QuarterlyEarningsReport>();
            AnnualReports = new ObservableCollection<AnnualEarningsReport>();
            StatusMessage = "Enter a symbol to load earnings data.";
        }

        #region Properties

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

        public EarningsData EarningsData
        {
            get => _earningsData;
            set
            {
                if (_earningsData != value)
                {
                    _earningsData = value;
                    OnPropertyChanged();
                    OnPropertyChanged(nameof(HasData));
                    OnPropertyChanged(nameof(UpcomingEarningsDate));
                    UpdateDisplayedReports();
                }
            }
        }

        public ObservableCollection<QuarterlyEarningsReport> QuarterlyReports
        {
            get => _quarterlyReports;
            set
            {
                if (_quarterlyReports != value)
                {
                    _quarterlyReports = value;
                    OnPropertyChanged();
                }
            }
        }

        public ObservableCollection<AnnualEarningsReport> AnnualReports
        {
            get => _annualReports;
            set
            {
                if (_annualReports != value)
                {
                    _annualReports = value;
                    OnPropertyChanged();
                }
            }
        }

        public bool ShowQuarterly
        {
            get => _showQuarterly;
            set
            {
                if (_showQuarterly != value)
                {
                    _showQuarterly = value;
                    OnPropertyChanged();
                }
            }
        }

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

        public string ErrorMessage
        {
            get => _errorMessage;
            set
            {
                if (_errorMessage != value)
                {
                    _errorMessage = value;
                    OnPropertyChanged();
                    OnPropertyChanged(nameof(HasError));
                }
            }
        }

        public string UpcomingEarningsDate => EarningsData?.UpcomingEarningsDate ?? "N/A";

        public bool HasData => EarningsData != null;
        public bool HasError => !string.IsNullOrEmpty(ErrorMessage);

        #endregion

        #region Methods

        public async Task LoadEarningsAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                ErrorMessage = "Please enter a valid symbol.";
                return;
            }

            try
            {
                IsLoading = true;
                ErrorMessage = null;
                StatusMessage = $"Loading earnings data for {symbol.ToUpper()}...";
                Symbol = symbol.ToUpper();

                var earnings = await _alphaVantageService.GetEarningsAsync(Symbol);

                if (earnings != null)
                {
                    EarningsData = earnings;
                    StatusMessage = $"Last updated: {earnings.LastUpdated:g}";
                    _loggingService?.Log("Info", $"Loaded earnings for {Symbol}");
                }
                else
                {
                    ErrorMessage = $"No data found for symbol '{Symbol}'. Please check the symbol and try again.";
                    StatusMessage = "No data found.";
                }
            }
            catch (Exception ex)
            {
                ErrorMessage = $"Error loading data: {ex.Message}";
                StatusMessage = "Error occurred.";
                _loggingService?.LogErrorWithContext(ex, $"Error loading earnings for {symbol}");
            }
            finally
            {
                IsLoading = false;
            }
        }

        private void UpdateDisplayedReports()
        {
            QuarterlyReports.Clear();
            AnnualReports.Clear();

            if (EarningsData == null) return;

            foreach (var report in EarningsData.QuarterlyEarnings)
            {
                QuarterlyReports.Add(report);
            }

            foreach (var report in EarningsData.AnnualEarnings)
            {
                AnnualReports.Add(report);
            }
        }

        #endregion

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
