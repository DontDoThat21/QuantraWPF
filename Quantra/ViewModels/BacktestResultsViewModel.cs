using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Media;
using Quantra.Commands;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the BacktestResults control
    /// </summary>
    public class BacktestResultsViewModel : ViewModelBase
    {
        private readonly HistoricalDataService _historicalDataService;
        private readonly CustomBenchmarkService _customBenchmarkService;
        private readonly IUserSettingsService _userSettingsService;
        private readonly IAlphaVantageService _alphaVantageService;

        private BacktestingEngine.BacktestResult _currentResult;
        private List<HistoricalPrice> _historicalData;
        private List<BenchmarkComparisonData> _benchmarkData;
        private double _strategyEquityVolatility;
        private bool _showRelativeReturns;
        private string _activeBenchmarkText;
        private bool _isMonteCarloRunning;
        private string _monteCarloStatusText;
        private int _selectedSimulationCount;

        // Performance metrics
        private string _totalReturnText;
        private string _maxDrawdownText;
        private string _winRateText;
        private string _cagrText;
        private string _sharpeRatioText;
        private string _sortinoRatioText;
        private string _calmarRatioText;
        private string _profitFactorText;
        private string _informationRatioText;

        // Monte Carlo statistics
        private string _return5PercentText;
        private string _return25PercentText;
        private string _return50PercentText;
        private string _return75PercentText;
        private string _return95PercentText;
        private string _drawdown5PercentText;
        private string _drawdown25PercentText;
        private string _drawdown50PercentText;
        private string _drawdown75PercentText;
        private string _drawdown95PercentText;
        private string _vaR95Text;
        private string _vaR99Text;
        private string _cVaR95Text;
        private string _profitProbabilityText;
        private string _beatBacktestProbabilityText;

        // Benchmark checkboxes
        private bool _spyChecked;
        private bool _qqqChecked;
        private bool _iwmChecked;
        private bool _diaChecked;

        private readonly Dictionary<string, Brush> _benchmarkColors = new Dictionary<string, Brush>
        {
            { "SPY", Brushes.DarkGreen },
            { "QQQ", Brushes.DarkBlue },
            { "IWM", Brushes.DarkOrange },
            { "DIA", Brushes.Purple },
            { "CUSTOM", Brushes.Magenta }
        };

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public BacktestResultsViewModel(
            HistoricalDataService historicalDataService,
            CustomBenchmarkService customBenchmarkService,
            IUserSettingsService userSettingsService,
            IAlphaVantageService alphaVantageService)
        {
            _historicalDataService = historicalDataService ?? throw new ArgumentNullException(nameof(historicalDataService));
            _customBenchmarkService = customBenchmarkService ?? throw new ArgumentNullException(nameof(customBenchmarkService));
            _userSettingsService = userSettingsService ?? throw new ArgumentNullException(nameof(userSettingsService));
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));

            _benchmarkData = new List<BenchmarkComparisonData>();
            CustomBenchmarks = new ObservableCollection<CustomBenchmark>();

            // Initialize default values
            _activeBenchmarkText = "S&P 500 (SPY)";
            _monteCarloStatusText = "Ready for simulation";
            _selectedSimulationCount = 1000;
            _showRelativeReturns = false;

            // Initialize performance metrics
            _totalReturnText = "0.00%";
            _maxDrawdownText = "0.00%";
            _winRateText = "0.00%";
            _cagrText = "0.00%";
            _sharpeRatioText = "0.00";
            _sortinoRatioText = "0.00";
            _calmarRatioText = "0.00";
            _profitFactorText = "0.00";
            _informationRatioText = "0.00";

            // Initialize Monte Carlo statistics
            _return5PercentText = "0.0%";
            _return25PercentText = "0.0%";
            _return50PercentText = "0.0%";
            _return75PercentText = "0.0%";
            _return95PercentText = "0.0%";
            _drawdown5PercentText = "0.0%";
            _drawdown25PercentText = "0.0%";
            _drawdown50PercentText = "0.0%";
            _drawdown75PercentText = "0.0%";
            _drawdown95PercentText = "0.0%";
            _vaR95Text = "0.0%";
            _vaR99Text = "0.0%";
            _cVaR95Text = "0.0%";
            _profitProbabilityText = "0.0%";
            _beatBacktestProbabilityText = "0.0%";

            // Initialize benchmark checkboxes - default SPY checked
            _spyChecked = true;
            _qqqChecked = true;
            _iwmChecked = true;

            InitializeCommands();
            LoadCustomBenchmarks();
            ApplyActiveBenchmarkSelection();
        }

        #region Services Access

        /// <summary>
        /// Historical data service
        /// </summary>
        public HistoricalDataService HistoricalDataService => _historicalDataService;

        /// <summary>
        /// Custom benchmark service
        /// </summary>
        public CustomBenchmarkService CustomBenchmarkService => _customBenchmarkService;

        /// <summary>
        /// User settings service
        /// </summary>
        public IUserSettingsService UserSettingsService => _userSettingsService;

        /// <summary>
        /// Alpha Vantage service
        /// </summary>
        public IAlphaVantageService AlphaVantageService => _alphaVantageService;

        #endregion

        #region Properties

        /// <summary>
        /// Current backtest result
        /// </summary>
        public BacktestingEngine.BacktestResult CurrentResult
        {
            get => _currentResult;
            set => SetProperty(ref _currentResult, value);
        }

        /// <summary>
        /// Historical price data
        /// </summary>
        public List<HistoricalPrice> HistoricalData
        {
            get => _historicalData;
            set => SetProperty(ref _historicalData, value);
        }

        /// <summary>
        /// Benchmark comparison data
        /// </summary>
        public List<BenchmarkComparisonData> BenchmarkData
        {
            get => _benchmarkData;
            set => SetProperty(ref _benchmarkData, value);
        }

        /// <summary>
        /// Custom benchmarks collection
        /// </summary>
        public ObservableCollection<CustomBenchmark> CustomBenchmarks { get; }

        /// <summary>
        /// Selected custom benchmark
        /// </summary>
        private CustomBenchmark _selectedCustomBenchmark;
        public CustomBenchmark SelectedCustomBenchmark
        {
            get => _selectedCustomBenchmark;
            set
            {
                if (SetProperty(ref _selectedCustomBenchmark, value) && value != null)
                {
                    _userSettingsService.SetActiveCustomBenchmark(value.Id);
                    UpdateActiveBenchmarkDisplay();
                }
            }
        }

        /// <summary>
        /// Strategy equity volatility
        /// </summary>
        public double StrategyEquityVolatility
        {
            get => _strategyEquityVolatility;
            set => SetProperty(ref _strategyEquityVolatility, value);
        }

        /// <summary>
        /// Whether to show relative returns
        /// </summary>
        public bool ShowRelativeReturns
        {
            get => _showRelativeReturns;
            set => SetProperty(ref _showRelativeReturns, value);
        }

        /// <summary>
        /// Active benchmark display text
        /// </summary>
        public string ActiveBenchmarkText
        {
            get => _activeBenchmarkText;
            set => SetProperty(ref _activeBenchmarkText, value);
        }

        /// <summary>
        /// Whether Monte Carlo simulation is running
        /// </summary>
        public bool IsMonteCarloRunning
        {
            get => _isMonteCarloRunning;
            set => SetProperty(ref _isMonteCarloRunning, value);
        }

        /// <summary>
        /// Monte Carlo status text
        /// </summary>
        public string MonteCarloStatusText
        {
            get => _monteCarloStatusText;
            set => SetProperty(ref _monteCarloStatusText, value);
        }

        /// <summary>
        /// Selected simulation count
        /// </summary>
        public int SelectedSimulationCount
        {
            get => _selectedSimulationCount;
            set => SetProperty(ref _selectedSimulationCount, value);
        }

        /// <summary>
        /// Benchmark colors dictionary
        /// </summary>
        public Dictionary<string, Brush> BenchmarkColors => _benchmarkColors;

        #region Benchmark Checkboxes

        public bool SpyChecked
        {
            get => _spyChecked;
            set
            {
                if (SetProperty(ref _spyChecked, value) && value)
                {
                    _userSettingsService.SetActiveBenchmark("SPY");
                    UpdateActiveBenchmarkDisplay();
                }
            }
        }

        public bool QqqChecked
        {
            get => _qqqChecked;
            set
            {
                if (SetProperty(ref _qqqChecked, value) && value)
                {
                    _userSettingsService.SetActiveBenchmark("QQQ");
                    UpdateActiveBenchmarkDisplay();
                }
            }
        }

        public bool IwmChecked
        {
            get => _iwmChecked;
            set
            {
                if (SetProperty(ref _iwmChecked, value) && value)
                {
                    _userSettingsService.SetActiveBenchmark("IWM");
                    UpdateActiveBenchmarkDisplay();
                }
            }
        }

        public bool DiaChecked
        {
            get => _diaChecked;
            set
            {
                if (SetProperty(ref _diaChecked, value) && value)
                {
                    _userSettingsService.SetActiveBenchmark("DIA");
                    UpdateActiveBenchmarkDisplay();
                }
            }
        }

        #endregion

        #region Performance Metrics

        public string TotalReturnText
        {
            get => _totalReturnText;
            set => SetProperty(ref _totalReturnText, value);
        }

        public string MaxDrawdownText
        {
            get => _maxDrawdownText;
            set => SetProperty(ref _maxDrawdownText, value);
        }

        public string WinRateText
        {
            get => _winRateText;
            set => SetProperty(ref _winRateText, value);
        }

        public string CagrText
        {
            get => _cagrText;
            set => SetProperty(ref _cagrText, value);
        }

        public string SharpeRatioText
        {
            get => _sharpeRatioText;
            set => SetProperty(ref _sharpeRatioText, value);
        }

        public string SortinoRatioText
        {
            get => _sortinoRatioText;
            set => SetProperty(ref _sortinoRatioText, value);
        }

        public string CalmarRatioText
        {
            get => _calmarRatioText;
            set => SetProperty(ref _calmarRatioText, value);
        }

        public string ProfitFactorText
        {
            get => _profitFactorText;
            set => SetProperty(ref _profitFactorText, value);
        }

        public string InformationRatioText
        {
            get => _informationRatioText;
            set => SetProperty(ref _informationRatioText, value);
        }

        #endregion

        #region Monte Carlo Statistics

        public string Return5PercentText
        {
            get => _return5PercentText;
            set => SetProperty(ref _return5PercentText, value);
        }

        public string Return25PercentText
        {
            get => _return25PercentText;
            set => SetProperty(ref _return25PercentText, value);
        }

        public string Return50PercentText
        {
            get => _return50PercentText;
            set => SetProperty(ref _return50PercentText, value);
        }

        public string Return75PercentText
        {
            get => _return75PercentText;
            set => SetProperty(ref _return75PercentText, value);
        }

        public string Return95PercentText
        {
            get => _return95PercentText;
            set => SetProperty(ref _return95PercentText, value);
        }

        public string Drawdown5PercentText
        {
            get => _drawdown5PercentText;
            set => SetProperty(ref _drawdown5PercentText, value);
        }

        public string Drawdown25PercentText
        {
            get => _drawdown25PercentText;
            set => SetProperty(ref _drawdown25PercentText, value);
        }

        public string Drawdown50PercentText
        {
            get => _drawdown50PercentText;
            set => SetProperty(ref _drawdown50PercentText, value);
        }

        public string Drawdown75PercentText
        {
            get => _drawdown75PercentText;
            set => SetProperty(ref _drawdown75PercentText, value);
        }

        public string Drawdown95PercentText
        {
            get => _drawdown95PercentText;
            set => SetProperty(ref _drawdown95PercentText, value);
        }

        public string VaR95Text
        {
            get => _vaR95Text;
            set => SetProperty(ref _vaR95Text, value);
        }

        public string VaR99Text
        {
            get => _vaR99Text;
            set => SetProperty(ref _vaR99Text, value);
        }

        public string CVaR95Text
        {
            get => _cVaR95Text;
            set => SetProperty(ref _cVaR95Text, value);
        }

        public string ProfitProbabilityText
        {
            get => _profitProbabilityText;
            set => SetProperty(ref _profitProbabilityText, value);
        }

        public string BeatBacktestProbabilityText
        {
            get => _beatBacktestProbabilityText;
            set => SetProperty(ref _beatBacktestProbabilityText, value);
        }

        #endregion

        #endregion

        #region Commands

        public ICommand RefreshBenchmarksCommand { get; private set; }
        public ICommand ManageCustomBenchmarksCommand { get; private set; }
        public ICommand RunMonteCarloCommand { get; private set; }
        public ICommand ResetChartZoomCommand { get; private set; }
        public ICommand HighlightOutperformanceCommand { get; private set; }

        #endregion

        #region Events

        /// <summary>
        /// Event fired when benchmark data is loaded
        /// </summary>
        public event EventHandler BenchmarkDataLoaded;

        /// <summary>
        /// Event fired when Monte Carlo simulation completes
        /// </summary>
        public event EventHandler MonteCarloCompleted;

        /// <summary>
        /// Event fired when chart zoom should be reset
        /// </summary>
        public event EventHandler ResetZoomRequested;

        /// <summary>
        /// Event fired when outperformance highlighting is requested
        /// </summary>
        public event EventHandler HighlightOutperformanceRequested;

        /// <summary>
        /// Event fired when custom benchmark management is requested
        /// </summary>
        public event EventHandler ManageCustomBenchmarksRequested;

        #endregion

        #region Public Methods

        /// <summary>
        /// Load results from a backtest
        /// </summary>
        public void LoadResults(BacktestingEngine.BacktestResult result, List<HistoricalPrice> historical)
        {
            CurrentResult = result;
            HistoricalData = historical;

            // Calculate equity volatility
            CalculateStrategyVolatility();

            // Update performance metrics
            UpdatePerformanceMetrics(result);

            // Update Monte Carlo status
            MonteCarloStatusText = "Ready for simulation";

            // Update Monte Carlo statistics if results are available
            if (result.HasMonteCarloResults)
            {
                UpdateMonteCarloStatistics(result.MonteCarloResults);
            }
        }

        /// <summary>
        /// Load benchmark data asynchronously
        /// </summary>
        public async Task LoadBenchmarkDataAsync()
        {
            if (_currentResult == null || _historicalData == null || _historicalData.Count == 0)
            {
                return;
            }

            try
            {
                _benchmarkData.Clear();

                var selectedBenchmarks = GetSelectedBenchmarks();

                if (selectedBenchmarks.Count == 0)
                {
                    return;
                }

                DateTime startDate = _currentResult.StartDate;
                DateTime endDate = _currentResult.EndDate;

                foreach (var benchmarkInfo in selectedBenchmarks)
                {
                    var customBenchmark = CustomBenchmarks.FirstOrDefault(b => b.DisplaySymbol == benchmarkInfo.symbol);

                    BenchmarkComparisonData benchmarkData;

                    if (customBenchmark != null)
                    {
                        benchmarkData = await _customBenchmarkService.CalculateCustomBenchmarkData(
                            customBenchmark, startDate, endDate);
                    }
                    else
                    {
                        benchmarkData = await LoadBenchmarkHistoricalDataAsync(benchmarkInfo.symbol, benchmarkInfo.name, startDate, endDate);
                    }

                    if (benchmarkData != null)
                    {
                        _benchmarkData.Add(benchmarkData);
                    }
                }

                BenchmarkDataLoaded?.Invoke(this, EventArgs.Empty);
            }
            catch (Exception)
            {
                // Log error if needed
            }
        }

        /// <summary>
        /// Run Monte Carlo simulation
        /// </summary>
        public async Task RunMonteCarloSimulationAsync()
        {
            if (_currentResult == null)
            {
                MonteCarloStatusText = "No backtest results to simulate";
                return;
            }

            try
            {
                IsMonteCarloRunning = true;
                MonteCarloStatusText = $"Running {SelectedSimulationCount} simulations...";

                var engine = new BacktestingEngine(_historicalDataService);

                await Task.Run(() =>
                {
                    CurrentResult = engine.RunMonteCarloSimulation(_currentResult, SelectedSimulationCount);
                });

                UpdateMonteCarloStatistics(_currentResult.MonteCarloResults);
                MonteCarloStatusText = $"Completed {SelectedSimulationCount} simulations";

                MonteCarloCompleted?.Invoke(this, EventArgs.Empty);
            }
            catch (Exception)
            {
                MonteCarloStatusText = "Simulation failed";
            }
            finally
            {
                IsMonteCarloRunning = false;
            }
        }

        /// <summary>
        /// Get the color for a benchmark symbol
        /// </summary>
        public Brush GetBenchmarkColor(string symbol)
        {
            if (_benchmarkColors.ContainsKey(symbol))
                return _benchmarkColors[symbol];

            if (symbol.Contains("+") || symbol.Contains("%"))
            {
                return _benchmarkColors["CUSTOM"];
            }

            return Brushes.Gray;
        }

        /// <summary>
        /// Get selected benchmarks
        /// </summary>
        public List<(string symbol, string name)> GetSelectedBenchmarks()
        {
            var benchmarks = new List<(string symbol, string name)>();
            var (activeBenchmarkType, activeBenchmarkId) = _userSettingsService.GetActiveBenchmark();

            bool activeBenchmarkAdded = false;

            if (activeBenchmarkType == "CUSTOM" && !string.IsNullOrEmpty(activeBenchmarkId))
            {
                var activeCustomBenchmark = CustomBenchmarks.FirstOrDefault(b => b.Id == activeBenchmarkId);
                if (activeCustomBenchmark != null)
                {
                    benchmarks.Add((activeCustomBenchmark.DisplaySymbol, activeCustomBenchmark.Name));
                    activeBenchmarkAdded = true;
                }
            }
            else
            {
                switch (activeBenchmarkType)
                {
                    case "SPY":
                        if (SpyChecked)
                        {
                            benchmarks.Add(("SPY", "S&P 500"));
                            activeBenchmarkAdded = true;
                        }
                        break;
                    case "QQQ":
                        if (QqqChecked)
                        {
                            benchmarks.Add(("QQQ", "NASDAQ"));
                            activeBenchmarkAdded = true;
                        }
                        break;
                    case "IWM":
                        if (IwmChecked)
                        {
                            benchmarks.Add(("IWM", "Russell 2000"));
                            activeBenchmarkAdded = true;
                        }
                        break;
                    case "DIA":
                        if (DiaChecked)
                        {
                            benchmarks.Add(("DIA", "Dow Jones"));
                            activeBenchmarkAdded = true;
                        }
                        break;
                }
            }

            if (SpyChecked && (activeBenchmarkType != "SPY" || !activeBenchmarkAdded))
                benchmarks.Add(("SPY", "S&P 500"));

            if (QqqChecked && (activeBenchmarkType != "QQQ" || !activeBenchmarkAdded))
                benchmarks.Add(("QQQ", "NASDAQ"));

            if (IwmChecked && (activeBenchmarkType != "IWM" || !activeBenchmarkAdded))
                benchmarks.Add(("IWM", "Russell 2000"));

            if (DiaChecked && (activeBenchmarkType != "DIA" || !activeBenchmarkAdded))
                benchmarks.Add(("DIA", "Dow Jones"));

            if (SelectedCustomBenchmark != null &&
                (activeBenchmarkType != "CUSTOM" || activeBenchmarkId != SelectedCustomBenchmark.Id || !activeBenchmarkAdded))
            {
                benchmarks.Add((SelectedCustomBenchmark.DisplaySymbol, SelectedCustomBenchmark.Name));
            }

            return benchmarks;
        }

        /// <summary>
        /// Refresh custom benchmarks list
        /// </summary>
        public void LoadCustomBenchmarks()
        {
            CustomBenchmark currentSelection = SelectedCustomBenchmark;

            CustomBenchmarks.Clear();

            var benchmarks = _customBenchmarkService.GetCustomBenchmarks();
            foreach (var benchmark in benchmarks)
            {
                CustomBenchmarks.Add(benchmark);
            }

            if (currentSelection != null)
            {
                SelectedCustomBenchmark = CustomBenchmarks.FirstOrDefault(b => b.Id == currentSelection.Id);
            }

            UpdateActiveBenchmarkDisplay();
        }

        /// <summary>
        /// Apply the active benchmark selection from user settings
        /// </summary>
        public void ApplyActiveBenchmarkSelection()
        {
            var (benchmarkType, benchmarkId) = _userSettingsService.GetActiveBenchmark();

            // Clear all selections first
            _spyChecked = false;
            _qqqChecked = false;
            _iwmChecked = false;
            _diaChecked = false;
            _selectedCustomBenchmark = null;

            // Set the active benchmark based on user settings
            switch (benchmarkType)
            {
                case "SPY":
                    _spyChecked = true;
                    break;
                case "QQQ":
                    _qqqChecked = true;
                    break;
                case "IWM":
                    _iwmChecked = true;
                    break;
                case "DIA":
                    _diaChecked = true;
                    break;
                case "CUSTOM":
                    if (!string.IsNullOrEmpty(benchmarkId))
                    {
                        _selectedCustomBenchmark = CustomBenchmarks.FirstOrDefault(b => b.Id == benchmarkId);
                    }
                    break;
            }

            // Notify property changes
            OnPropertyChanged(nameof(SpyChecked));
            OnPropertyChanged(nameof(QqqChecked));
            OnPropertyChanged(nameof(IwmChecked));
            OnPropertyChanged(nameof(DiaChecked));
            OnPropertyChanged(nameof(SelectedCustomBenchmark));

            UpdateActiveBenchmarkDisplay();
        }

        #endregion

        #region Private Methods

        private void InitializeCommands()
        {
            RefreshBenchmarksCommand = new RelayCommand(async _ => await LoadBenchmarkDataAsync());
            ManageCustomBenchmarksCommand = new RelayCommand(_ => ManageCustomBenchmarksRequested?.Invoke(this, EventArgs.Empty));
            RunMonteCarloCommand = new RelayCommand(async _ => await RunMonteCarloSimulationAsync(), _ => !IsMonteCarloRunning);
            ResetChartZoomCommand = new RelayCommand(_ => ResetZoomRequested?.Invoke(this, EventArgs.Empty));
            HighlightOutperformanceCommand = new RelayCommand(_ => HighlightOutperformanceRequested?.Invoke(this, EventArgs.Empty));
        }

        private void UpdatePerformanceMetrics(BacktestingEngine.BacktestResult result)
        {
            TotalReturnText = result.TotalReturn.ToString("P2");
            MaxDrawdownText = result.MaxDrawdown.ToString("P2");
            WinRateText = result.WinRate.ToString("P2");
            CagrText = result.CAGR.ToString("P2");
            SharpeRatioText = result.SharpeRatio.ToString("F2");
            SortinoRatioText = result.SortinoRatio.ToString("F2");
            CalmarRatioText = result.CalmarRatio.ToString("F2");
            ProfitFactorText = result.ProfitFactor.ToString("F2");
            InformationRatioText = result.InformationRatio.ToString("F2");
        }

        private void UpdateMonteCarloStatistics(BacktestingEngine.MonteCarloSimulationResult mcResult)
        {
            if (mcResult == null || _currentResult == null)
                return;

            double initialCapital = _currentResult.EquityCurve.FirstOrDefault()?.Equity ?? 10000;

            Return5PercentText = ((mcResult.ReturnPercentiles["5%"] - initialCapital) / initialCapital).ToString("P2");
            Return25PercentText = ((mcResult.ReturnPercentiles["25%"] - initialCapital) / initialCapital).ToString("P2");
            Return50PercentText = ((mcResult.ReturnPercentiles["50%"] - initialCapital) / initialCapital).ToString("P2");
            Return75PercentText = ((mcResult.ReturnPercentiles["75%"] - initialCapital) / initialCapital).ToString("P2");
            Return95PercentText = ((mcResult.ReturnPercentiles["95%"] - initialCapital) / initialCapital).ToString("P2");

            Drawdown5PercentText = mcResult.DrawdownPercentiles["5%"].ToString("P2");
            Drawdown25PercentText = mcResult.DrawdownPercentiles["25%"].ToString("P2");
            Drawdown50PercentText = mcResult.DrawdownPercentiles["50%"].ToString("P2");
            Drawdown75PercentText = mcResult.DrawdownPercentiles["75%"].ToString("P2");
            Drawdown95PercentText = mcResult.DrawdownPercentiles["95%"].ToString("P2");

            VaR95Text = mcResult.ValueAtRisk95.ToString("P2");
            VaR99Text = mcResult.ValueAtRisk99.ToString("P2");
            CVaR95Text = mcResult.ConditionalValueAtRisk95.ToString("P2");
            ProfitProbabilityText = mcResult.ProbabilityOfProfit.ToString("P1");
            BeatBacktestProbabilityText = mcResult.ProbabilityOfExceedingBacktestReturn.ToString("P1");
        }

        private void CalculateStrategyVolatility()
        {
            if (_currentResult == null || _currentResult.EquityCurve == null || _currentResult.EquityCurve.Count <= 1)
            {
                StrategyEquityVolatility = 0;
                return;
            }

            var equityValues = _currentResult.EquityCurve.Select(e => e.Equity).ToList();
            var dailyReturns = new List<double>();

            for (int i = 1; i < equityValues.Count; i++)
            {
                double dailyReturn = (equityValues[i] - equityValues[i - 1]) / equityValues[i - 1];
                dailyReturns.Add(dailyReturn);
            }

            StrategyEquityVolatility = CalculateStandardDeviation(dailyReturns);
        }

        private double CalculateStandardDeviation(List<double> values)
        {
            if (values == null || values.Count <= 1)
                return 0;

            double avg = values.Average();
            double sumOfSquaresOfDifferences = values.Sum(val => Math.Pow(val - avg, 2));
            return Math.Sqrt(sumOfSquaresOfDifferences / (values.Count - 1));
        }

        private void UpdateActiveBenchmarkDisplay()
        {
            var (benchmarkType, benchmarkId) = _userSettingsService.GetActiveBenchmark();

            switch (benchmarkType)
            {
                case "SPY":
                    ActiveBenchmarkText = "S&P 500 (SPY)";
                    break;
                case "QQQ":
                    ActiveBenchmarkText = "NASDAQ (QQQ)";
                    break;
                case "IWM":
                    ActiveBenchmarkText = "Russell 2000 (IWM)";
                    break;
                case "DIA":
                    ActiveBenchmarkText = "Dow Jones (DIA)";
                    break;
                case "CUSTOM":
                    if (!string.IsNullOrEmpty(benchmarkId))
                    {
                        var customBenchmark = CustomBenchmarks.FirstOrDefault(b => b.Id == benchmarkId);
                        if (customBenchmark != null)
                        {
                            ActiveBenchmarkText = $"{customBenchmark.Name} ({customBenchmark.DisplaySymbol})";
                        }
                        else
                        {
                            ActiveBenchmarkText = "Custom (Not Found)";
                        }
                    }
                    else
                    {
                        ActiveBenchmarkText = "Custom (None Selected)";
                    }
                    break;
                default:
                    ActiveBenchmarkText = "S&P 500 (SPY)";
                    break;
            }
        }

        private async Task<BenchmarkComparisonData> LoadBenchmarkHistoricalDataAsync(string symbol, string name, DateTime startDate, DateTime endDate)
        {
            try
            {
                var historicalData = await _historicalDataService.GetComprehensiveHistoricalData(symbol);

                var filteredData = historicalData
                    .Where(h => h.Date >= startDate && h.Date <= endDate)
                    .OrderBy(h => h.Date)
                    .ToList();

                if (filteredData.Count == 0)
                {
                    return null;
                }

                var benchmarkData = new BenchmarkComparisonData
                {
                    Symbol = symbol,
                    Name = name,
                    HistoricalData = filteredData,
                    Dates = filteredData.Select(h => h.Date).ToList()
                };

                double initialPrice = filteredData.First().Close;
                benchmarkData.NormalizedReturns = filteredData
                    .Select(h => h.Close / initialPrice)
                    .ToList();

                benchmarkData.TotalReturn = (filteredData.Last().Close / filteredData.First().Close) - 1;

                double peak = filteredData.First().Close;
                double maxDrawdown = 0;
                List<double> dailyReturns = new List<double>();
                List<double> downsideReturns = new List<double>();

                for (int i = 1; i < filteredData.Count; i++)
                {
                    if (filteredData[i].Close > peak)
                    {
                        peak = filteredData[i].Close;
                    }

                    double drawdown = (peak - filteredData[i].Close) / peak;
                    if (drawdown > maxDrawdown)
                    {
                        maxDrawdown = drawdown;
                    }

                    double dailyReturn = (filteredData[i].Close - filteredData[i - 1].Close) / filteredData[i - 1].Close;
                    dailyReturns.Add(dailyReturn);

                    if (dailyReturn < 0)
                    {
                        downsideReturns.Add(dailyReturn);
                    }
                }

                benchmarkData.MaxDrawdown = maxDrawdown;
                benchmarkData.Volatility = CalculateStandardDeviation(dailyReturns);

                double totalDays = (endDate - startDate).TotalDays;
                if (totalDays > 0 && filteredData.Count > 0)
                {
                    benchmarkData.CAGR = Math.Pow(1 + benchmarkData.TotalReturn, 365.0 / totalDays) - 1;
                }

                double averageReturn = dailyReturns.Count > 0 ? dailyReturns.Average() : 0;
                double riskFreeRate = 0.0;

                benchmarkData.SharpeRatio = benchmarkData.Volatility > 0 ?
                    (averageReturn - riskFreeRate) / benchmarkData.Volatility * Math.Sqrt(252) : 0;

                double downsideDeviation = CalculateStandardDeviation(downsideReturns);
                benchmarkData.SortinoRatio = downsideDeviation > 0 ?
                    (averageReturn - riskFreeRate) / downsideDeviation * Math.Sqrt(252) : 0;

                benchmarkData.CalmarRatio = benchmarkData.MaxDrawdown > 0 ?
                    benchmarkData.CAGR / benchmarkData.MaxDrawdown : 0;

                benchmarkData.InformationRatio = benchmarkData.Volatility > 0 ?
                    (averageReturn - riskFreeRate) / benchmarkData.Volatility * Math.Sqrt(252) : 0;

                if (_currentResult != null && _currentResult.EquityCurve.Count > 0)
                {
                    var alignedReturns = AlignReturnsForComparison(_currentResult, filteredData);
                    if (alignedReturns.strategyReturns.Count > 0 && alignedReturns.benchmarkReturns.Count > 0)
                    {
                        benchmarkData.Beta = CalculateBeta(alignedReturns.strategyReturns, alignedReturns.benchmarkReturns);
                        benchmarkData.Alpha = CalculateAlpha(alignedReturns.strategyReturns, alignedReturns.benchmarkReturns, benchmarkData.Beta, riskFreeRate);
                        benchmarkData.Correlation = CalculateCorrelation(alignedReturns.strategyReturns, alignedReturns.benchmarkReturns);
                    }
                }

                return benchmarkData;
            }
            catch (Exception)
            {
                return null;
            }
        }

        private (List<double> strategyReturns, List<double> benchmarkReturns) AlignReturnsForComparison(
            BacktestingEngine.BacktestResult strategyResult,
            List<HistoricalPrice> benchmarkData)
        {
            List<double> strategyReturns = new List<double>();
            List<double> benchmarkReturns = new List<double>();

            var benchmarkByDate = benchmarkData.ToDictionary(h => h.Date.Date, h => h);

            for (int i = 1; i < strategyResult.EquityCurve.Count; i++)
            {
                DateTime currentDate = strategyResult.EquityCurve[i].Date.Date;
                DateTime previousDate = strategyResult.EquityCurve[i - 1].Date.Date;

                if (benchmarkByDate.ContainsKey(currentDate) && benchmarkByDate.ContainsKey(previousDate))
                {
                    double strategyReturn = (strategyResult.EquityCurve[i].Equity - strategyResult.EquityCurve[i - 1].Equity) /
                                          strategyResult.EquityCurve[i - 1].Equity;

                    double benchmarkReturn = (benchmarkByDate[currentDate].Close - benchmarkByDate[previousDate].Close) /
                                          benchmarkByDate[previousDate].Close;

                    strategyReturns.Add(strategyReturn);
                    benchmarkReturns.Add(benchmarkReturn);
                }
            }

            return (strategyReturns, benchmarkReturns);
        }

        private double CalculateBeta(List<double> strategyReturns, List<double> benchmarkReturns)
        {
            if (strategyReturns.Count != benchmarkReturns.Count || strategyReturns.Count < 2)
                return 1;

            double covariance = CalculateCovariance(strategyReturns, benchmarkReturns);
            double benchmarkVariance = CalculateVariance(benchmarkReturns);

            return benchmarkVariance != 0 ? covariance / benchmarkVariance : 1;
        }

        private double CalculateAlpha(List<double> strategyReturns, List<double> benchmarkReturns, double beta, double riskFreeRate)
        {
            if (strategyReturns.Count != benchmarkReturns.Count || strategyReturns.Count < 2)
                return 0;

            double avgStrategyReturn = strategyReturns.Average();
            double avgBenchmarkReturn = benchmarkReturns.Average();

            double annualFactor = 252;
            return (avgStrategyReturn - riskFreeRate) - beta * (avgBenchmarkReturn - riskFreeRate) * annualFactor;
        }

        private double CalculateCovariance(List<double> x, List<double> y)
        {
            if (x.Count != y.Count || x.Count < 2)
                return 0;

            double xMean = x.Average();
            double yMean = y.Average();
            double sum = 0;

            for (int i = 0; i < x.Count; i++)
            {
                sum += (x[i] - xMean) * (y[i] - yMean);
            }

            return sum / (x.Count - 1);
        }

        private double CalculateVariance(List<double> values)
        {
            if (values.Count < 2)
                return 0;

            double mean = values.Average();
            double sum = 0;

            foreach (double val in values)
            {
                sum += Math.Pow(val - mean, 2);
            }

            return sum / (values.Count - 1);
        }

        private double CalculateCorrelation(List<double> x, List<double> y)
        {
            if (x.Count != y.Count || x.Count < 2)
                return 0;

            double xStdDev = CalculateStandardDeviation(x);
            double yStdDev = CalculateStandardDeviation(y);
            double covariance = CalculateCovariance(x, y);

            return xStdDev > 0 && yStdDev > 0 ? covariance / (xStdDev * yStdDev) : 0;
        }

        #endregion
    }
}
