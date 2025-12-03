using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using Quantra.Commands;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Signal Creation Control
    /// </summary>
    public class SignalCreationViewModel : ViewModelBase
    {
        private readonly ITradingSignalService _tradingSignalService;

        private TradingSignal _currentSignal;
        private bool _isEditing;
        private bool _isBusy;
        private string _statusMessage;
        private SignalSymbol _selectedSymbol;
        private SignalIndicator _selectedIndicator;
        private SignalCondition _selectedCondition;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public SignalCreationViewModel(ITradingSignalService tradingSignalService)
        {
            _tradingSignalService = tradingSignalService ?? throw new ArgumentNullException(nameof(tradingSignalService));

            // Initialize collections
            Signals = new ObservableCollection<TradingSignal>();
            AvailableIndicators = new ObservableCollection<SignalIndicator>();
            ComparisonOperators = new ObservableCollection<string>(SignalCondition.GetComparisonOperators());
            LogicalOperators = new ObservableCollection<string>(SignalCondition.GetLogicalOperators());
            SeverityLevels = new ObservableCollection<string>(SignalAlertConfiguration.GetSeverityLevels());

            InitializeIndicators();
            InitializeCommands();

            // Create a new signal by default
            CreateNewSignal();

            // Load existing signals
            LoadSignalsAsync();
        }

        #region Properties

        /// <summary>
        /// Collection of all trading signals
        /// </summary>
        public ObservableCollection<TradingSignal> Signals { get; }

        /// <summary>
        /// Collection of available technical indicators
        /// </summary>
        public ObservableCollection<SignalIndicator> AvailableIndicators { get; }

        /// <summary>
        /// Collection of comparison operators
        /// </summary>
        public ObservableCollection<string> ComparisonOperators { get; }

        /// <summary>
        /// Collection of logical operators
        /// </summary>
        public ObservableCollection<string> LogicalOperators { get; }

        /// <summary>
        /// Collection of severity levels
        /// </summary>
        public ObservableCollection<string> SeverityLevels { get; }

        /// <summary>
        /// Current signal being edited
        /// </summary>
        public TradingSignal CurrentSignal
        {
            get => _currentSignal;
            set
            {
                if (SetProperty(ref _currentSignal, value))
                {
                    UpdateIndicatorSelections();
                }
            }
        }

        /// <summary>
        /// Indicates if currently editing a signal
        /// </summary>
        public bool IsEditing
        {
            get => _isEditing;
            set => SetProperty(ref _isEditing, value);
        }

        /// <summary>
        /// Indicates if a background operation is in progress
        /// </summary>
        public bool IsBusy
        {
            get => _isBusy;
            set => SetProperty(ref _isBusy, value);
        }

        /// <summary>
        /// Status message for the user
        /// </summary>
        public string StatusMessage
        {
            get => _statusMessage;
            set => SetProperty(ref _statusMessage, value);
        }

        /// <summary>
        /// Currently selected symbol
        /// </summary>
        public SignalSymbol SelectedSymbol
        {
            get => _selectedSymbol;
            set => SetProperty(ref _selectedSymbol, value);
        }

        /// <summary>
        /// Currently selected indicator
        /// </summary>
        public SignalIndicator SelectedIndicator
        {
            get => _selectedIndicator;
            set => SetProperty(ref _selectedIndicator, value);
        }

        /// <summary>
        /// Currently selected condition
        /// </summary>
        public SignalCondition SelectedCondition
        {
            get => _selectedCondition;
            set => SetProperty(ref _selectedCondition, value);
        }

        #endregion

        #region Commands

        public ICommand NewSignalCommand { get; private set; }
        public ICommand SaveSignalCommand { get; private set; }
        public ICommand DeleteSignalCommand { get; private set; }
        public ICommand AddSymbolCommand { get; private set; }
        public ICommand RemoveSymbolCommand { get; private set; }
        public ICommand ValidateSymbolCommand { get; private set; }
        public ICommand AddConditionCommand { get; private set; }
        public ICommand RemoveConditionCommand { get; private set; }
        public ICommand LoadSignalCommand { get; private set; }
        public ICommand RefreshCommand { get; private set; }

        #endregion

        #region Private Methods

        private void InitializeCommands()
        {
            NewSignalCommand = new RelayCommand(ExecuteNewSignal);
            SaveSignalCommand = new RelayCommand(async param => await ExecuteSaveSignalAsync(), CanExecuteSaveSignal);
            DeleteSignalCommand = new RelayCommand(async param => await ExecuteDeleteSignalAsync(), CanExecuteDeleteSignal);
            AddSymbolCommand = new RelayCommand(ExecuteAddSymbol);
            RemoveSymbolCommand = new RelayCommand(ExecuteRemoveSymbol, CanExecuteRemoveSymbol);
            ValidateSymbolCommand = new RelayCommand(async param => await ExecuteValidateSymbolAsync(param));
            AddConditionCommand = new RelayCommand(ExecuteAddCondition);
            RemoveConditionCommand = new RelayCommand(ExecuteRemoveCondition, CanExecuteRemoveCondition);
            LoadSignalCommand = new RelayCommand(ExecuteLoadSignal);
            RefreshCommand = new RelayCommand(async param => await LoadSignalsAsync());
        }

        private void InitializeIndicators()
        {
            var indicators = new[]
            {
                new SignalIndicator { IndicatorType = "RSI", DisplayName = "RSI (Relative Strength Index)", Parameters = SignalIndicator.GetDefaultParameters("RSI") },
                new SignalIndicator { IndicatorType = "MACD", DisplayName = "MACD (Moving Average Convergence Divergence)", Parameters = SignalIndicator.GetDefaultParameters("MACD") },
                new SignalIndicator { IndicatorType = "VWAP", DisplayName = "VWAP (Volume Weighted Average Price)", Parameters = SignalIndicator.GetDefaultParameters("VWAP") },
                new SignalIndicator { IndicatorType = "ADX", DisplayName = "ADX (Average Directional Index)", Parameters = SignalIndicator.GetDefaultParameters("ADX") },
                new SignalIndicator { IndicatorType = "BollingerBands", DisplayName = "Bollinger Bands", Parameters = SignalIndicator.GetDefaultParameters("BollingerBands") },
                new SignalIndicator { IndicatorType = "EMA", DisplayName = "EMA (Exponential Moving Average)", Parameters = SignalIndicator.GetDefaultParameters("EMA") },
                new SignalIndicator { IndicatorType = "SMA", DisplayName = "SMA (Simple Moving Average)", Parameters = SignalIndicator.GetDefaultParameters("SMA") },
                new SignalIndicator { IndicatorType = "StochasticRSI", DisplayName = "Stochastic RSI", Parameters = SignalIndicator.GetDefaultParameters("StochasticRSI") },
                new SignalIndicator { IndicatorType = "OBV", DisplayName = "OBV (On-Balance Volume)", Parameters = SignalIndicator.GetDefaultParameters("OBV") },
                new SignalIndicator { IndicatorType = "Momentum", DisplayName = "Momentum", Parameters = SignalIndicator.GetDefaultParameters("Momentum") },
                new SignalIndicator { IndicatorType = "CCI", DisplayName = "CCI (Commodity Channel Index)", Parameters = SignalIndicator.GetDefaultParameters("CCI") },
                new SignalIndicator { IndicatorType = "ATR", DisplayName = "ATR (Average True Range)", Parameters = SignalIndicator.GetDefaultParameters("ATR") }
            };

            foreach (var indicator in indicators)
            {
                AvailableIndicators.Add(indicator);
            }
        }

        private void CreateNewSignal()
        {
            CurrentSignal = new TradingSignal
            {
                Name = string.Empty,
                Description = string.Empty,
                IsEnabled = true,
                AlertConfiguration = new SignalAlertConfiguration()
            };

            // Add a default empty symbol row
            CurrentSignal.Symbols.Add(new SignalSymbol { AllocationPercentage = 100, IsActive = true });

            IsEditing = true;
            StatusMessage = "Creating new signal...";
        }

        private void UpdateIndicatorSelections()
        {
            if (CurrentSignal == null) return;

            foreach (var indicator in AvailableIndicators)
            {
                indicator.IsSelected = CurrentSignal.Indicators.Any(i => i.IndicatorType == indicator.IndicatorType);
            }
        }

        private async Task LoadSignalsAsync()
        {
            try
            {
                IsBusy = true;
                StatusMessage = "Loading signals...";

                var signals = await _tradingSignalService.GetAllSignalsAsync();

                Signals.Clear();
                foreach (var signal in signals)
                {
                    Signals.Add(signal);
                }

                StatusMessage = $"Loaded {signals.Count} signals.";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error loading signals: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
            }
        }

        #endregion

        #region Command Implementations

        private void ExecuteNewSignal(object parameter)
        {
            CreateNewSignal();
            foreach (var indicator in AvailableIndicators)
            {
                indicator.IsSelected = false;
            }
        }

        private bool CanExecuteSaveSignal(object parameter)
        {
            return CurrentSignal != null && !string.IsNullOrWhiteSpace(CurrentSignal.Name);
        }

        private async Task ExecuteSaveSignalAsync()
        {
            if (CurrentSignal == null) return;

            try
            {
                IsBusy = true;
                StatusMessage = "Saving signal...";

                // Sync selected indicators
                CurrentSignal.Indicators.Clear();
                foreach (var indicator in AvailableIndicators.Where(i => i.IsSelected))
                {
                    CurrentSignal.Indicators.Add(new SignalIndicator
                    {
                        IndicatorType = indicator.IndicatorType,
                        DisplayName = indicator.DisplayName,
                        IsSelected = true,
                        Parameters = new System.Collections.Generic.Dictionary<string, object>(indicator.Parameters)
                    });
                }

                // Validate signal
                if (!CurrentSignal.Validate(out string errorMessage))
                {
                    StatusMessage = errorMessage;
                    MessageBox.Show(errorMessage, "Validation Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                var result = await _tradingSignalService.SaveSignalAsync(CurrentSignal);

                if (result)
                {
                    StatusMessage = "Signal saved successfully.";
                    await LoadSignalsAsync();
                }
                else
                {
                    StatusMessage = "Failed to save signal.";
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error saving signal: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
            }
        }

        private bool CanExecuteDeleteSignal(object parameter)
        {
            return CurrentSignal != null && CurrentSignal.Id > 0;
        }

        private async Task ExecuteDeleteSignalAsync()
        {
            if (CurrentSignal == null || CurrentSignal.Id == 0) return;

            var result = MessageBox.Show(
                $"Are you sure you want to delete the signal '{CurrentSignal.Name}'?",
                "Confirm Delete",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question);

            if (result != MessageBoxResult.Yes) return;

            try
            {
                IsBusy = true;
                StatusMessage = "Deleting signal...";

                var deleteResult = await _tradingSignalService.DeleteSignalAsync(CurrentSignal.Id);

                if (deleteResult)
                {
                    StatusMessage = "Signal deleted successfully.";
                    CreateNewSignal();
                    await LoadSignalsAsync();
                }
                else
                {
                    StatusMessage = "Failed to delete signal.";
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error deleting signal: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
            }
        }

        private void ExecuteAddSymbol(object parameter)
        {
            if (CurrentSignal == null) return;

            // Check for duplicates
            var newSymbol = new SignalSymbol
            {
                AllocationPercentage = 0,
                IsActive = true
            };

            CurrentSignal.Symbols.Add(newSymbol);

            // Recalculate allocations if there's only one symbol
            if (CurrentSignal.Symbols.Count == 1)
            {
                CurrentSignal.Symbols[0].AllocationPercentage = 100;
            }

            StatusMessage = "Symbol added. Enter the ticker and validate.";
        }

        private bool CanExecuteRemoveSymbol(object parameter)
        {
            return SelectedSymbol != null && CurrentSignal?.Symbols.Count > 1;
        }

        private void ExecuteRemoveSymbol(object parameter)
        {
            if (SelectedSymbol == null || CurrentSignal == null) return;

            CurrentSignal.Symbols.Remove(SelectedSymbol);
            SelectedSymbol = null;

            // Recalculate allocation for remaining symbols
            if (CurrentSignal.Symbols.Count == 1)
            {
                CurrentSignal.Symbols[0].AllocationPercentage = 100;
            }

            StatusMessage = "Symbol removed.";
        }

        private async Task ExecuteValidateSymbolAsync(object parameter)
        {
            if (parameter is not SignalSymbol symbol) return;

            if (string.IsNullOrWhiteSpace(symbol.Symbol))
            {
                symbol.IsValidated = false;
                symbol.ValidationMessage = "Enter a symbol to validate.";
                return;
            }

            try
            {
                IsBusy = true;
                StatusMessage = $"Validating symbol {symbol.Symbol}...";

                // Check for duplicates
                var duplicates = CurrentSignal?.Symbols.Count(s => 
                    s != symbol && 
                    string.Equals(s.Symbol, symbol.Symbol, StringComparison.OrdinalIgnoreCase)) ?? 0;

                if (duplicates > 0)
                {
                    symbol.IsValidated = false;
                    symbol.ValidationMessage = "Duplicate symbol detected.";
                    StatusMessage = "Duplicate symbol detected.";
                    return;
                }

                var (isValid, message) = await _tradingSignalService.ValidateSymbolAsync(symbol.Symbol);

                symbol.IsValidated = isValid;
                symbol.ValidationMessage = message;
                StatusMessage = message;
            }
            catch (Exception ex)
            {
                symbol.IsValidated = false;
                symbol.ValidationMessage = $"Validation error: {ex.Message}";
                StatusMessage = $"Error validating symbol: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
            }
        }

        private void ExecuteAddCondition(object parameter)
        {
            if (CurrentSignal == null) return;

            var newCondition = new SignalCondition
            {
                ComparisonOperator = ">",
                LogicalOperator = "AND",
                Order = CurrentSignal.Conditions.Count
            };

            CurrentSignal.Conditions.Add(newCondition);
            StatusMessage = "Condition added.";
        }

        private bool CanExecuteRemoveCondition(object parameter)
        {
            return SelectedCondition != null;
        }

        private void ExecuteRemoveCondition(object parameter)
        {
            if (SelectedCondition == null || CurrentSignal == null) return;

            CurrentSignal.Conditions.Remove(SelectedCondition);
            SelectedCondition = null;

            // Reorder remaining conditions
            int order = 0;
            foreach (var condition in CurrentSignal.Conditions)
            {
                condition.Order = order++;
            }

            StatusMessage = "Condition removed.";
        }

        private void ExecuteLoadSignal(object parameter)
        {
            if (parameter is not TradingSignal signal) return;

            CurrentSignal = signal;
            IsEditing = true;
            StatusMessage = $"Editing signal: {signal.Name}";
        }

        #endregion
    }
}
