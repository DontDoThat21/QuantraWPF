using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Threading;
using Quantra.Commands;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.ViewModels.Base;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Alerts Control
    /// </summary>
    public class AlertsControlViewModel : ViewModelBase, IDisposable
    {
        private const int MaxSymbolsToCheck = 50;
        
        private readonly ITechnicalIndicatorService _indicatorService;
        private readonly IHistoricalDataService _historicalDataService;
        private readonly SettingsService _settingsService;
        private readonly StockDataCacheService _stockDataCacheService;
        
        private readonly TechnicalIndicatorAlertService _technicalIndicatorAlertService;
        private readonly VolumeAlertService _volumeAlertService;
        private readonly PatternAlertService _patternAlertService;
        private readonly DispatcherTimer _alertMonitoringTimer;
        
        private string _selectedCategoryFilter;
        private AlertModel _currentEditAlert;
        private bool _isMonitoring;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public AlertsControlViewModel(
            ITechnicalIndicatorService indicatorService,
            IHistoricalDataService historicalDataService,
            SettingsService settingsService,
            StockDataCacheService stockDataCacheService)
        {
            _indicatorService = indicatorService ?? throw new ArgumentNullException(nameof(indicatorService));
            _historicalDataService = historicalDataService ?? throw new ArgumentNullException(nameof(historicalDataService));
            _settingsService = settingsService ?? throw new ArgumentNullException(nameof(settingsService));
            _stockDataCacheService = stockDataCacheService ?? throw new ArgumentNullException(nameof(stockDataCacheService));
            
            // Initialize services
            _technicalIndicatorAlertService = new TechnicalIndicatorAlertService(_indicatorService);
            _volumeAlertService = new VolumeAlertService(_historicalDataService, _indicatorService);
            
            var patternRecognitionService = new PricePatternRecognitionService(_stockDataCacheService);
            _patternAlertService = new PatternAlertService(patternRecognitionService);
            
            // Initialize collections
            Alerts = new ObservableCollection<AlertModel>();
            Categories = new ObservableCollection<string>
            {
                "All Categories",
                "Price Alert",
                "Technical Indicator",
                "Volume Spike",
                "Pattern Recognition"
            };
            
            _selectedCategoryFilter = "All Categories";
            
            // Initialize alert monitoring timer
            _alertMonitoringTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(30)
            };
            _alertMonitoringTimer.Tick += async (s, e) => await CheckTechnicalIndicatorAlertsAsync();
            
            InitializeCommands();
        }

        #region Properties

        /// <summary>
        /// Collection of alerts
        /// </summary>
        public ObservableCollection<AlertModel> Alerts { get; }

        /// <summary>
        /// Available alert categories
        /// </summary>
        public ObservableCollection<string> Categories { get; }

        /// <summary>
        /// Selected category filter
        /// </summary>
        public string SelectedCategoryFilter
        {
            get => _selectedCategoryFilter;
            set => SetProperty(ref _selectedCategoryFilter, value);
        }

        /// <summary>
        /// Currently edited alert
        /// </summary>
        public AlertModel CurrentEditAlert
        {
            get => _currentEditAlert;
            set => SetProperty(ref _currentEditAlert, value);
        }

        /// <summary>
        /// Indicates if alert monitoring is active
        /// </summary>
        public bool IsMonitoring
        {
            get => _isMonitoring;
            private set => SetProperty(ref _isMonitoring, value);
        }

        #endregion

        #region Commands

        public ICommand AddAlertCommand { get; private set; }
        public ICommand EditAlertCommand { get; private set; }
        public ICommand DeleteAlertCommand { get; private set; }
        public ICommand StartMonitoringCommand { get; private set; }
        public ICommand StopMonitoringCommand { get; private set; }

        #endregion

        #region Events

        /// <summary>
        /// Event fired when an alert is triggered
        /// </summary>
        public event EventHandler<AlertModel> AlertTriggered;

        #endregion

        #region Public Methods

        /// <summary>
        /// Start alert monitoring
        /// </summary>
        public void StartMonitoring()
        {
            if (!_alertMonitoringTimer.IsEnabled)
            {
                _alertMonitoringTimer.Start();
                IsMonitoring = true;
            }
        }

        /// <summary>
        /// Stop alert monitoring
        /// </summary>
        public void StopMonitoring()
        {
            if (_alertMonitoringTimer.IsEnabled)
            {
                _alertMonitoringTimer.Stop();
                IsMonitoring = false;
            }
        }

        /// <summary>
        /// Handle global alert emission
        /// </summary>
        public void OnGlobalAlertEmitted(AlertModel alert)
        {
            if (alert != null && !Alerts.Contains(alert))
            {
                Alerts.Add(alert);
                AlertTriggered?.Invoke(this, alert);
            }
        }

        #endregion

        #region Private Methods

        private void InitializeCommands()
        {
            AddAlertCommand = new RelayCommand(ExecuteAddAlert);
            EditAlertCommand = new RelayCommand(ExecuteEditAlert, CanExecuteEditAlert);
            DeleteAlertCommand = new RelayCommand(ExecuteDeleteAlert, CanExecuteDeleteAlert);
            StartMonitoringCommand = new RelayCommand(_ => StartMonitoring(), _ => !IsMonitoring);
            StopMonitoringCommand = new RelayCommand(_ => StopMonitoring(), _ => IsMonitoring);
        }

        private async Task CheckTechnicalIndicatorAlertsAsync()
        {
            try
            {
                // Only check active, non-triggered technical indicator alerts
                var indicatorAlerts = Alerts
                    .Where(a => a.Category == AlertCategory.TechnicalIndicator && a.IsActive && !a.IsTriggered)
                    .ToList();

                if (indicatorAlerts.Count > 0)
                {
                    int triggeredCount = await _technicalIndicatorAlertService.CheckAllAlertsAsync(indicatorAlerts);
                    int volumeTriggeredCount = await _volumeAlertService.CheckAllVolumeAlertsAsync(indicatorAlerts);
                    
                    // Fire events for triggered alerts
                    foreach (var alert in indicatorAlerts.Where(a => a.IsTriggered))
                    {
                        AlertTriggered?.Invoke(this, alert);
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error checking technical indicator alerts: {ex.Message}");
            }
        }

        #endregion

        #region Command Implementations

        private void ExecuteAddAlert(object parameter)
        {
            var newAlert = new AlertModel
            {
                Id = Guid.NewGuid().ToString(),
                CreatedDate = DateTime.Now,
                IsActive = true,
                IsTriggered = false
            };
            
            CurrentEditAlert = newAlert;
        }

        private bool CanExecuteEditAlert(object parameter)
        {
            return parameter is AlertModel;
        }

        private void ExecuteEditAlert(object parameter)
        {
            if (parameter is AlertModel alert)
            {
                CurrentEditAlert = alert;
            }
        }

        private bool CanExecuteDeleteAlert(object parameter)
        {
            return parameter is AlertModel;
        }

        private void ExecuteDeleteAlert(object parameter)
        {
            if (parameter is AlertModel alert)
            {
                Alerts.Remove(alert);
            }
        }

        #endregion

        #region IDisposable

        private bool _disposed = false;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _alertMonitoringTimer?.Stop();
                }
                _disposed = true;
            }
        }

        #endregion
    }
}
