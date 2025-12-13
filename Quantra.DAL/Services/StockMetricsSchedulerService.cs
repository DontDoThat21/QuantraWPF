using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace Quantra.DAL.Services
{
    public class StockMetricsSchedulerService : IDisposable
    {
        private readonly StockMetricsCalculationService _metricsCalculationService;
        private readonly LoggingService _loggingService;
        private DispatcherTimer _scheduledTimer;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isRunning;
        private DateTime _lastRunTime;

        public bool IsRunning => _isRunning;
        public DateTime LastRunTime => _lastRunTime;
        public TimeSpan UpdateInterval { get; private set; }

        public event EventHandler<MetricsCalculationProgressEventArgs> ProgressChanged;

        public StockMetricsSchedulerService(
            StockMetricsCalculationService metricsCalculationService,
            LoggingService loggingService)
        {
            _metricsCalculationService = metricsCalculationService ?? throw new ArgumentNullException(nameof(metricsCalculationService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
            
            UpdateInterval = TimeSpan.FromHours(4);
            
            _metricsCalculationService.ProgressChanged += OnMetricsProgressChanged;
        }

        public void Start(TimeSpan? interval = null)
        {
            if (_isRunning)
            {
                _loggingService.Log("Warning", "Metrics scheduler is already running");
                return;
            }

            if (interval.HasValue)
            {
                UpdateInterval = interval.Value;
            }

            _scheduledTimer = new DispatcherTimer
            {
                Interval = UpdateInterval
            };
            _scheduledTimer.Tick += OnScheduledTick;
            _scheduledTimer.Start();

            _isRunning = true;
            _loggingService.Log("Info", $"Metrics scheduler started with interval: {UpdateInterval}");

            Task.Run(async () => await RunMetricsCalculationAsync());
        }

        public void Stop()
        {
            if (!_isRunning)
            {
                return;
            }

            _scheduledTimer?.Stop();
            _cancellationTokenSource?.Cancel();
            _isRunning = false;

            _loggingService.Log("Info", "Metrics scheduler stopped");
        }

        public async Task RunMetricsCalculationAsync()
        {
            if (_cancellationTokenSource != null && !_cancellationTokenSource.IsCancellationRequested)
            {
                _cancellationTokenSource.Cancel();
            }

            _cancellationTokenSource = new CancellationTokenSource();

            try
            {
                _loggingService.Log("Info", "Starting scheduled metrics calculation...");
                _lastRunTime = DateTime.Now;

                await _metricsCalculationService.CalculateAllMetricsAsync(_cancellationTokenSource.Token);

                _loggingService.Log("Info", "Scheduled metrics calculation completed successfully");
            }
            catch (OperationCanceledException)
            {
                _loggingService.Log("Info", "Metrics calculation was cancelled");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error during scheduled metrics calculation", ex.ToString());
            }
        }

        private async void OnScheduledTick(object sender, EventArgs e)
        {
            await RunMetricsCalculationAsync();
        }

        private void OnMetricsProgressChanged(object sender, MetricsCalculationProgressEventArgs e)
        {
            ProgressChanged?.Invoke(this, e);
        }

        public void Dispose()
        {
            Stop();
            _scheduledTimer = null;
            _cancellationTokenSource?.Dispose();
            
            if (_metricsCalculationService != null)
            {
                _metricsCalculationService.ProgressChanged -= OnMetricsProgressChanged;
            }
        }
    }
}
