using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using QuantraLogging = Quantra.CrossCutting.Logging;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Background service that runs overnight batch predictions on a schedule.
    /// Designed to generate predictions for 12,000+ symbols during off-hours.
    /// </summary>
    public class ScheduledPredictionService
    {
        private readonly ILogger<ScheduledPredictionService> _logger;
        private readonly BatchPredictionService _batchPredictionService;
        private readonly QuantraLogging.ILogger _loggingService;
        private Timer _timer;
        private CancellationTokenSource _cancellationTokenSource;

        // Configuration - Schedule for overnight execution (e.g., 2 AM)
        private readonly TimeSpan _scheduledTime = new TimeSpan(2, 0, 0); // 2:00 AM
        private readonly bool _enableScheduledPredictions = true; // Can be configured via settings

        public ScheduledPredictionService(
            ILogger<ScheduledPredictionService> logger,
            BatchPredictionService batchPredictionService,
            QuantraLogging.ILogger loggingService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _batchPredictionService = batchPredictionService ?? throw new ArgumentNullException(nameof(batchPredictionService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        public async Task StartAsync(CancellationToken cancellationToken = default)
        {
            if (!_enableScheduledPredictions)
            {
                _logger?.LogInformation("Scheduled predictions are disabled");
                return;
            }

            _logger?.LogInformation("Scheduled Prediction Service started");
            _loggingService?.Information("Scheduled Prediction Service started");

            while (!cancellationToken.IsCancellationRequested)
            {
                var now = DateTime.Now;
                var nextRun = CalculateNextRunTime(now);
                var delay = nextRun - now;

                _logger?.LogInformation("Next prediction batch scheduled for {NextRun} (in {Delay})", 
                    nextRun, delay);

                try
                {
                    // Wait until scheduled time
                    await Task.Delay(delay, cancellationToken);

                    if (!cancellationToken.IsCancellationRequested)
                    {
                        await RunScheduledPredictionBatchAsync(cancellationToken);
                    }
                }
                catch (OperationCanceledException)
                {
                    _logger?.LogInformation("Scheduled prediction service is stopping");
                    break;
                }
                catch (Exception ex)
                {
                    _logger?.LogError(ex, "Error in scheduled prediction service");
                    _loggingService?.Error(ex, "Error in scheduled prediction service");
                    
                    // Wait a bit before retrying after an error
                    await Task.Delay(TimeSpan.FromMinutes(5), cancellationToken);
                }
            }

            _logger?.LogInformation("Scheduled Prediction Service stopped");
        }

        /// <summary>
        /// Runs the scheduled prediction batch.
        /// </summary>
        private async Task RunScheduledPredictionBatchAsync(CancellationToken cancellationToken)
        {
            _logger?.LogInformation("Starting scheduled prediction batch");
            _loggingService?.Information("Starting scheduled overnight prediction batch");

            var progress = new Progress<BatchPredictionProgress>(p =>
            {
                // Log progress every so often
                if (p.ProcessedSymbols % 100 == 0)
                {
                    _logger?.LogInformation(
                        "Batch progress: {Processed}/{Total} ({Percent:F1}%) - {Success} successful, {Failed} failed - ETA: {ETA}",
                        p.ProcessedSymbols,
                        p.TotalSymbols,
                        p.ProgressPercentage,
                        p.SuccessfulPredictions,
                        p.FailedPredictions,
                        p.EstimatedTimeRemaining.ToString(@"hh\:mm\:ss"));
                }
            });

            var options = new BatchPredictionOptions
            {
                SkipRecentPredictions = true,
                PredictionAgeThresholdHours = 12, // Only regenerate if older than 12 hours
                Timeframe = "1day",
                ModelType = "auto"
            };

            try
            {
                var result = await _batchPredictionService.GenerateOvernightPredictionsAsync(
                    options, progress, cancellationToken);

                if (result.Success)
                {
                    _logger?.LogInformation(
                        "Batch prediction completed successfully: {Success}/{Total} ({Rate:F1}%) in {Duration} - {Speed:F2} predictions/sec",
                        result.SuccessfulPredictions,
                        result.TotalSymbols,
                        result.SuccessRate,
                        result.Duration.ToString(@"hh\:mm\:ss"),
                        result.PredictionsPerSecond);

                    _loggingService?.Information(
                        "Overnight predictions complete: {Success}/{Total} successful", 
                        result.SuccessfulPredictions, result.TotalSymbols);
                }
                else
                {
                    _logger?.LogWarning("Batch prediction completed with errors: {Message}", result.Message);
                    _loggingService?.Warning("Overnight predictions had errors: {Message}", result.Message);
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error running scheduled prediction batch");
                _loggingService?.Error(ex, "Error running scheduled prediction batch");
            }
        }

        /// <summary>
        /// Calculates the next run time based on the scheduled time.
        /// </summary>
        private DateTime CalculateNextRunTime(DateTime current)
        {
            var scheduledToday = current.Date + _scheduledTime;
            
            if (current < scheduledToday)
            {
                // Scheduled time hasn't occurred today yet
                return scheduledToday;
            }
            else
            {
                // Scheduled time already passed today, schedule for tomorrow
                return scheduledToday.AddDays(1);
            }
        }

        public Task StopAsync(CancellationToken cancellationToken = default)
        {
            _logger?.LogInformation("Scheduled Prediction Service is stopping");
            _cancellationTokenSource?.Cancel();
            _timer?.Dispose();
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            _timer?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }
}
