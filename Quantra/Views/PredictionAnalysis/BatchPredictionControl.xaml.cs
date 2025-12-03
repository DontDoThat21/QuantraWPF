using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using Quantra.DAL.Services;

namespace Quantra.Controls
{
    /// <summary>
    /// User control for managing overnight batch predictions
    /// </summary>
    public partial class BatchPredictionControl : UserControl
    {
        private readonly BatchPredictionService _batchPredictionService;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isRunning;

        public BatchPredictionControl()
        {
            InitializeComponent();
            
            // Get service from DI container
            _batchPredictionService = App.ServiceProvider?.GetService(typeof(BatchPredictionService)) as BatchPredictionService;
            
            if (_batchPredictionService == null && StatusText != null && StartButton != null)
            {
                StatusText.Text = "Error: BatchPredictionService not available";
                StatusText.Foreground = Brushes.Red;
                StartButton.IsEnabled = false;
            }
        }

        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            if (_isRunning)
            {
                // Cancel running operation
                _cancellationTokenSource?.Cancel();
                return;
            }

            await StartBatchPredictionAsync();
        }

        private async Task StartBatchPredictionAsync()
        {
            if (_batchPredictionService == null) return;

            _isRunning = true;
            _cancellationTokenSource = new CancellationTokenSource();
            
            // Update UI
            StartButton.Content = "? Cancel";
            StatusText.Text = "Initializing batch prediction...";
            StatusText.Foreground = Brushes.Yellow;
            ProgressBar.Value = 0;
            ProgressBar.IsIndeterminate = true;

            // Create options from UI
            var options = new BatchPredictionOptions
            {
                SkipRecentPredictions = SkipRecentCheckBox.IsChecked ?? true,
                PredictionAgeThresholdHours = 24,
                Timeframe = (TimeframeComboBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "1day",
                ModelType = (ModelTypeComboBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "auto"
            };

            // Parse max symbols if specified
            if (int.TryParse(MaxSymbolsTextBox.Text, out int maxSymbols) && maxSymbols > 0)
            {
                options.MaxSymbolsToProcess = maxSymbols;
            }

            // Filter by sector if selected
            if (SectorComboBox.SelectedIndex > 0)
            {
                options.SectorFilter = (SectorComboBox.SelectedItem as ComboBoxItem)?.Content as string;
            }

            // Create progress reporter
            var progress = new Progress<BatchPredictionProgress>(p =>
            {
                Dispatcher.InvokeAsync(() =>
                {
                    ProgressBar.IsIndeterminate = false;
                    ProgressBar.Value = p.ProgressPercentage;
                    
                    TotalSymbolsText.Text = $"Total: {p.TotalSymbols}";
                    ProcessedText.Text = $"Processed: {p.ProcessedSymbols}";
                    SuccessText.Text = $"Success: {p.SuccessfulPredictions}";
                    FailedText.Text = $"Failed: {p.FailedPredictions}";
                    SuccessRateText.Text = $"Success Rate: {p.SuccessRate:F1}%";
                    
                    CurrentSymbolText.Text = $"Current: {p.CurrentSymbol}";
                    ElapsedTimeText.Text = $"Elapsed: {p.ElapsedTime:hh\\:mm\\:ss}";
                    EstimatedTimeText.Text = $"ETA: {p.EstimatedTimeRemaining:hh\\:mm\\:ss}";
                    
                    StatusText.Text = $"Processing... {p.ProgressPercentage:F1}%";
                    StatusText.Foreground = Brushes.Cyan;
                });
            });

            try
            {
                var result = await _batchPredictionService.GenerateOvernightPredictionsAsync(
                    options, progress, _cancellationTokenSource.Token);

                // Update final status
                await Dispatcher.InvokeAsync(() =>
                {
                    ProgressBar.IsIndeterminate = false;
                    
                    if (result.Success)
                    {
                        StatusText.Text = $"? Complete: {result.SuccessfulPredictions}/{result.TotalSymbols} successful in {result.Duration:hh\\:mm\\:ss}";
                        StatusText.Foreground = Brushes.LimeGreen;
                        ProgressBar.Value = 100;
                    }
                    else
                    {
                        StatusText.Text = $"? {result.Message}";
                        StatusText.Foreground = Brushes.Orange;
                    }

                    // Update metrics
                    SpeedText.Text = $"Speed: {result.PredictionsPerSecond:F2} pred/sec";
                });
            }
            catch (OperationCanceledException)
            {
                await Dispatcher.InvokeAsync(() =>
                {
                    StatusText.Text = "? Cancelled by user";
                    StatusText.Foreground = Brushes.Orange;
                });
            }
            catch (Exception ex)
            {
                await Dispatcher.InvokeAsync(() =>
                {
                    StatusText.Text = $"? Error: {ex.Message}";
                    StatusText.Foreground = Brushes.Red;
                });
            }
            finally
            {
                _isRunning = false;
                await Dispatcher.InvokeAsync(() =>
                {
                    StartButton.Content = "? Start Batch Prediction";
                    ProgressBar.IsIndeterminate = false;
                });
                
                _cancellationTokenSource?.Dispose();
                _cancellationTokenSource = null;
            }
        }

        private void SkipRecentCheckBox_Checked(object sender, RoutedEventArgs e)
        {
            // Could enable/disable age threshold UI here
        }
    }
}
