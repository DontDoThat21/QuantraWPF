using System.Diagnostics;
using System.Text.Json;
using System.Collections.Concurrent;
using Quantra.Models;
using Quantra.Utilities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Real-time ML inference service that provides low-latency predictions
    /// for live market data processing and trading decisions.
    /// </summary>
    public class RealTimeInferenceService
    {
        private readonly string _pythonScript;
        private readonly ConcurrentDictionary<string, TaskCompletionSource<PredictionResult>> _pendingRequests;
        private readonly SemaphoreSlim _requestSemaphore;
        private readonly SemaphoreSlim _streamAccessSemaphore;
        private readonly Timer _metricsTimer;
        private readonly ConcurrentTaskThrottler _batchThrottler;
        private Process _pythonProcess;
        private bool _isInitialized;
        private bool _disposed;

        // Performance tracking
        private long _totalRequests;
        private long _successfulRequests;
        private readonly List<double> _recentInferenceTimes;
        private readonly object _metricsLock = new object();

        public RealTimeInferenceService(int maxConcurrentRequests = 10)
        {
            _pythonScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "real_time_inference.py");
            _pendingRequests = new ConcurrentDictionary<string, TaskCompletionSource<PredictionResult>>();
            _requestSemaphore = new SemaphoreSlim(maxConcurrentRequests, maxConcurrentRequests);
            _streamAccessSemaphore = new SemaphoreSlim(1, 1); // Only one stream operation at a time
            _batchThrottler = new ConcurrentTaskThrottler(maxConcurrentRequests); // Use same limit for batch operations
            _recentInferenceTimes = new List<double>();

            // Start metrics collection timer
            _metricsTimer = new Timer(CollectMetrics, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
        }

        /// <summary>
        /// Initialize the real-time inference service by starting the Python pipeline.
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            if (_isInitialized) return true;

            try
            {
                if (!File.Exists(_pythonScript))
                {
                    throw new FileNotFoundException($"Real-time inference script not found at: {_pythonScript}");
                }

                // Start the Python process in interactive mode
                var psi = new ProcessStartInfo
                {
                    FileName = "python3",
                    Arguments = $"\"{_pythonScript}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    RedirectStandardInput = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(_pythonScript)
                };

                _pythonProcess = Process.Start(psi);
                if (_pythonProcess == null)
                {
                    throw new Exception("Failed to start Python inference process");
                }

                // Send initialization command
                var initCommand = new
                {
                    command = "initialize",
                    config = new
                    {
                        model_types = new[] { "auto" },
                        max_queue_size = 1000,
                        prediction_timeout = 0.1,
                        enable_monitoring = true
                    }
                };

                await _streamAccessSemaphore.WaitAsync().ConfigureAwait(false);
                try
                {
                    await _pythonProcess.StandardInput.WriteLineAsync(JsonSerializer.Serialize(initCommand)).ConfigureAwait(false);
                    await _pythonProcess.StandardInput.FlushAsync().ConfigureAwait(false);
                }
                finally
                {
                    _streamAccessSemaphore.Release();
                }

                _isInitialized = true;
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to initialize real-time inference service: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Get a real-time prediction for the given market data.
        /// This method provides low-latency inference suitable for live trading.
        /// </summary>
        public async Task<PredictionResult> GetPredictionAsync(Dictionary<string, double> marketData,
                                                              string modelType = "auto",
                                                              CancellationToken cancellationToken = default)
        {
            if (!_isInitialized)
            {
                var initialized = await InitializeAsync().ConfigureAwait(false);
                if (!initialized)
                {
                    throw new InvalidOperationException("Failed to initialize inference service");
                }
            }

            // Throttle concurrent requests
            await _requestSemaphore.WaitAsync(cancellationToken).ConfigureAwait(false);

            try
            {
                return await GetPredictionInternalAsync(marketData, modelType, cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                _requestSemaphore.Release();
            }
        }

        private async Task<PredictionResult> GetPredictionInternalAsync(Dictionary<string, double> marketData,
                                                                       string modelType,
                                                                       CancellationToken cancellationToken)
        {
            var requestId = Guid.NewGuid().ToString();
            var stopwatch = Stopwatch.StartNew();

            Interlocked.Increment(ref _totalRequests);

            try
            {
                var predictionRequest = new
                {
                    command = "predict_sync",
                    request_id = requestId,
                    market_data = marketData,
                    model_type = modelType,
                    timeout = 5.0
                };

                var json = JsonSerializer.Serialize(predictionRequest);

                // Create task completion source for this request
                var tcs = new TaskCompletionSource<PredictionResult>();
                _pendingRequests[requestId] = tcs;

                try
                {
                    // Ensure Python process is alive before attempting to write
                    if (!await EnsurePythonProcessAliveAsync(cancellationToken).ConfigureAwait(false))
                    {
                        throw new InvalidOperationException("Failed to ensure Python process is running");
                    }

                    // Send request to Python process
                    await _streamAccessSemaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
                    try
                    {
                        await _pythonProcess.StandardInput.WriteLineAsync(json).ConfigureAwait(false);
                        await _pythonProcess.StandardInput.FlushAsync().ConfigureAwait(false);
                    }
                    finally
                    {
                        _streamAccessSemaphore.Release();
                    }

                    // Wait for response with timeout
                    using (var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
                    {
                        timeoutCts.CancelAfter(TimeSpan.FromSeconds(10));

                        // Start reading response concurrently without Task.Run
                        _ = ReadResponseAsync(timeoutCts.Token);

                        var result = await tcs.Task.WaitAsync(timeoutCts.Token).ConfigureAwait(false);

                        stopwatch.Stop();
                        var inferenceTime = stopwatch.Elapsed.TotalMilliseconds;

                        // Track performance metrics
                        lock (_metricsLock)
                        {
                            _recentInferenceTimes.Add(inferenceTime);
                            if (_recentInferenceTimes.Count > 100)
                            {
                                _recentInferenceTimes.RemoveAt(0);
                            }
                        }

                        Interlocked.Increment(ref _successfulRequests);

                        // Add timing information
                        result.InferenceTimeMs = inferenceTime;
                        result.RequestId = requestId;

                        return result;
                    }
                }
                finally
                {
                    _pendingRequests.TryRemove(requestId, out _);
                }
            }
            catch (IOException ioEx) when (ioEx.Message.Contains("pipe") || ioEx.Message.Contains("closed"))
            {
                stopwatch.Stop();
                Console.WriteLine($"Pipe closed exception: {ioEx.Message}. Will restart Python process on next request.");

                // Mark process as dead so it gets restarted on next request
                _isInitialized = false;

                // Return safe fallback prediction
                return new PredictionResult
                {
                    Symbol = marketData.ContainsKey("symbol") ? marketData["symbol"].ToString() : "UNKNOWN",
                    Action = "HOLD",
                    Confidence = 0.5,
                    CurrentPrice = marketData.ContainsKey("close") ? marketData["close"] : 0,
                    TargetPrice = marketData.ContainsKey("close") ? marketData["close"] : 0,
                    PredictionDate = DateTime.Now,
                    RequestId = requestId,
                    InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                    Error = "Python process pipe closed - service will restart automatically"
                };
            }
            catch (Exception ex)
            {
                stopwatch.Stop();

                // Return safe fallback prediction
                return new PredictionResult
                {
                    Symbol = marketData.ContainsKey("symbol") ? marketData["symbol"].ToString() : "UNKNOWN",
                    Action = "HOLD",
                    Confidence = 0.5,
                    CurrentPrice = marketData.ContainsKey("close") ? marketData["close"] : 0,
                    TargetPrice = marketData.ContainsKey("close") ? marketData["close"] : 0,
                    PredictionDate = DateTime.Now,
                    RequestId = requestId,
                    InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                    Error = ex.Message
                };
            }
        }

        /// <summary>
        /// Ensures the Python process is alive and running. Restarts it if necessary.
        /// </summary>
        private async Task<bool> EnsurePythonProcessAliveAsync(CancellationToken cancellationToken = default)
        {
            // Check if process is still running
            if (_pythonProcess != null && !_pythonProcess.HasExited)
            {
                return true;
            }

            // Process has exited or was never started, need to restart
            Console.WriteLine("Python process has exited, restarting...");

            // Clean up existing process
            try
            {
                _pythonProcess?.Kill();
                _pythonProcess?.Dispose();
            }
            catch { }

            _pythonProcess = null;
            _isInitialized = false;

            // Clear pending requests as they will never be fulfilled
            foreach (var kvp in _pendingRequests)
            {
                kvp.Value.TrySetException(new InvalidOperationException("Python process restarted"));
            }
            _pendingRequests.Clear();

            // Restart the process
            return await InitializeAsync().ConfigureAwait(false);
        }

        private async Task ReadResponseAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                // Check if process is still alive before attempting to read
                if (_pythonProcess?.HasExited == true)
                {
                    Console.WriteLine("Python process has exited, cannot read response");
                    return;
                }

                await _streamAccessSemaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
                try
                {
                    var response = await _pythonProcess.StandardOutput.ReadLineAsync().ConfigureAwait(false);
                    if (!string.IsNullOrWhiteSpace(response))
                    {
                        var pythonResult = JsonSerializer.Deserialize<PythonInferenceResult>(response);

                        if (_pendingRequests.TryGetValue(pythonResult.RequestId, out var tcs))
                        {
                            var predictionResult = ConvertToPredictionResult(pythonResult);
                            tcs.SetResult(predictionResult);
                        }
                    }
                }
                finally
                {
                    _streamAccessSemaphore.Release();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading inference response: {ex.Message}");
            }
        }

        private PredictionResult ConvertToPredictionResult(PythonInferenceResult pythonResult)
        {
            return new PredictionResult
            {
                Symbol = pythonResult.Symbol ?? "UNKNOWN",
                Action = pythonResult.Action ?? "HOLD",
                Confidence = pythonResult.Confidence,
                CurrentPrice = pythonResult.CurrentPrice,
                TargetPrice = pythonResult.PredictedPrice,
                PredictionDate = DateTime.Now,
                RequestId = pythonResult.RequestId,
                InferenceTimeMs = pythonResult.InferenceTimeMs,

                // Additional real-time specific data
                RiskScore = pythonResult.Risk?.RiskLevel ?? 0.5,
                ValueAtRisk = pythonResult.Risk?.ValueAtRisk ?? 0.0,
                MaxDrawdown = pythonResult.Risk?.MaxDrawdown ?? 0.0,
                SharpeRatio = pythonResult.Risk?.SharpeRatio ?? 0.0
            };
        }

        /// <summary>
        /// Get current performance metrics for the inference service.
        /// </summary>
        public InferenceMetrics GetPerformanceMetrics()
        {
            lock (_metricsLock)
            {
                var totalRequests = Interlocked.Read(ref _totalRequests);
                var successfulRequests = Interlocked.Read(ref _successfulRequests);

                return new InferenceMetrics
                {
                    TotalRequests = totalRequests,
                    SuccessfulRequests = successfulRequests,
                    ErrorRate = totalRequests > 0 ? (totalRequests - successfulRequests) / (double)totalRequests : 0,
                    AverageInferenceTimeMs = _recentInferenceTimes.Count > 0 ? _recentInferenceTimes.Average() : 0,
                    P95InferenceTimeMs = _recentInferenceTimes.Count > 0 ?
                        _recentInferenceTimes.OrderBy(x => x).Skip((int)(_recentInferenceTimes.Count * 0.95)).FirstOrDefault() : 0,
                    RequestsPerMinute = CalculateRequestsPerMinute(),
                    IsHealthy = _isInitialized && _pythonProcess != null && !_pythonProcess.HasExited
                };
            }
        }

        private double CalculateRequestsPerMinute()
        {
            // This is a simplified calculation - in a real implementation,
            // you would track request timestamps and calculate the rate over the last minute
            return _recentInferenceTimes.Count;
        }

        private void CollectMetrics(object state)
        {
            try
            {
                var metrics = GetPerformanceMetrics();

                // Log metrics for monitoring
                Console.WriteLine($"Inference Metrics - Requests: {metrics.TotalRequests}, " +
                                 $"Success Rate: {1 - metrics.ErrorRate:P2}, " +
                                 $"Avg Time: {metrics.AverageInferenceTimeMs:F2}ms");

                // Here you could send metrics to monitoring systems like Application Insights,
                // Prometheus, or custom logging systems
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error collecting metrics: {ex.Message}");
            }
        }

        /// <summary>
        /// Batch prediction for multiple market data points (useful for bulk processing).
        /// </summary>
        public async Task<List<PredictionResult>> GetBatchPredictionsAsync(
            List<Dictionary<string, double>> marketDataBatch,
            string modelType = "auto",
            CancellationToken cancellationToken = default)
        {
            var taskFactories = marketDataBatch.Select(data =>
                new Func<Task<PredictionResult>>(() => GetPredictionAsync(data, modelType, cancellationToken)));
            var results = await _batchThrottler.ExecuteThrottledAsync(taskFactories, cancellationToken).ConfigureAwait(false);
            return results.ToList();
        }

        /// <summary>
        /// Health check for the inference service.
        /// </summary>
        public async Task<bool> HealthCheckAsync()
        {
            try
            {
                if (!_isInitialized || _pythonProcess?.HasExited == true)
                {
                    return false;
                }

                // Perform a simple prediction to verify the service is working
                var testData = new Dictionary<string, double>
                {
                    ["close"] = 100.0,
                    ["volume"] = 1000000
                };

                var result = await GetPredictionAsync(testData, cancellationToken: new CancellationTokenSource(5000).Token).ConfigureAwait(false);
                return result != null && string.IsNullOrEmpty(result.Error);
            }
            catch
            {
                return false;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            _metricsTimer?.Dispose();
            _requestSemaphore?.Dispose();
            _streamAccessSemaphore?.Dispose();
            _batchThrottler?.Dispose();

            try
            {
                _pythonProcess?.Kill();
                _pythonProcess?.Dispose();
            }
            catch { }

            _disposed = true;
        }
    }

    /// <summary>
    /// Performance metrics for the real-time inference service.
    /// </summary>
    public class InferenceMetrics
    {
        public long TotalRequests { get; set; }
        public long SuccessfulRequests { get; set; }
        public double ErrorRate { get; set; }
        public double AverageInferenceTimeMs { get; set; }
        public double P95InferenceTimeMs { get; set; }
        public double RequestsPerMinute { get; set; }
        public bool IsHealthy { get; set; }
    }

    /// <summary>
    /// Python inference result structure for deserialization.
    /// </summary>
    public class PythonInferenceResult
    {
        public string RequestId { get; set; }
        public string Symbol { get; set; }
        public string Action { get; set; }
        public double Confidence { get; set; }
        public double CurrentPrice { get; set; }
        public double PredictedPrice { get; set; }
        public double InferenceTimeMs { get; set; }
        public string Error { get; set; }
        public RiskMetrics Risk { get; set; }
    }

    public class RiskMetrics
    {
        public double RiskLevel { get; set; }
        public double ValueAtRisk { get; set; }
        public double MaxDrawdown { get; set; }
        public double SharpeRatio { get; set; }
    }
}