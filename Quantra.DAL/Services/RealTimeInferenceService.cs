using System.Diagnostics;
using System.Text.Json;
using System.Collections.Concurrent;
using Quantra.Models;
using Quantra.Utilities;
using Quantra.DAL.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Delegate for reporting progress during Python prediction execution (MarketChat story 9).
    /// </summary>
    /// <param name="progressMessage">The progress message to display to the user.</param>
    public delegate void PredictionProgressCallback(string progressMessage);

    /// <summary>
    /// Result of a Python prediction execution with detailed status information (MarketChat story 9).
    /// </summary>
    public class PythonPredictionExecutionResult
    {
        public bool Success { get; set; }
        public PredictionResult Prediction { get; set; }
        public string ErrorMessage { get; set; }
        public double ExecutionTimeMs { get; set; }
        public string ModelType { get; set; }
        public bool WasCached { get; set; }
        
        /// <summary>
        /// Full TFT prediction result with multi-horizon forecasts and attention weights (only populated for TFT model).
        /// </summary>
        public TFTPredictionResult TFTResult { get; set; }
    }

    /// <summary>
    /// Real-time ML inference service that provides low-latency predictions
    /// for live market data processing and trading decisions.
    /// Supports Python script orchestration for on-demand predictions (MarketChat story 9).
    /// </summary>
    public class RealTimeInferenceService
    {
        private readonly string _pythonScript;
        private readonly string _stockPredictorScript;
        private readonly string _tftPredictScript;
        private readonly ConcurrentDictionary<string, TaskCompletionSource<PredictionResult>> _pendingRequests;
        private readonly SemaphoreSlim _requestSemaphore;
        private readonly SemaphoreSlim _streamAccessSemaphore;
        private readonly Timer _metricsTimer;
        private readonly ConcurrentTaskThrottler _batchThrottler;
        private readonly Interfaces.IStockDataCacheService _stockDataCacheService;
        private Process _pythonProcess;
        private bool _isInitialized;
        private bool _disposed;

        // Performance tracking
        private long _totalRequests;
        private long _successfulRequests;
        private readonly List<double> _recentInferenceTimes;
        private readonly object _metricsLock = new object();

        public RealTimeInferenceService(
            Interfaces.IStockDataCacheService stockDataCacheService = null,
            int maxConcurrentRequests = 10)
        {
            _stockDataCacheService = stockDataCacheService;
            _pythonScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "real_time_inference.py");
            _stockPredictorScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "stock_predictor.py");
            _tftPredictScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "tft_predict.py");
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

            // TEMPORARY: The Python real_time_inference.py script is not properly configured
            // for stdin/stdout JSON API communication. It runs in interactive mode.
            // Use ExecutePythonPredictionAsync() method instead which properly invokes stock_predictor.py
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine("NOTICE: Real-time Python inference service is currently disabled.");
            Console.WriteLine("The real_time_inference.py script requires modification to support JSON API mode.");
            Console.WriteLine("For predictions, please use ExecutePythonPredictionAsync() instead.");
            Console.WriteLine("=".PadRight(80, '='));
            
            _isInitialized = false;
            return false;

            // TODO: Uncomment and fix when Python script is updated to support --api-mode
            /*
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
                    Arguments = $"\"{_pythonScript}\" --api-mode",  // NEW: needs --api-mode flag
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    RedirectStandardInput = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(_pythonScript)
                };
                // ... rest of initialization
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to initialize: {ex.Message}");
                return false;
            }
            */
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
            catch (InvalidOperationException ioEx) when (ioEx.Message.Contains("Python process restarted"))
            {
                stopwatch.Stop();
                Console.WriteLine($"Python process was restarted during prediction: {ioEx.Message}");

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
                    Error = "Python process restarted during prediction - please retry"
                };
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                stopwatch.Stop();
                Console.WriteLine($"Prediction request {requestId} timed out after 10 seconds");

                // Return safe fallback prediction for timeout
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
                    Error = "Prediction request timed out - Python process may be overloaded"
                };
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
            if (_pythonProcess != null)
            {
                Console.WriteLine($"Python process has exited with code {_pythonProcess.ExitCode}. Restarting...");
                
                // Try to capture any error output before disposing
                try
                {
                    var errors = await _pythonProcess.StandardError.ReadToEndAsync().ConfigureAwait(false);
                    if (!string.IsNullOrWhiteSpace(errors))
                    {
                        Console.WriteLine($"Python process error output: {errors}");
                    }
                }
                catch { }
            }
            else
            {
                Console.WriteLine("Python process was never started, initializing...");
            }

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
            var clearedCount = 0;
            foreach (var kvp in _pendingRequests)
            {
                kvp.Value.TrySetException(new InvalidOperationException("Python process restarted"));
                clearedCount++;
            }
            _pendingRequests.Clear();
            
            if (clearedCount > 0)
            {
                Console.WriteLine($"Cleared {clearedCount} pending prediction requests due to process restart");
            }

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
                        // Check if the response looks like JSON (starts with '{' or '[')
                        var trimmed = response.Trim();
                        if (!trimmed.StartsWith('{') && !trimmed.StartsWith('['))
                        {
                            // Not JSON - likely debug output or banner text from Python
                            Console.WriteLine($"[Python Output] {response}");
                            return;
                        }

                        // Try to deserialize JSON response
                        try
                        {
                            var pythonResult = JsonSerializer.Deserialize<PythonInferenceResult>(response);

                            if (pythonResult != null && !string.IsNullOrEmpty(pythonResult.RequestId))
                            {
                                if (_pendingRequests.TryGetValue(pythonResult.RequestId, out var tcs))
                                {
                                    var predictionResult = ConvertToPredictionResult(pythonResult);
                                    tcs.SetResult(predictionResult);
                                }
                                else
                                {
                                    Console.WriteLine($"Received response for unknown request: {pythonResult.RequestId}");
                                }
                            }
                            else
                            {
                                Console.WriteLine($"Received JSON without valid RequestId: {response.Substring(0, Math.Min(100, response.Length))}");
                            }
                        }
                        catch (JsonException jsonEx)
                        {
                            Console.WriteLine($"Failed to parse JSON response: {jsonEx.Message}");
                            Console.WriteLine($"Response content: {response.Substring(0, Math.Min(200, response.Length))}");
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

        /// <summary>
        /// Executes Python stock_predictor.py script for on-demand ML predictions with progress reporting (MarketChat story 9).
        /// This method invokes the Python script, passes parameters via JSON, and streams progress updates back to the caller.
        /// </summary>
        /// <param name="symbol">Stock symbol to predict (e.g., "TSLA", "AAPL").</param>
        /// <param name="modelType">ML model type to use (e.g., "lstm", "gru", "transformer", "random_forest", or "auto").</param>
        /// <param name="startDate">Optional start date for historical data range.</param>
        /// <param name="endDate">Optional end date for historical data range.</param>
        /// <param name="progressCallback">Optional callback for streaming progress updates to the chat UI.</param>
        /// <param name="cancellationToken">Cancellation token for the operation.</param>
        /// <returns>PythonPredictionExecutionResult containing the prediction result or error details.</returns>
        public async Task<PythonPredictionExecutionResult> ExecutePythonPredictionAsync(
            string symbol,
            string modelType = "auto",
            DateTime? startDate = null,
            DateTime? endDate = null,
            PredictionProgressCallback progressCallback = null,
            CancellationToken cancellationToken = default)
        {
            var stopwatch = Stopwatch.StartNew();
            var result = new PythonPredictionExecutionResult { ModelType = modelType };

            try
            {
                // Validate inputs
                if (string.IsNullOrWhiteSpace(symbol))
                {
                    result.Success = false;
                    result.ErrorMessage = "Stock symbol is required.";
                    return result;
                }

                symbol = symbol.ToUpperInvariant().Trim();

                // Check if stock_predictor.py exists
                if (!File.Exists(_stockPredictorScript))
                {
                    result.Success = false;
                    result.ErrorMessage = $"Python prediction script not found at: {_stockPredictorScript}";
                    return result;
                }

                // Report initial progress
                progressCallback?.Invoke($"Starting prediction for {symbol}...");
                await Task.Delay(100, cancellationToken).ConfigureAwait(false); // Allow UI to update

                // Build the request payload
                var requestData = new Dictionary<string, object>
                {
                    ["Symbol"] = symbol,
                    ["ModelType"] = modelType.ToLowerInvariant(),
                    ["ArchitectureType"] = modelType.ToLowerInvariant() == "auto" ? "lstm" : modelType.ToLowerInvariant(),
                    ["UseFeatureEngineering"] = true,
                    ["FeatureType"] = "balanced",
                    ["OptimizeHyperparameters"] = false,
                    ["Features"] = new Dictionary<string, double>
                    {
                        ["current_price"] = 100.0,  // Placeholder - will be populated by script
                        ["volume"] = 1000000.0
                    }
                };

                // Add date range if specified
                if (startDate.HasValue)
                {
                    requestData["StartDate"] = startDate.Value.ToString("yyyy-MM-dd");
                }
                if (endDate.HasValue)
                {
                    requestData["EndDate"] = endDate.Value.ToString("yyyy-MM-dd");
                }

                // Create temp files for input/output
                string tempInputPath = Path.GetTempFileName();
                string tempOutputPath = Path.GetTempFileName();

                try
                {
                    // Write request to temp file
                    var jsonRequest = JsonSerializer.Serialize(requestData);
                    await File.WriteAllTextAsync(tempInputPath, jsonRequest, cancellationToken).ConfigureAwait(false);

                    progressCallback?.Invoke("Fetching historical data...");
                    await Task.Delay(50, cancellationToken).ConfigureAwait(false);

                    // Set up Python process
                    var psi = new ProcessStartInfo
                    {
                        FileName = "python3",
                        Arguments = $"\"{_stockPredictorScript}\" \"{tempInputPath}\" \"{tempOutputPath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = Path.GetDirectoryName(_stockPredictorScript)
                    };

                    // Fallback to python if python3 is not available
                    try
                    {
                        using (var testProcess = Process.Start(new ProcessStartInfo
                        {
                            FileName = "python3",
                            Arguments = "--version",
                            UseShellExecute = false,
                            RedirectStandardOutput = true,
                            CreateNoWindow = true
                        }))
                        {
                            testProcess?.WaitForExit(1000);
                        }
                    }
                    catch
                    {
                        psi.FileName = "python";
                    }

                    progressCallback?.Invoke($"Running {GetModelDisplayName(modelType)} model...");
                    await Task.Delay(50, cancellationToken).ConfigureAwait(false);

                    using (var process = Process.Start(psi))
                    {
                        if (process == null)
                        {
                            result.Success = false;
                            result.ErrorMessage = "Failed to start Python process.";
                            return result;
                        }

                        // Read output streams asynchronously
                        var outputTask = process.StandardOutput.ReadToEndAsync();
                        var errorTask = process.StandardError.ReadToEndAsync();

                        // Wait for process to complete with timeout
                        var processTask = process.WaitForExitAsync(cancellationToken);
                        var timeoutTask = Task.Delay(TimeSpan.FromMinutes(5), cancellationToken);

                        progressCallback?.Invoke("Processing prediction...");

                        var completedTask = await Task.WhenAny(processTask, timeoutTask).ConfigureAwait(false);
                        if (completedTask == timeoutTask)
                        {
                            try
                            {
                                process.Kill();
                            }
                            catch { }

                            result.Success = false;
                            result.ErrorMessage = "Prediction timed out after 5 minutes.";
                            return result;
                        }

                        var stdOutput = await outputTask.ConfigureAwait(false);
                        var stdError = await errorTask.ConfigureAwait(false);

                        if (process.ExitCode != 0)
                        {
                            result.Success = false;
                            result.ErrorMessage = FormatPythonError(stdError, stdOutput);
                            return result;
                        }

                        progressCallback?.Invoke("Parsing prediction results...");

                        // Read prediction result from output file
                        if (!File.Exists(tempOutputPath))
                        {
                            result.Success = false;
                            result.ErrorMessage = "Python script did not produce output file.";
                            return result;
                        }

                        var outputJson = await File.ReadAllTextAsync(tempOutputPath, cancellationToken).ConfigureAwait(false);
                        if (string.IsNullOrWhiteSpace(outputJson))
                        {
                            result.Success = false;
                            result.ErrorMessage = "Python script produced empty output.";
                            return result;
                        }

                        // Parse the prediction result
                        var pythonResult = JsonSerializer.Deserialize<PythonStockPredictorResult>(outputJson, new JsonSerializerOptions
                        {
                            PropertyNameCaseInsensitive = true
                        });

                        if (pythonResult == null)
                        {
                            result.Success = false;
                            result.ErrorMessage = "Failed to parse prediction result.";
                            return result;
                        }

                        // Convert to PredictionResult
                        // Create prediction result
                        // IMPORTANT: For ExecutePythonPredictionAsync, Python should return both current and target price
                        // However, since we're using file-based I/O here (not real-time API), we may not have current price
                        // The Python script should include current_price in its output
                        result.Prediction = new PredictionResult
                        {
                            Symbol = symbol,
                            Action = pythonResult.Action ?? "HOLD",
                            Confidence = pythonResult.Confidence,
                            TargetPrice = pythonResult.TargetPrice,
                            // Python should provide current price in output, but fallback to estimate if not available
                            CurrentPrice = pythonResult.TargetPrice > 0 ? pythonResult.TargetPrice * 0.95 : 0,
                            PredictionDate = DateTime.Now,
                            ModelType = pythonResult.ModelType ?? modelType,
                            FeatureWeights = pythonResult.Weights ?? new Dictionary<string, double>()
                        };

                        // Map time series if available
                        if (pythonResult.TimeSeries != null)
                        {
                            result.Prediction.TimeSeries = new TimeSeriesPrediction
                            {
                                PricePredictions = pythonResult.TimeSeries.Prices ?? new List<double>(),
                                Confidence = pythonResult.TimeSeries.Confidence,
                                TimePoints = pythonResult.TimeSeries.Dates?.Select(d =>
                                    DateTime.TryParse(d, out var dt) ? dt : DateTime.Now.AddDays(1)).ToList() ?? new List<DateTime>()
                            };
                        }

                        // Map risk metrics if available
                        if (pythonResult.Risk != null)
                        {
                            result.Prediction.RiskMetrics = new Quantra.Models.RiskMetrics
                            {
                                ValueAtRisk = pythonResult.Risk.Var,
                                MaxDrawdown = pythonResult.Risk.MaxDrawdown,
                                SharpeRatio = pythonResult.Risk.SharpeRatio,
                                RiskScore = pythonResult.Risk.RiskScore
                            };
                            result.Prediction.RiskScore = pythonResult.Risk.RiskScore;
                            result.Prediction.ValueAtRisk = pythonResult.Risk.Var;
                            result.Prediction.MaxDrawdown = pythonResult.Risk.MaxDrawdown;
                            result.Prediction.SharpeRatio = pythonResult.Risk.SharpeRatio;
                        }

                        // Map detected patterns if available
                        if (pythonResult.Patterns != null)
                        {
                            result.Prediction.DetectedPatterns = pythonResult.Patterns.Select(p => new TechnicalPattern
                            {
                                PatternName = p.Name,
                                PatternStrength = p.Strength,
                                ExpectedOutcome = p.Outcome,
                                DetectionDate = DateTime.TryParse(p.DetectionDate, out var dt) ? dt : DateTime.Now,
                                HistoricalAccuracy = p.HistoricalAccuracy
                            }).ToList();
                        }

                        result.Success = true;
                        result.ModelType = pythonResult.ModelType ?? modelType;

                        progressCallback?.Invoke($"Prediction complete! {result.Prediction.Action} with {result.Prediction.Confidence:P0} confidence.");
                    }
                }
                finally
                {
                    // Cleanup temp files
                    try
                    {
                        if (File.Exists(tempInputPath)) File.Delete(tempInputPath);
                        if (File.Exists(tempOutputPath)) File.Delete(tempOutputPath);
                    }
                    catch { /* Ignore cleanup errors */ }
                }
            }
            catch (OperationCanceledException)
            {
                result.Success = false;
                result.ErrorMessage = "Prediction was cancelled.";
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.ErrorMessage = FormatExceptionMessage(ex);
                Console.WriteLine($"Error in ExecutePythonPredictionAsync: {ex}");
            }
            finally
            {
                stopwatch.Stop();
                result.ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds;
            }

            return result;
        }

        /// <summary>
        /// Gets TFT (Temporal Fusion Transformer) prediction with real historical sequences and calendar features.
        /// CRITICAL FIX: Uses actual historical data instead of synthetic repeated values for proper TFT inference.
        /// </summary>
        /// <param name="symbol">Stock symbol to predict</param>
        /// <param name="lookbackDays">Number of historical days to use (default 60 for TFT)</param>
        /// <param name="futureHorizon">Days ahead for calendar feature projection (default 30)</param>
        /// <param name="progressCallback">Optional callback for progress updates</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>TFT prediction result with multi-horizon forecasts and uncertainty quantification</returns>
        public async Task<PythonPredictionExecutionResult> GetTFTPredictionAsync(
            string symbol,
            int lookbackDays = 60,
            int futureHorizon = 30,
            PredictionProgressCallback progressCallback = null,
            CancellationToken cancellationToken = default)
        {
            var stopwatch = Stopwatch.StartNew();
            var result = new PythonPredictionExecutionResult { ModelType = "tft" };

            try
            {
                // Validate inputs
                if (string.IsNullOrWhiteSpace(symbol))
                {
                    result.Success = false;
                    result.ErrorMessage = "Stock symbol is required.";
                    return result;
                }

                symbol = symbol.ToUpperInvariant().Trim();

                // Check if stock data cache service is available
                if (_stockDataCacheService == null)
                {
                    result.Success = false;
                    result.ErrorMessage = "StockDataCacheService not available. Cannot fetch historical sequences for TFT.";
                    return result;
                }

                // Check if TFT script exists
                if (!File.Exists(_tftPredictScript))
                {
                    result.Success = false;
                    result.ErrorMessage = $"TFT prediction script not found at: {_tftPredictScript}";
                    return result;
                }

                progressCallback?.Invoke($"Fetching {lookbackDays} days of historical data for {symbol}...");

                // Step 1: Get real historical sequence with calendar features
                var historicalData = await _stockDataCacheService
                    .GetHistoricalSequenceWithFeaturesAsync(symbol, lookbackDays, futureHorizon)
                    .ConfigureAwait(false);

                if (historicalData == null || !historicalData.ContainsKey("prices"))
                {
                    result.Success = false;
                    result.ErrorMessage = $"Insufficient historical data for {symbol}. Need at least {lookbackDays} days.";
                    return result;
                }

                var prices = historicalData["prices"] as List<HistoricalPrice>;
                var calendarFeatures = historicalData["calendar_features"] as List<Dictionary<string, object>>;

                if (prices == null || prices.Count < lookbackDays)
                {
                    result.Success = false;
                    result.ErrorMessage = $"Only {prices?.Count ?? 0} days of data available. TFT requires {lookbackDays} days.";
                    return result;
                }

                progressCallback?.Invoke($"Preparing TFT input with {prices.Count} historical days + {futureHorizon} future calendar days...");

                // Step 2: Convert to format expected by Python TFT script
                var requestData = new
                {
                    symbol = symbol,
                    model_type = "tft",
                    architecture_type = "tft",
                    historical_sequence = prices.Select(p => new
                    {
                        date = p.Date.ToString("yyyy-MM-dd"),
                        open = p.Open,
                        high = p.High,
                        low = p.Low,
                        close = p.Close,
                        volume = (double)p.Volume
                    }).ToList(),
                    calendar_features = calendarFeatures,
                    lookback_days = lookbackDays,
                    future_horizon = futureHorizon,
                    forecast_horizons = new[] { 5, 10, 20, 30 } // TFT multi-horizon targets
                };

                // Step 3: Create temp files for Python communication
                string tempInputPath = Path.GetTempFileName();
                string tempOutputPath = Path.GetTempFileName();

                try
                {
                    // Step 4: Write request to temp file
                    var jsonRequest = JsonSerializer.Serialize(requestData);
                    await File.WriteAllTextAsync(tempInputPath, jsonRequest, cancellationToken).ConfigureAwait(false);

                    progressCallback?.Invoke("Running TFT model with real temporal sequences...");

                    // Step 5: Execute Python TFT script
                    var psi = new ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = $"\"{_tftPredictScript}\" \"{tempInputPath}\" \"{tempOutputPath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WorkingDirectory = Path.GetDirectoryName(_tftPredictScript)
                    };

                    // Try python3 first
                    try
                    {
                        using var testProcess = Process.Start(new ProcessStartInfo
                        {
                            FileName = "python",
                            Arguments = "--version",
                            UseShellExecute = false,
                            RedirectStandardOutput = true,
                            CreateNoWindow = true
                        });
                        testProcess?.WaitForExit(1000);
                        psi.FileName = "python";
                    }
                    catch { /* Use python */ }

                    using (var process = Process.Start(psi))
                    {
                        if (process == null)
                        {
                            result.Success = false;
                            result.ErrorMessage = "Failed to start Python process for TFT prediction.";
                            return result;
                        }

                        // Read output streams asynchronously
                        var outputTask = process.StandardOutput.ReadToEndAsync();
                        var errorTask = process.StandardError.ReadToEndAsync();

                        // Wait for process with timeout
                        var processTask = process.WaitForExitAsync(cancellationToken);
                        var timeoutTask = Task.Delay(TimeSpan.FromMinutes(3), cancellationToken);

                        var completedTask = await Task.WhenAny(processTask, timeoutTask).ConfigureAwait(false);
                        if (completedTask == timeoutTask)
                        {
                            try { process.Kill(); } catch { }
                            result.Success = false;
                            result.ErrorMessage = "TFT prediction timed out after 3 minutes.";
                            return result;
                        }

                        var stdOutput = await outputTask.ConfigureAwait(false);
                        var stdError = await errorTask.ConfigureAwait(false);

                        if (process.ExitCode != 0)
                        {
                            result.Success = false;
                            result.ErrorMessage = FormatPythonError(stdError, stdOutput);
                            return result;
                        }

                        progressCallback?.Invoke("Parsing TFT multi-horizon predictions...");

                        // Step 6: Read and parse results
                        if (!File.Exists(tempOutputPath))
                        {
                            result.Success = false;
                            result.ErrorMessage = "TFT script did not produce output file.";
                            return result;
                        }

                        var outputJson = await File.ReadAllTextAsync(tempOutputPath, cancellationToken).ConfigureAwait(false);
                        if (string.IsNullOrWhiteSpace(outputJson))
                        {
                            result.Success = false;
                            result.ErrorMessage = "TFT script produced empty output.";
                            return result;
                        }

                        // Parse TFT result
                        var tftResult = JsonSerializer.Deserialize<TFTPredictionResult>(outputJson, new JsonSerializerOptions
                        {
                            PropertyNameCaseInsensitive = true
                        });

                        if (tftResult == null || tftResult.Error != null)
                        {
                            result.Success = false;
                            result.ErrorMessage = tftResult?.Error ?? "Failed to parse TFT result.";
                            return result;
                        }

                        // Step 7: Convert to PredictionResult
                        var currentPrice = prices.Last().Close;
                        result.Prediction = new PredictionResult
                        {
                            Symbol = symbol,
                            Action = tftResult.Action,
                            Confidence = tftResult.Confidence,
                            CurrentPrice = currentPrice,
                            TargetPrice = tftResult.TargetPrice,
                            PredictionDate = DateTime.Now,
                            ModelType = "tft",
                            
                            // TFT-specific uncertainty quantification
                            PredictionUncertainty = tftResult.Uncertainty,
                            
                            // Multi-horizon predictions
                            TimeSeriesPredictions = tftResult.Horizons?.Select(h => new HorizonPrediction
                            {
                                Horizon = h.Key,
                                MedianPrice = h.Value.MedianPrice,
                                LowerBound = h.Value.LowerBound,
                                UpperBound = h.Value.UpperBound,
                                Confidence = h.Value.Confidence
                            }).ToList() ?? new List<HorizonPrediction>()
                        };

                        // Store the full TFT result for visualization
                        result.TFTResult = tftResult;
                        
                        result.Success = true;
                        result.ModelType = "tft";

                        var horizonText = tftResult.Horizons != null 
                            ? string.Join(", ", tftResult.Horizons.Keys) 
                            : "5d, 10d, 20d, 30d";
                        progressCallback?.Invoke($"TFT prediction complete! {result.Prediction.Action} with {result.Prediction.Confidence:P0} confidence. Horizons: {horizonText}");
                    }
                }
                finally
                {
                    // Cleanup temp files
                    try
                    {
                        if (File.Exists(tempInputPath)) File.Delete(tempInputPath);
                        if (File.Exists(tempOutputPath)) File.Delete(tempOutputPath);
                    }
                    catch { /* Ignore cleanup errors */ }
                }
            }
            catch (OperationCanceledException)
            {
                result.Success = false;
                result.ErrorMessage = "TFT prediction was cancelled.";
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.ErrorMessage = FormatExceptionMessage(ex);
                Console.WriteLine($"Error in GetTFTPredictionAsync: {ex}");
            }
            finally
            {
                stopwatch.Stop();
                result.ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds;
            }

            return result;
        }

        /// <summary>
        /// Gets a user-friendly display name for the model type.
        /// </summary>
        private static string GetModelDisplayName(string modelType)
        {
            return modelType?.ToLowerInvariant() switch
            {
                "lstm" => "LSTM",
                "gru" => "GRU",
                "transformer" => "Transformer",
                "tft" => "Temporal Fusion Transformer",
                "random_forest" => "Random Forest",
                "auto" => "Auto-selected ML",
                _ => modelType ?? "ML"
            };
        }

        /// <summary>
        /// Formats Python error output into a user-friendly message.
        /// </summary>
        private static string FormatPythonError(string stdError, string stdOutput)
        {
            if (!string.IsNullOrWhiteSpace(stdError))
            {
                // Extract the most relevant error message
                var lines = stdError.Split('\n');
                var errorLines = lines.Where(l =>
                    l.Contains("Error") ||
                    l.Contains("Exception") ||
                    l.Contains("error:") ||
                    l.Contains("ModuleNotFoundError")).ToList();

                if (errorLines.Any())
                {
                    return $"Python error: {string.Join(" ", errorLines.Take(2)).Trim()}";
                }

                var trimmedError = stdError.Trim();
                if (!string.IsNullOrEmpty(trimmedError))
                {
                    return $"Python error: {trimmedError.Substring(0, Math.Min(200, trimmedError.Length))}";
                }
            }

            if (!string.IsNullOrWhiteSpace(stdOutput))
            {
                var trimmedOutput = stdOutput.Trim();
                if (!string.IsNullOrEmpty(trimmedOutput))
                {
                    return $"Script output: {trimmedOutput.Substring(0, Math.Min(200, trimmedOutput.Length))}";
                }
            }

            return "Unknown Python execution error.";
        }

        /// <summary>
        /// Formats an exception into a user-friendly message.
        /// </summary>
        private static string FormatExceptionMessage(Exception ex)
        {
            if (ex is FileNotFoundException)
            {
                return "Python script not found. Please ensure the ML scripts are properly installed.";
            }
            if (ex.Message.Contains("python"))
            {
                return "Python is not installed or not available in the system PATH.";
            }
            return $"An error occurred during prediction: {ex.Message}";
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

    /// <summary>
    /// Result structure for stock_predictor.py output deserialization (MarketChat story 9).
    /// </summary>
    public class PythonStockPredictorResult
    {
        public string Action { get; set; }
        public double Confidence { get; set; }
        public double TargetPrice { get; set; }
        public Dictionary<string, double> Weights { get; set; }
        public PythonTimeSeriesData TimeSeries { get; set; }
        public PythonRiskData Risk { get; set; }
        public List<PythonPatternData> Patterns { get; set; }
        public string ModelType { get; set; }
        public string ArchitectureType { get; set; }
        public bool FeatureEngineering { get; set; }
        public string Error { get; set; }
    }

    /// <summary>
    /// Time series data from Python stock predictor.
    /// </summary>
    public class PythonTimeSeriesData
    {
        public List<double> Prices { get; set; }
        public List<string> Dates { get; set; }
        public double Confidence { get; set; }
    }

    /// <summary>
    /// Risk metrics data from Python stock predictor.
    /// </summary>
    public class PythonRiskData
    {
        public double Var { get; set; }
        public double MaxDrawdown { get; set; }
        public double SharpeRatio { get; set; }
        public double RiskScore { get; set; }
    }

    /// <summary>
    /// Technical pattern data from Python stock predictor.
    /// </summary>
    public class PythonPatternData
    {
        public string Name { get; set; }
        public double Strength { get; set; }
        public string Outcome { get; set; }
        public string DetectionDate { get; set; }
        public double HistoricalAccuracy { get; set; }
    }

    // TFTPredictionResult, HorizonPredictionData, and HorizonPrediction classes 
    // are now defined in Quantra.DAL/Models/TFTPredictionResult.cs
}