using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Quantra.CrossCutting.Logging;
using Quantra.CrossCutting.Monitoring.Models;

namespace Quantra.CrossCutting.Monitoring
{
    /// <summary>
    /// Centralized monitoring system for tracking performance metrics and system health.
    /// </summary>
    public class MonitoringManager : IMonitoringManager
    {
        private static readonly Lazy<MonitoringManager> _instance = new Lazy<MonitoringManager>(() => new MonitoringManager());
        private readonly ILogger _logger;
        private readonly ConcurrentDictionary<string, OperationMetrics> _metrics;
        private readonly DateTime _startTime;
        private readonly Process _currentProcess;
        private readonly ConcurrentDictionary<string, DateTime> _lastMemoryRecordings;
        private readonly ConcurrentDictionary<string, ComponentHealth> _componentHealth;
        private readonly int _maxRecentExecutionSamples = 100;
        private readonly int _maxRecentFailures = 10;
        private readonly Lazy<PredictiveMonitor> _predictiveMonitor;
        
        /// <summary>
        /// Gets the singleton instance of the MonitoringManager.
        /// </summary>
        public static MonitoringManager Instance => _instance.Value;

        /// <inheritdoc />
        public string ModuleName => "Monitoring";

        /// <summary>
        /// Private constructor to enforce singleton pattern.
        /// </summary>
        private MonitoringManager()
        {
            _metrics = new ConcurrentDictionary<string, OperationMetrics>();
            _lastMemoryRecordings = new ConcurrentDictionary<string, DateTime>();
            _componentHealth = new ConcurrentDictionary<string, ComponentHealth>();
            _startTime = DateTime.Now;
            _currentProcess = Process.GetCurrentProcess();
            _logger = Log.ForType<MonitoringManager>();
            _predictiveMonitor = new Lazy<PredictiveMonitor>(() => new PredictiveMonitor(this));
        }

        /// <inheritdoc />
        public void Initialize(string configurationSection = null)
        {
            // Record initial memory usage
            RecordMemoryUsage("Initialize");
            _logger.Information("MonitoringManager initialized");
        }

        /// <inheritdoc />
        public void RecordMetric(string name, double value, IDictionary<string, string> dimensions = null)
        {
            if (string.IsNullOrEmpty(name))
            {
                throw new ArgumentNullException(nameof(name));
            }

            try
            {
                // Log the metric
                var contextualLogger = _logger.ForContext("MetricName", name)
                                              .ForContext("MetricValue", value);
                
                if (dimensions != null)
                {
                    foreach (var dim in dimensions)
                    {
                        contextualLogger = contextualLogger.ForContext($"Dimension_{dim.Key}", dim.Value);
                    }
                }
                
                contextualLogger.Debug("Recorded metric: {MetricName}={MetricValue}", name, value);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to record metric {MetricName}", name);
            }
        }

        /// <inheritdoc />
        public TimeSpan RecordExecutionTime(string operationName, Action action, IDictionary<string, string> dimensions = null)
        {
            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }

            var stopwatch = Stopwatch.StartNew();
            bool success = true;
            Exception error = null;
            
            try
            {
                action();
            }
            catch (Exception ex)
            {
                success = false;
                error = ex;
                throw;
            }
            finally
            {
                stopwatch.Stop();
                var duration = stopwatch.Elapsed;
                UpdateOperationMetrics(operationName, duration.TotalMilliseconds, success, error, dimensions);
            }
            
            return stopwatch.Elapsed;
        }

        /// <inheritdoc />
        public (T Result, TimeSpan Duration) RecordExecutionTime<T>(string operationName, Func<T> func, IDictionary<string, string> dimensions = null)
        {
            if (func == null)
            {
                throw new ArgumentNullException(nameof(func));
            }

            var stopwatch = Stopwatch.StartNew();
            bool success = true;
            Exception error = null;
            T result;
            
            try
            {
                result = func();
            }
            catch (Exception ex)
            {
                success = false;
                error = ex;
                throw;
            }
            finally
            {
                stopwatch.Stop();
                var duration = stopwatch.Elapsed;
                UpdateOperationMetrics(operationName, duration.TotalMilliseconds, success, error, dimensions);
            }
            
            return (result, stopwatch.Elapsed);
        }

        /// <inheritdoc />
        public async Task<TimeSpan> RecordExecutionTimeAsync(string operationName, Func<Task> action, IDictionary<string, string> dimensions = null)
        {
            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }

            var stopwatch = Stopwatch.StartNew();
            bool success = true;
            Exception error = null;
            
            try
            {
                await action().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                success = false;
                error = ex;
                throw;
            }
            finally
            {
                stopwatch.Stop();
                var duration = stopwatch.Elapsed;
                UpdateOperationMetrics(operationName, duration.TotalMilliseconds, success, error, dimensions);
            }
            
            return stopwatch.Elapsed;
        }

        /// <inheritdoc />
        public async Task<(T Result, TimeSpan Duration)> RecordExecutionTimeAsync<T>(string operationName, Func<Task<T>> func, IDictionary<string, string> dimensions = null)
        {
            if (func == null)
            {
                throw new ArgumentNullException(nameof(func));
            }

            var stopwatch = Stopwatch.StartNew();
            bool success = true;
            Exception error = null;
            T result;
            
            try
            {
                result = await func().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                success = false;
                error = ex;
                throw;
            }
            finally
            {
                stopwatch.Stop();
                var duration = stopwatch.Elapsed;
                UpdateOperationMetrics(operationName, duration.TotalMilliseconds, success, error, dimensions);
            }
            
            return (result, stopwatch.Elapsed);
        }

        /// <inheritdoc />
        public void RecordSuccess(string operationName, IDictionary<string, string> dimensions = null)
        {
            UpdateOperationMetrics(operationName, null, true, null, dimensions);
        }

        /// <inheritdoc />
        public void RecordFailure(string operationName, Exception exception = null, IDictionary<string, string> dimensions = null)
        {
            UpdateOperationMetrics(operationName, null, false, exception, dimensions);
        }

        /// <inheritdoc />
        public void RecordMemoryUsage(string context = null)
        {
            try
            {
                context = context ?? "Default";
                
                // Don't record too frequently for the same context
                var now = DateTime.Now;
                if (_lastMemoryRecordings.TryGetValue(context, out var lastTime) && 
                    (now - lastTime).TotalSeconds < 30)
                {
                    return;
                }
                
                _lastMemoryRecordings[context] = now;
                
                // Refresh process info
                _currentProcess.Refresh();
                
                var memoryUsageMb = _currentProcess.WorkingSet64 / (1024.0 * 1024.0);
                var privateMemoryMb = _currentProcess.PrivateMemorySize64 / (1024.0 * 1024.0);
                var virtualMemoryMb = _currentProcess.VirtualMemorySize64 / (1024.0 * 1024.0);
                
                _logger.ForContext("Context", context)
                      .ForContext("WorkingSetMB", memoryUsageMb)
                      .ForContext("PrivateMemoryMB", privateMemoryMb)
                      .ForContext("VirtualMemoryMB", virtualMemoryMb)
                      .Debug("Memory Usage: {WorkingSetMB:F2}MB working set, {PrivateMemoryMB:F2}MB private", 
                          memoryUsageMb, privateMemoryMb);
                
                // Record as a metric as well
                RecordMetric($"Memory.{context}.WorkingSetMB", memoryUsageMb);
                RecordMetric($"Memory.{context}.PrivateMemoryMB", privateMemoryMb);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to record memory usage for context {Context}", context);
            }
        }

        /// <inheritdoc />
        public OperationMetrics GetMetrics(string operationName)
        {
            return _metrics.TryGetValue(operationName, out var metrics) ? metrics : null;
        }

        /// <inheritdoc />
        public IReadOnlyDictionary<string, OperationMetrics> GetAllMetrics()
        {
            return _metrics;
        }

        /// <inheritdoc />
        public void ClearHistoricalMetrics(TimeSpan age)
        {
            var cutoffTime = DateTime.Now - age;
            
            foreach (var metrics in _metrics.Values)
            {
                lock (metrics)
                {
                    metrics.RecentFailures.RemoveAll(f => f.OccurredAt < cutoffTime);
                }
            }
        }

        /// <inheritdoc />
        public async Task<ComponentHealth> CheckComponentHealthAsync(string componentName)
        {
            if (string.IsNullOrEmpty(componentName))
            {
                throw new ArgumentNullException(nameof(componentName));
            }

            // Return cached health if recent
            if (_componentHealth.TryGetValue(componentName, out var cachedHealth))
            {
                if ((DateTime.Now - cachedHealth.CheckedAt).TotalMinutes < 5)
                {
                    return cachedHealth;
                }
            }
            
            // Create a new health check
            var health = new ComponentHealth
            {
                Name = componentName,
                CheckedAt = DateTime.Now,
                Status = HealthStatus.Unknown,
                Description = "Health check not implemented for this component",
                Metrics = new Dictionary<string, object>()
            };
            
            try
            {
                // Check operation metrics for this component
                var componentMetrics = _metrics
                    .Where(m => m.Key.StartsWith(componentName, StringComparison.OrdinalIgnoreCase))
                    .Select(m => m.Value)
                    .ToList();
                
                if (componentMetrics.Any())
                {
                    health.Metrics["OperationCount"] = componentMetrics.Count;
                    health.Metrics["TotalExecutions"] = componentMetrics.Sum(m => m.ExecutionCount);
                    health.Metrics["SuccessRate"] = componentMetrics.Sum(m => m.SuccessCount) * 100.0 / 
                                                   Math.Max(1, componentMetrics.Sum(m => m.ExecutionCount));
                    health.Metrics["AverageExecutionTimeMs"] = componentMetrics.Any(m => m.ExecutionCount > 0) ?
                        componentMetrics.Sum(m => m.AverageExecutionTimeMs * m.ExecutionCount) / 
                        Math.Max(1, componentMetrics.Sum(m => m.ExecutionCount)) : 0;
                    
                    // Determine status based on success rates
                    var successRate = (double)health.Metrics["SuccessRate"];
                    if (successRate >= 99)
                    {
                        health.Status = HealthStatus.Healthy;
                        health.Description = $"Component is healthy with {successRate:F1}% success rate";
                    }
                    else if (successRate >= 90)
                    {
                        health.Status = HealthStatus.Degraded;
                        health.Description = $"Component is degraded with {successRate:F1}% success rate";
                    }
                    else
                    {
                        health.Status = HealthStatus.Unhealthy;
                        health.Description = $"Component is unhealthy with {successRate:F1}% success rate";
                    }
                }

                // Perform component-specific checks
                switch (componentName.ToLowerInvariant())
                {
                    case "database":
                        await CheckDatabaseHealthAsync(health);
                        break;
                    case "alphavantage":
                    case "webull":
                    case "api":
                        CheckApiHealth(health);
                        break;
                    case "mlinference":
                        CheckMlInferenceHealth(health);
                        break;
                    case "memory":
                        CheckMemoryHealth(health);
                        break;
                }
            }
            catch (Exception ex)
            {
                health.Status = HealthStatus.Unknown;
                health.Description = "Failed to check component health";
                health.Error = ex.Message;
                _logger.Error(ex, "Failed to check health for component {ComponentName}", componentName);
            }
            
            // Cache the health status
            _componentHealth[componentName] = health;
            
            return health;
        }

        /// <inheritdoc />
        public async Task<SystemHealth> CheckOverallHealthAsync()
        {
            var systemHealth = new SystemHealth
            {
                CheckedAt = DateTime.Now,
                Status = HealthStatus.Healthy,
                Components = new ConcurrentDictionary<string, ComponentHealth>(),
                Uptime = DateTime.Now - _startTime,
                Resources = GetResourceUtilization()
            };
            
            // Check key components concurrently without using Task.Run for async operations
            var componentCheckTasks = new List<Task>
            {
                CheckComponentHealthWithErrorHandlingAsync("Database", systemHealth),
                CheckComponentHealthWithErrorHandlingAsync("API", systemHealth),
                CheckComponentHealthWithErrorHandlingAsync("MLInference", systemHealth),
                CheckComponentHealthWithErrorHandlingAsync("Memory", systemHealth)
            };
            
            await Task.WhenAll(componentCheckTasks).ConfigureAwait(false);
            
            // Determine overall status
            if (systemHealth.Components.Values.Any(c => c != null && c.Status == HealthStatus.Unhealthy))
            {
                systemHealth.Status = HealthStatus.Unhealthy;
            }
            else if (systemHealth.Components.Values.Any(c => c != null && c.Status == HealthStatus.Degraded))
            {
                systemHealth.Status = HealthStatus.Degraded;
            }
            
            return systemHealth;
        }

        /// <inheritdoc />
        public ResourceUtilization GetResourceUtilization()
        {
            try
            {
                _currentProcess.Refresh();
                
                long workingSet = _currentProcess.WorkingSet64;
                long privateBytes = _currentProcess.PrivateMemorySize64;
                
                var cpuTime = _currentProcess.TotalProcessorTime;
                var cpuPercent = cpuTime.TotalMilliseconds / 
                                (Environment.ProcessorCount * (DateTime.Now - _startTime).TotalMilliseconds) * 100;
                
                var diskUsage = GetDiskUsage();
                
                return new ResourceUtilization
                {
                    CpuUsagePercent = Math.Min(100, cpuPercent),
                    MemoryUsageBytes = workingSet,
                    DiskUsageBytes = diskUsage.Used,
                    DiskTotalBytes = diskUsage.Total,
                    MeasuredAt = DateTime.Now
                };
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to get resource utilization");
                return new ResourceUtilization
                {
                    MeasuredAt = DateTime.Now
                };
            }
        }
        
        /// <summary>
        /// Updates the metrics for an operation.
        /// </summary>
        private void UpdateOperationMetrics(string operationName, double? executionTimeMs, bool success, 
            Exception error = null, IDictionary<string, string> dimensions = null)
        {
            try
            {
                var metrics = _metrics.GetOrAdd(operationName, name => new OperationMetrics { Name = name });
                
                lock (metrics)
                {
                    metrics.ExecutionCount++;
                    metrics.LastExecutionTime = DateTime.Now;
                    
                    if (success)
                    {
                        metrics.SuccessCount++;
                    }
                    else
                    {
                        metrics.FailureCount++;
                        
                        // Add failure record if we have error details
                        if (error != null)
                        {
                            var failureRecord = new FailureRecord
                            {
                                OccurredAt = DateTime.Now,
                                ExceptionType = error.GetType().Name,
                                Message = error.Message,
                                Dimensions = dimensions?.ToDictionary(kvp => kvp.Key, kvp => kvp.Value)
                            };
                            
                            metrics.RecentFailures.Add(failureRecord);
                            
                            // Trim the failure list if it gets too long
                            if (metrics.RecentFailures.Count > _maxRecentFailures)
                            {
                                metrics.RecentFailures.RemoveAt(0);
                            }
                        }
                    }
                    
                    // Update timing statistics if execution time was provided
                    if (executionTimeMs.HasValue)
                    {
                        double timeMs = executionTimeMs.Value;
                        
                        // Update min/max
                        if (metrics.ExecutionCount == 1)
                        {
                            metrics.MinExecutionTimeMs = timeMs;
                            metrics.MaxExecutionTimeMs = timeMs;
                            metrics.AverageExecutionTimeMs = timeMs;
                        }
                        else
                        {
                            metrics.MinExecutionTimeMs = Math.Min(metrics.MinExecutionTimeMs, timeMs);
                            metrics.MaxExecutionTimeMs = Math.Max(metrics.MaxExecutionTimeMs, timeMs);
                            
                            // Update running average
                            metrics.AverageExecutionTimeMs = ((metrics.AverageExecutionTimeMs * (metrics.ExecutionCount - 1)) + timeMs) / 
                                                           metrics.ExecutionCount;
                        }
                        
                        // Add to recent execution times
                        metrics.RecentExecutionTimesMs.Add(timeMs);
                        
                        // Log performance timing for profiling
                        var contextualLogger = _logger.ForContext("OperationName", operationName)
                                                      .ForContext("ExecutionTimeMs", timeMs)
                                                      .ForContext("Success", success);
                        
                        if (dimensions != null)
                        {
                            foreach (var dim in dimensions)
                            {
                                contextualLogger = contextualLogger.ForContext($"Dimension_{dim.Key}", dim.Value);
                            }
                        }
                        
                        // Log timing information to console/debug for profiling
                        contextualLogger.Information("PERFORMANCE: {OperationName} completed in {ExecutionTimeMs:F2}ms (Success: {Success})", 
                            operationName, timeMs, success);
                        
                        // Trim the recent times list if it gets too long
                        if (metrics.RecentExecutionTimesMs.Count > _maxRecentExecutionSamples)
                        {
                            metrics.RecentExecutionTimesMs.RemoveAt(0);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to update metrics for operation {OperationName}", operationName);
            }
        }

        /// <summary>
        /// Checks the health of the database.
        /// </summary>
        private async Task CheckDatabaseHealthAsync(ComponentHealth health)
        {
            try
            {
                using var connection = Quantra.DatabaseMonolith.GetConnection();
                await connection.OpenAsync().ConfigureAwait(false);
                
                using var command = connection.CreateCommand();
                command.CommandText = "SELECT 1";
                await command.ExecuteScalarAsync().ConfigureAwait(false);
                
                health.Status = HealthStatus.Healthy;
                health.Description = "Database connection is working properly";
            }
            catch (Exception ex)
            {
                health.Status = HealthStatus.Unhealthy;
                health.Description = "Cannot connect to database";
                health.Error = ex.Message;
            }
        }
        
        /// <summary>
        /// Checks the health of the API services.
        /// </summary>
        private void CheckApiHealth(ComponentHealth health)
        {
            var componentName = health.Name.ToLowerInvariant();
            var apiMetrics = _metrics
                .Where(m => m.Key.Contains("API") || 
                          m.Key.StartsWith(componentName, StringComparison.OrdinalIgnoreCase))
                .Select(m => m.Value)
                .ToList();
            
            if (apiMetrics.Any())
            {
                // Calculate recent success rate (last 100 calls max)
                var recentSuccessRate = apiMetrics
                    .Where(m => m.RecentExecutionTimesMs != null && m.RecentExecutionTimesMs.Count > 0)
                    .Sum(m => m.SuccessCount) * 100.0 / 
                    Math.Max(1, apiMetrics.Sum(m => m.ExecutionCount));
                
                health.Metrics["RecentSuccessRate"] = recentSuccessRate;
                
                // Calculate average response times
                var metricsWithTiming = apiMetrics.Where(m => m.RecentExecutionTimesMs != null && m.RecentExecutionTimesMs.Count > 0).ToList();
                double avgResponseTime = 0;
                if (metricsWithTiming.Count > 0)
                {
                    avgResponseTime = metricsWithTiming.Average(m => m.AverageExecutionTimeMs);
                }
                health.Metrics["AverageResponseTimeMs"] = avgResponseTime;
                
                if (recentSuccessRate > 98)
                {
                    health.Status = HealthStatus.Healthy;
                    health.Description = $"API is operating normally ({recentSuccessRate:F1}% success rate)";
                }
                else if (recentSuccessRate > 90)
                {
                    health.Status = HealthStatus.Degraded;
                    health.Description = $"API is experiencing some failures ({recentSuccessRate:F1}% success rate)";
                }
                else
                {
                    health.Status = HealthStatus.Unhealthy;
                    health.Description = $"API is failing frequently ({recentSuccessRate:F1}% success rate)";
                }
            }
            else
            {
                health.Status = HealthStatus.Unknown;
                health.Description = "No API metrics available";
            }
        }
        
        /// <summary>
        /// Checks the health of the ML inference service.
        /// </summary>
        private void CheckMlInferenceHealth(ComponentHealth health)
        {
            var mlMetrics = _metrics
                .Where(m => m.Key.Contains("MLInference") || m.Key.Contains("Prediction"))
                .Select(m => m.Value)
                .ToList();
            
            if (mlMetrics.Any())
            {
                var recentSuccessRate = mlMetrics.Sum(m => m.SuccessCount) * 100.0 / 
                                       Math.Max(1, mlMetrics.Sum(m => m.ExecutionCount));
                
                health.Metrics["SuccessRate"] = recentSuccessRate;
                health.Metrics["AverageInferenceTimeMs"] = mlMetrics
                    .Where(m => m.ExecutionCount > 0)
                    .Average(m => m.AverageExecutionTimeMs);
                
                if (recentSuccessRate > 95)
                {
                    health.Status = HealthStatus.Healthy;
                    health.Description = $"ML inference is working properly ({recentSuccessRate:F1}% success)";
                }
                else if (recentSuccessRate > 80)
                {
                    health.Status = HealthStatus.Degraded;
                    health.Description = $"ML inference has degraded performance ({recentSuccessRate:F1}% success)";
                }
                else
                {
                    health.Status = HealthStatus.Unhealthy;
                    health.Description = $"ML inference is failing frequently ({recentSuccessRate:F1}% success)";
                }
            }
            else
            {
                health.Status = HealthStatus.Unknown;
                health.Description = "No ML inference metrics available";
            }
        }
        
        /// <summary>
        /// Checks memory health.
        /// </summary>
        private void CheckMemoryHealth(ComponentHealth health)
        {
            _currentProcess.Refresh();
            
            long workingSetBytes = _currentProcess.WorkingSet64;
            long privateMemoryBytes = _currentProcess.PrivateMemorySize64;
            long workingSetMb = workingSetBytes / (1024 * 1024);
            long privateMemoryMb = privateMemoryBytes / (1024 * 1024);
            
            health.Metrics["WorkingSetMB"] = workingSetMb;
            health.Metrics["PrivateMemoryMB"] = privateMemoryMb;
            
            // Evaluate memory status
            const int highWatermarkMb = 1024; // 1GB
            const int criticalWatermarkMb = 1536; // 1.5GB
            
            if (privateMemoryMb > criticalWatermarkMb)
            {
                health.Status = HealthStatus.Unhealthy;
                health.Description = $"Memory usage is critical: {privateMemoryMb}MB";
            }
            else if (privateMemoryMb > highWatermarkMb)
            {
                health.Status = HealthStatus.Degraded;
                health.Description = $"Memory usage is high: {privateMemoryMb}MB";
            }
            else
            {
                health.Status = HealthStatus.Healthy;
                health.Description = $"Memory usage is normal: {privateMemoryMb}MB";
            }
        }
        
        /// <summary>
        /// Gets disk usage information.
        /// </summary>
        private (long Used, long Total) GetDiskUsage()
        {
            try
            {
                var appDirectory = AppDomain.CurrentDomain.BaseDirectory;
                var driveInfo = new DriveInfo(Path.GetPathRoot(appDirectory));
                
                return (driveInfo.TotalSize - driveInfo.AvailableFreeSpace, driveInfo.TotalSize);
            }
            catch (Exception)
            {
                return (0, 0);
            }
        }
        
        /// <inheritdoc />
        public PredictiveMonitor GetPredictiveMonitor()
        {
            return _predictiveMonitor.Value;
        }
        
        /// <inheritdoc />
        public void RecordTimeSeriesDataPoint(string metricName, double value, IDictionary<string, string> dimensions = null)
        {
            GetPredictiveMonitor().RecordMetricDataPoint(metricName, value, dimensions);
        }
        
        /// <inheritdoc />
        public IEnumerable<TimeSeriesMetricPoint> GetMetricTimeSeries(string metricName, TimeSpan? timeWindow = null)
        {
            return GetPredictiveMonitor().GetMetricTimeSeries(metricName, timeWindow);
        }
        
        /// <inheritdoc />
        public MetricPrediction PredictMetricValue(string metricName, TimeSpan predictionHorizon, TimeSpan? historyWindow = null)
        {
            return GetPredictiveMonitor().PredictMetricValue(metricName, predictionHorizon, historyWindow);
        }
        
        /// <inheritdoc />
        public PredictiveAnalysisResult PerformPredictiveAnalysis(
            TimeSpan predictionHorizon,
            TimeSpan analysisTimeWindow,
            Func<string, bool> metricFilter = null,
            Dictionary<string, double> thresholds = null)
        {
            return GetPredictiveMonitor().PerformPredictiveAnalysis(predictionHorizon, analysisTimeWindow, metricFilter, thresholds);
        }
        
        /// <inheritdoc />
        public Task<dynamic> DetectAnomaliesAsync(string metricName, TimeSpan analysisTimeWindow)
        {
            return GetPredictiveMonitor().DetectAnomaliesAsync(metricName, analysisTimeWindow);
        }
        
        /// <summary>
        /// Helper method to check component health with error handling.
        /// </summary>
        private async Task CheckComponentHealthWithErrorHandlingAsync(string componentName, SystemHealth systemHealth)
        {
            try
            {
                systemHealth.Components[componentName] = await CheckComponentHealthAsync(componentName).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                systemHealth.Components[componentName] = new ComponentHealth
                {
                    Name = componentName,
                    Status = HealthStatus.Unknown,
                    Description = $"Failed to check {componentName.ToLowerInvariant()} health",
                    Error = ex.Message,
                    CheckedAt = DateTime.Now
                };
            }
        }
    }
}