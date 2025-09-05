using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.Diagnostics;
using System.Linq;
using System.Net.NetworkInformation;
using System.Threading;
using System.Threading.Tasks;
using Quantra.CrossCutting;
using Quantra.CrossCutting.Logging;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.CrossCutting.Monitoring;
using Quantra.CrossCutting.Monitoring.Models;
using System.Text.Json;
using Quantra.DAL.Services.Interfaces;
using Quantra.Utilities;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service that monitors system health and emits alerts for technical issues
    /// including connectivity disruptions, data retrieval problems, or resource constraints.
    /// </summary>
    public class SystemHealthMonitorService : IDisposable
    {
        private readonly Timer _monitorTimer;
        private readonly Timer _predictiveAnalysisTimer;
        private readonly RealTimeInferenceService _inferenceService;
        private readonly IApiConnectivityService _apiConnectivityService;
        private readonly int _checkIntervalSeconds;
        private readonly int _memoryThresholdMb;
        private readonly int _cpuThresholdPercent;
        private readonly int _diskSpaceThresholdMb;
        private readonly ILogger _logger;
        private bool _disposed = false;

        // Tracking the last time an alert was sent for each issue to avoid spam
        private readonly Dictionary<string, DateTime> _lastAlertTimes = new Dictionary<string, DateTime>();
        // Minimum time between repeat alerts for the same issue
        private readonly TimeSpan _alertCooldown = TimeSpan.FromMinutes(15);
        // How often to run predictive analysis (less frequently than regular health checks)
        private readonly TimeSpan _predictiveAnalysisInterval = TimeSpan.FromMinutes(15);

        public SystemHealthMonitorService(
            RealTimeInferenceService inferenceService = null,
            IApiConnectivityService apiConnectivityService = null,
            int checkIntervalSeconds = 60,
            int memoryThresholdMb = 1024,
            int cpuThresholdPercent = 80,
            int diskSpaceThresholdMb = 1000)
        {
            // Ensure cross-cutting concerns are initialized
            CrossCuttingRegistry.Initialize();
            
            _inferenceService = inferenceService;
            _apiConnectivityService = apiConnectivityService;
            _checkIntervalSeconds = checkIntervalSeconds;
            _memoryThresholdMb = memoryThresholdMb;
            _cpuThresholdPercent = cpuThresholdPercent;
            _diskSpaceThresholdMb = diskSpaceThresholdMb;
            _logger = Log.ForType<SystemHealthMonitorService>();

            // Start monitoring after initialization
            _monitorTimer = new Timer(
                callback: async (state) => await CheckSystemHealthAsync(),
                state: null,
                dueTime: TimeSpan.FromSeconds(5), // Initial delay
                period: TimeSpan.FromSeconds(_checkIntervalSeconds)
            );
            
            // Start predictive analysis on a less frequent schedule
            _predictiveAnalysisTimer = new Timer(
                callback: async (state) => await PerformPredictiveAnalysisAsync(),
                state: null,
                dueTime: TimeSpan.FromMinutes(1), // Initial delay
                period: _predictiveAnalysisInterval
            );
            
            _logger.Information("SystemHealthMonitorService initialized with check interval {CheckIntervalSeconds}s", 
                _checkIntervalSeconds);
        }

        public async Task CheckSystemHealthAsync()
        {
            try
            {
                using var operation = _logger.BeginTimedOperation("SystemHealthCheck");
                
                // Use the new monitoring system to get overall health
                var systemHealth = await Performance.CheckSystemHealthAsync();
                
                // If system is not healthy, raise an alert
                if (systemHealth.Status != HealthStatus.Healthy)
                {
                    EmitSystemHealthAlert(
                        "System Health Degraded",
                        $"System health status: {systemHealth.Status}",
                        "Multiple component issues detected. See details in health dashboard.",
                        systemHealth.Status == HealthStatus.Unhealthy ? 3 : 2);
                }
                
                // Also run legacy checks for backward compatibility
                var tasks = new List<Task>
                {
                    CheckDatabaseConnectivityAsync(),
                    CheckDataIntegrityAsync(),
                    CheckResourceUtilizationAsync()
                };

                // Add optional checks if services are available
                if (_inferenceService != null)
                {
                    tasks.Add(CheckMlInferenceServiceAsync());
                }

                if (_apiConnectivityService != null)
                {
                    tasks.Add(CheckApiConnectivityAsync());
                }

                await Task.WhenAll(tasks);
            }
            catch (Exception ex)
            {
                // Log exception but don't cause further issues
                _logger.Error(ex, "System health monitor failed");
            }
        }

        /// <summary>
        /// Checks if database connections are working properly
        /// </summary>
        private async Task CheckDatabaseConnectivityAsync()
        {
            try
            {
                // Use resilience patterns for the database check
                await ResilienceHelper.RetryAsync(async () => 
                {
                    using var connection = new SQLiteConnection(DatabaseMonolith.ConnectionString);
                    await connection.OpenAsync();

                    // Quick query to check connection is working
                    using var command = connection.CreateCommand();
                    command.CommandText = "SELECT 1";
                    await command.ExecuteScalarAsync();
                    
                    Performance.RecordSuccess("DatabaseConnectivity");
                }, 
                RetryOptions.ForUserFacingOperation());
            }
            catch (Exception ex)
            {
                Performance.RecordFailure("DatabaseConnectivity", ex);
                
                EmitSystemHealthAlert(
                    "Database Connectivity Issue",
                    "Unable to connect to the database or execute queries",
                    ex.ToString(),
                    3); // High priority
            }
        }

        /// <summary>
        /// Checks the health of the ML inference service
        /// </summary>
        private async Task CheckMlInferenceServiceAsync()
        {
            try
            {
                // Use performance tracking for ML health checks
                bool isInferenceHealthy = await Performance.TimeAsync(
                    "MLInferenceHealthCheck", 
                    () => _inferenceService.HealthCheckAsync()
                );
                
                if (!isInferenceHealthy)
                {
                    // Get metrics to understand the issue better
                    var metrics = _inferenceService.GetPerformanceMetrics();
                    string details = $"Error Rate: {metrics.ErrorRate:P2}\n" +
                                    $"Avg Inference Time: {metrics.AverageInferenceTimeMs:F2}ms\n" +
                                    $"Requests Per Minute: {metrics.RequestsPerMinute}";

                    EmitSystemHealthAlert(
                        "ML Inference Service Issue",
                        "The prediction engine is not responding properly",
                        details,
                        2); // Medium priority
                }
                else
                {
                    // Occasionally check performance metrics even when healthy
                    var metrics = _inferenceService.GetPerformanceMetrics();
                    
                    // Track these metrics in our new monitoring system
                    Performance.RecordMetric("ML.ErrorRate", metrics.ErrorRate);
                    Performance.RecordMetric("ML.AvgInferenceTimeMs", metrics.AverageInferenceTimeMs);
                    Performance.RecordMetric("ML.RequestsPerMinute", metrics.RequestsPerMinute);
                    
                    if (metrics.ErrorRate > 0.20) // 20% error rate threshold
                    {
                        string details = $"Error Rate: {metrics.ErrorRate:P2}\n" +
                                        $"Avg Inference Time: {metrics.AverageInferenceTimeMs:F2}ms\n" +
                                        $"Requests Per Minute: {metrics.RequestsPerMinute}";

                        EmitSystemHealthAlert(
                            "ML Inference Performance Degradation",
                            "The prediction engine has an elevated error rate",
                            details,
                            2); // Medium priority
                    }
                    else if (metrics.AverageInferenceTimeMs > 500) // 500ms threshold
                    {
                        string details = $"Error Rate: {metrics.ErrorRate:P2}\n" +
                                        $"Avg Inference Time: {metrics.AverageInferenceTimeMs:F2}ms\n" +
                                        $"P95 Inference Time: {metrics.P95InferenceTimeMs:F2}ms";

                        EmitSystemHealthAlert(
                            "ML Inference Latency Issue",
                            "The prediction engine is experiencing high latency",
                            details,
                            1); // Low priority
                    }
                }
            }
            catch (Exception ex)
            {
                Performance.RecordFailure("MLInferenceHealthCheck", ex);
                
                EmitSystemHealthAlert(
                    "ML Service Monitoring Error",
                    "Failed to check ML inference service health",
                    ex.ToString(),
                    2); // Medium priority
            }
        }

        /// <summary>
        /// Checks API connectivity to external data providers
        /// </summary>
        private async Task CheckApiConnectivityAsync()
        {
            try
            {
                // Use circuit breaker for API connectivity checks
                var apiStatus = await ResilienceHelper.WithCircuitBreakerAsync(
                    "APIsConnectivity", 
                    () => _apiConnectivityService.CheckConnectivityAsync()
                );
                
                Performance.RecordMetric("API.Connected", apiStatus.IsConnected ? 1 : 0);
                
                if (!apiStatus.IsConnected)
                {
                    Performance.RecordFailure("APIConnectivity", null, 
                        new Dictionary<string, string> { ["ApiName"] = apiStatus.ApiName });
                        
                    EmitSystemHealthAlert(
                        "API Connectivity Issue",
                        $"Cannot connect to {apiStatus.ApiName}",
                        $"Status: {apiStatus.StatusMessage}\nLast Success: {apiStatus.LastSuccessfulConnection}",
                        2); // Medium priority
                }
                else
                {
                    Performance.RecordSuccess("APIConnectivity");
                }
            }
            catch (Exception ex)
            {
                Performance.RecordFailure("APIConnectivity", ex);
                
                EmitSystemHealthAlert(
                    "API Monitoring Error",
                    "Failed to check API connectivity",
                    ex.ToString(),
                    2); // Medium priority
            }
        }

        /// <summary>
        /// Checks data integrity and cache validity
        /// </summary>
        private async Task CheckDataIntegrityAsync()
        {
            try
            {
                // Check if stock symbol cache is valid
                if (!DatabaseMonolith.IsSymbolCacheValid())
                {
                    Performance.RecordFailure("SymbolCacheValidity");
                    
                    EmitSystemHealthAlert(
                        "Symbol Cache Invalid",
                        "Stock symbol cache requires updating",
                        "The stock symbol cache is too old or incomplete",
                        1); // Low priority
                }
                else
                {
                    Performance.RecordSuccess("SymbolCacheValidity");
                }

                // We could add more data integrity checks here
                // such as for historical price data, etc.
            }
            catch (Exception ex)
            {
                Performance.RecordFailure("DataIntegrityCheck", ex);
                
                EmitSystemHealthAlert(
                    "Data Integrity Check Error",
                    "Failed to validate data integrity",
                    ex.ToString(),
                    2); // Medium priority
            }
            
            await Task.CompletedTask; // For async signature
        }

        /// <summary>
        /// Checks system resource utilization
        /// </summary>
        private async Task CheckResourceUtilizationAsync()
        {
            try
            {
                // Use our new monitoring system to get resource utilization
                var resources = Performance.GetResourceUtilization();
                
                // Register metrics in our monitoring system
                Performance.RecordMetric("System.CpuUsagePercent", resources.CpuUsagePercent);
                Performance.RecordMetric("System.MemoryUsageMB", resources.MemoryUsageBytes / (1024 * 1024));
                Performance.RecordMetric("System.DiskUsagePercent", resources.DiskUsagePercent);
                
                // Memory check
                var memoryUsageMb = resources.MemoryUsageBytes / (1024 * 1024);
                
                if (memoryUsageMb > _memoryThresholdMb)
                {
                    EmitSystemHealthAlert(
                        "High Memory Usage",
                        $"Application memory usage exceeds threshold ({memoryUsageMb}MB / {_memoryThresholdMb}MB)",
                        $"Current: {memoryUsageMb}MB\nThreshold: {_memoryThresholdMb}MB",
                        2); // Medium priority
                }
                
                // Network connectivity check
                using (Ping ping = new Ping())
                {
                    var result = await ping.SendPingAsync("8.8.8.8", 3000);
                    
                    Performance.RecordMetric("Network.PingLatencyMs", result.Status == IPStatus.Success ? result.RoundtripTime : -1);
                    
                    if (result.Status != IPStatus.Success)
                    {
                        EmitSystemHealthAlert(
                            "Network Connectivity Issue",
                            "Cannot reach internet (ping to 8.8.8.8 failed)",
                            $"Status: {result.Status}\nRoundtrip time: {result.RoundtripTime}ms",
                            3); // High priority
                    }
                }
            }
            catch (Exception ex)
            {
                // Only log this internally to avoid circular alerts
                _logger.Error(ex, "Resource utilization check failed");
            }
        }

        /// <summary>
        /// Emits a system health alert with cooldown to avoid spam
        /// </summary>
        private void EmitSystemHealthAlert(string name, string condition, string details, int priority)
        {
            try
            {
                // Check if we've recently sent this alert and should be in cooldown
                string alertKey = $"{name}_{condition}";
                if (_lastAlertTimes.TryGetValue(alertKey, out DateTime lastAlertTime))
                {
                    // If alert was sent recently, don't send again
                    if (DateTime.Now - lastAlertTime < _alertCooldown)
                    {
                        return;
                    }
                }

                // Create and emit the alert
                var alert = new AlertModel
                {
                    Name = name,
                    Condition = condition,
                    AlertType = "System Health",
                    IsActive = true,
                    IsTriggered = true,
                    Priority = priority,
                    CreatedDate = DateTime.Now,
                    TriggeredDate = DateTime.Now,
                    Category = AlertCategory.SystemHealth,
                    Notes = details
                };

                Alerting.EmitGlobalAlert(alert);

                // Update the last alert time
                _lastAlertTimes[alertKey] = DateTime.Now;
                
                // Log the alert using the new logging system
                _logger.ForContext("AlertName", name)
                       .ForContext("AlertPriority", priority)
                       .Warning("System health alert emitted: {AlertCondition}", condition);
            }
            catch (Exception ex)
            {
                // Log to new logging system
                _logger.Error(ex, "Failed to emit system health alert");
                
                // Also log to database for backward compatibility
                DatabaseMonolith.Log("Error", "Failed to emit system health alert", ex.ToString());
            }
        }
        
        /// <summary>
        /// Performs predictive analysis to forecast potential system issues before they occur.
        /// </summary>
        private async Task PerformPredictiveAnalysisAsync()
        {
            try
            {
                _logger.Information("Starting predictive analysis for potential system issues");
                
                using var operation = _logger.BeginTimedOperation("PredictiveAnalysis");
                
                // Record current metrics for time-series data
                var resources = Performance.GetResourceUtilization();
                
                // Record system metrics for predictive analysis
                Performance.RecordTimeSeriesDataPoint("System.CpuUsagePercent", resources.CpuUsagePercent);
                Performance.RecordTimeSeriesDataPoint("System.MemoryUsageMB", resources.MemoryUsageBytes / (1024 * 1024));
                Performance.RecordTimeSeriesDataPoint("System.DiskUsagePercent", resources.DiskUsagePercent);
                
                // Get metrics from the inference service if available
                if (_inferenceService != null)
                {
                    var metrics = _inferenceService.GetPerformanceMetrics();
                    Performance.RecordTimeSeriesDataPoint("ML.ErrorRate", metrics.ErrorRate);
                    Performance.RecordTimeSeriesDataPoint("ML.AvgInferenceTimeMs", metrics.AverageInferenceTimeMs);
                    Performance.RecordTimeSeriesDataPoint("ML.RequestsPerMinute", metrics.RequestsPerMinute);
                }
                
                // Define thresholds for alerts
                var thresholds = new Dictionary<string, double>
                {
                    ["System.CpuUsagePercent"] = _cpuThresholdPercent,
                    ["System.MemoryUsageMB"] = _memoryThresholdMb,
                    ["System.DiskUsagePercent"] = 90, // 90% disk usage
                    ["ML.ErrorRate"] = 0.2, // 20% error rate
                    ["ML.AvgInferenceTimeMs"] = 500 // 500ms inference time
                };
                
                // Perform predictive analysis with different time horizons
                var shortTermResult = Performance.PerformPredictiveAnalysis(
                    predictionHorizon: TimeSpan.FromMinutes(15),
                    analysisTimeWindow: TimeSpan.FromHours(6),
                    thresholds: thresholds
                );
                
                var mediumTermResult = Performance.PerformPredictiveAnalysis(
                    predictionHorizon: TimeSpan.FromHours(1),
                    analysisTimeWindow: TimeSpan.FromDays(1),
                    thresholds: thresholds
                );
                
                // Process and emit alerts for predictions
                await ProcessPredictiveAlertsAsync(shortTermResult);
                await ProcessPredictiveAlertsAsync(mediumTermResult);
                
                // Try to detect anomalies in key metrics using advanced algorithms
                await DetectMetricAnomaliesAsync();
                
                _logger.Information("Predictive analysis completed with {ShortTermAlerts} short-term and {MediumTermAlerts} medium-term alerts", 
                    shortTermResult.Alerts.Count, mediumTermResult.Alerts.Count);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error performing predictive analysis");
            }
        }
        
        /// <summary>
        /// Processes predictive alerts and emits system alerts for predicted issues.
        /// </summary>
        private async Task ProcessPredictiveAlertsAsync(PredictiveAnalysisResult result)
        {
            if (result?.Alerts == null || result.Alerts.Count == 0)
            {
                return;
            }
            
            foreach (var alert in result.Alerts)
            {
                try
                {
                    // Skip alerts with low confidence
                    if (alert.Confidence < 0.7)
                    {
                        continue;
                    }
                    
                    // Create user-friendly description
                    var timeToIssue = alert.TimeToIssue.TotalMinutes > 60 
                        ? $"{alert.TimeToIssue.TotalHours:F1} hours" 
                        : $"{alert.TimeToIssue.TotalMinutes:F0} minutes";
                        
                    string description = $"PREDICTIVE ALERT: {alert.Description}\n\n" +
                                        $"Current Value: {alert.CurrentValue:F2}\n" +
                                        $"Predicted Value: {alert.PredictedValue:F2}\n" +
                                        $"Threshold: {alert.Threshold:F2}\n" +
                                        $"Time to Issue: {timeToIssue}\n" +
                                        $"Confidence: {alert.Confidence:P0}";
                                        
                    // Add suggested actions
                    if (alert.SuggestedActions.Count > 0)
                    {
                        description += "\n\nSuggested Actions:\n";
                        for (int i = 0; i < alert.SuggestedActions.Count; i++)
                        {
                            description += $"{i+1}. {alert.SuggestedActions[i]}\n";
                        }
                    }
                    
                    // Map severity level
                    int priority = (int)alert.Severity;
                    if (priority > 3) priority = 3; // Max priority is 3 in our alert system
                    
                    // Emit the alert
                    EmitSystemHealthAlert(
                        $"Predicted {alert.MetricName} Issue",
                        $"Potential issue predicted in {timeToIssue}",
                        description,
                        priority
                    );
                }
                catch (Exception ex)
                {
                    _logger.Error(ex, "Error processing predictive alert for {MetricName}", alert.MetricName);
                }
            }
            
            await Task.CompletedTask; // For async signature
        }
        
        /// <summary>
        /// Detects anomalies in key metrics using advanced anomaly detection algorithms.
        /// </summary>
        private async Task DetectMetricAnomaliesAsync()
        {
            try
            {
                // Key metrics to analyze for anomalies
                var metricsToAnalyze = new[]
                {
                    "System.CpuUsagePercent",
                    "System.MemoryUsageMB",
                    "System.DiskUsagePercent",
                    "ML.ErrorRate",
                    "ML.AvgInferenceTimeMs"
                };
                
                foreach (var metricName in metricsToAnalyze)
                {
                    try
                    {
                        var anomalyResult = await Performance.DetectAnomaliesAsync(
                            metricName, 
                            TimeSpan.FromDays(3)
                        );
                        
                        // Skip if no result was returned
                        // Fix: Check if the object reference is null (a proper object reference check)
                        // instead of trying to compare a JsonElement with null
                        if (object.ReferenceEquals(anomalyResult, null))
                        {
                            continue;
                        }
                        
                        // Check for anomalies - first verify the property exists and then check its value
                        if (anomalyResult.TryGetProperty("anomalies_detected", out JsonElement anomaliesDetectedElement) && 
                            anomaliesDetectedElement.GetBoolean())
                        {
                            // Extract insights from the anomaly detection
                            string insights = "No detailed insights available.";
                            
                            if (anomalyResult.TryGetProperty("recent_anomalies", out JsonElement recentAnomalies) &&
                                recentAnomalies.GetArrayLength() > 0)
                            {
                                var anomaly = recentAnomalies[0];
                                
                                // Extract anomaly types if available
                                if (anomaly.TryGetProperty("types", out JsonElement typesElement))
                                {
                                    var anomalyTypes = string.Join(", ", typesElement.EnumerateArray()
                                        .Select(t => t.GetString())
                                        .Where(t => !string.IsNullOrEmpty(t)));
                                    
                                    insights = $"Anomaly Types: {anomalyTypes}\n\n";
                                }
                                
                                // Try to extract insights
                                if (anomaly.TryGetProperty("insights", out JsonElement insightData))
                                {
                                    // Extract descriptions
                                    if (insightData.TryGetProperty("description", out JsonElement descriptions) &&
                                        descriptions.GetArrayLength() > 0)
                                    {
                                        insights += "Description:\n";
                                        foreach (var desc in descriptions.EnumerateArray())
                                        {
                                            insights += $"- {desc.GetString()}\n";
                                        }
                                    }
                                    
                                    // Extract potential causes
                                    if (insightData.TryGetProperty("potential_causes", out JsonElement causes) &&
                                        causes.GetArrayLength() > 0)
                                    {
                                        insights += "\nPotential Causes:\n";
                                        foreach (var cause in causes.EnumerateArray())
                                        {
                                            insights += $"- {cause.GetString()}\n";
                                        }
                                    }
                                    
                                    // Extract suggested actions
                                    if (insightData.TryGetProperty("suggested_actions", out JsonElement actions) &&
                                        actions.GetArrayLength() > 0)
                                    {
                                        insights += "\nSuggested Actions:\n";
                                        foreach (var action in actions.EnumerateArray())
                                        {
                                            insights += $"- {action.GetString()}\n";
                                        }
                                    }
                                }
                            }
                            
                            EmitSystemHealthAlert(
                                $"Anomaly Detected in {metricName}",
                                "Unusual pattern detected in metric behavior",
                                $"The anomaly detection system has identified unusual behavior in {metricName}.\n\n{insights}",
                                2 // Medium priority
                            );
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.Warning(ex.Message, "Error detecting anomalies for metric {MetricName}", metricName);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error in anomaly detection");
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            
            _monitorTimer?.Dispose();
            _predictiveAnalysisTimer?.Dispose();
            _disposed = true;
            
            _logger.Information("SystemHealthMonitorService disposed");
        }
    }
}