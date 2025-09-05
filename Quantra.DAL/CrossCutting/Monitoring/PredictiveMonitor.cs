using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.Text.Json;
using Quantra.CrossCutting.Logging;
using Quantra.CrossCutting.Monitoring.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Interfaces;

namespace Quantra.CrossCutting.Monitoring
{
    /// <summary>
    /// Service that provides predictive monitoring capabilities by analyzing
    /// time-series metric data to forecast potential issues before they occur.
    /// </summary>
    public class PredictiveMonitor
    {
        private readonly ILogger _logger;
        private readonly IMonitoringManager _monitoringManager;
        private readonly Dictionary<string, List<TimeSeriesMetricPoint>> _metricHistory;
        private readonly object _metricHistoryLock = new object(); // Lock for thread-safe access
        private readonly int _maxHistoryPerMetric = 1000; // Maximum historical data points per metric
        private readonly string _metricsStoragePath;
        private readonly bool _enablePersistence;
        private readonly IGlobalLoadingStateService _globalLoadingStateService;

        /// <summary>
        /// Initializes a new instance of the <see cref="PredictiveMonitor"/> class.
        /// </summary>
        /// <param name="monitoringManager">The monitoring manager.</param>
        /// <param name="enablePersistence">Whether to enable persistence of metrics history.</param>
        /// <param name="metricsStoragePath">Path where to store metrics history (optional).</param>
        public PredictiveMonitor(
            IMonitoringManager monitoringManager,
            bool enablePersistence = true,
            string metricsStoragePath = null)
        {
            _monitoringManager = monitoringManager ?? throw new ArgumentNullException(nameof(monitoringManager));
            _logger = Log.ForType<PredictiveMonitor>();
            _metricHistory = new Dictionary<string, List<TimeSeriesMetricPoint>>();
            _enablePersistence = enablePersistence;
            
            // Set default storage path if not provided
            _metricsStoragePath = metricsStoragePath ?? 
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "MetricsHistory");
            
            if (_enablePersistence && !Directory.Exists(_metricsStoragePath))
            {
                Directory.CreateDirectory(_metricsStoragePath);
            }
            
            LoadHistoricalData();
            
            _logger.Information("PredictiveMonitor initialized with storage path: {StoragePath}", _metricsStoragePath);
        }
        
        /// <summary>
        /// Records a metric data point for time-series analysis.
        /// </summary>
        /// <param name="metricName">Name of the metric.</param>
        /// <param name="value">Value of the metric.</param>
        /// <param name="dimensions">Additional dimensions for the metric.</param>
        /// <param name="timestamp">Timestamp for the metric (defaults to now).</param>
        public void RecordMetricDataPoint(
            string metricName, 
            double value, 
            IDictionary<string, string> dimensions = null,
            DateTime? timestamp = null)
        {
            if (string.IsNullOrEmpty(metricName))
            {
                throw new ArgumentNullException(nameof(metricName));
            }
            
            try
            {
                var metricTimestamp = timestamp ?? DateTime.Now;
                
                var dataPoint = new TimeSeriesMetricPoint
                {
                    Timestamp = metricTimestamp,
                    Value = value,
                    Dimensions = dimensions?.ToDictionary(kvp => kvp.Key, kvp => kvp.Value)
                };
                
                bool shouldPersist = false;
                
                lock (_metricHistoryLock)
                {
                    // Ensure the list exists
                    if (!_metricHistory.ContainsKey(metricName))
                    {
                        _metricHistory[metricName] = new List<TimeSeriesMetricPoint>();
                    }
                    
                    // Add the data point
                    _metricHistory[metricName].Add(dataPoint);
                    
                    // Trim the history if it gets too long
                    if (_metricHistory[metricName].Count > _maxHistoryPerMetric)
                    {
                        _metricHistory[metricName].RemoveAt(0);
                    }
                    
                    // Check if we should persist (but don't do it inside the lock)
                    shouldPersist = _enablePersistence && _metricHistory[metricName].Count % 100 == 0;
                }
                
                // Periodically persist metrics (outside the lock to avoid blocking)
                if (shouldPersist)
                {
                    _globalLoadingStateService.WithLoadingState(Task.Run(() => PersistMetricHistory(metricName)))
                        .ContinueWith(task =>
                        {
                            if (task.IsFaulted && task.Exception != null)
                            {
                                _logger.Error(task.Exception, "Failed to persist metric history for {MetricName}", metricName);
                            }
                        });
                }
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to record metric data point for {MetricName}", metricName);
            }
        }
        
        /// <summary>
        /// Gets the time-series data for a specific metric.
        /// </summary>
        /// <param name="metricName">Name of the metric.</param>
        /// <param name="timeWindow">Time window to retrieve (null for all available data).</param>
        /// <returns>A collection of time-series data points.</returns>
        public IEnumerable<TimeSeriesMetricPoint> GetMetricTimeSeries(string metricName, TimeSpan? timeWindow = null)
        {
            if (string.IsNullOrEmpty(metricName))
            {
                throw new ArgumentNullException(nameof(metricName));
            }
            
            List<TimeSeriesMetricPoint> points;
            lock (_metricHistoryLock)
            {
                if (!_metricHistory.ContainsKey(metricName))
                {
                    return new List<TimeSeriesMetricPoint>();
                }
                
                // Create a copy to avoid external modifications
                points = new List<TimeSeriesMetricPoint>(_metricHistory[metricName]);
            }
            
            if (timeWindow.HasValue)
            {
                var cutoffTime = DateTime.Now - timeWindow.Value;
                points = points.Where(p => p.Timestamp >= cutoffTime).ToList();
            }
            
            return points;
        }
        
        /// <summary>
        /// Gets metrics with unusual patterns or trends using basic statistical analysis.
        /// </summary>
        /// <param name="analysisTimeWindow">Time window to analyze.</param>
        /// <param name="zScoreThreshold">Z-score threshold for anomaly detection.</param>
        /// <returns>Dictionary of metrics with their anomaly scores.</returns>
        public Dictionary<string, double> GetAnomalousMetrics(
            TimeSpan analysisTimeWindow, 
            double zScoreThreshold = 3.0)
        {
            var result = new Dictionary<string, double>();
            var cutoffTime = DateTime.Now - analysisTimeWindow;
            
            Dictionary<string, List<TimeSeriesMetricPoint>> metricHistoryCopy;
            lock (_metricHistoryLock)
            {
                // Create a copy of the dictionary to avoid holding the lock during analysis
                metricHistoryCopy = new Dictionary<string, List<TimeSeriesMetricPoint>>();
                foreach (var kvp in _metricHistory)
                {
                    metricHistoryCopy[kvp.Key] = new List<TimeSeriesMetricPoint>(kvp.Value);
                }
            }
            
            foreach (var metric in metricHistoryCopy)
            {
                try
                {
                    var recentPoints = metric.Value
                        .Where(p => p.Timestamp >= cutoffTime)
                        .OrderBy(p => p.Timestamp)
                        .ToList();
                    
                    if (recentPoints.Count < 5) // Need enough data points
                    {
                        continue;
                    }
                    
                    // Calculate basic statistics
                    var values = recentPoints.Select(p => p.Value).ToArray();
                    var mean = values.Average();
                    var stdDev = Math.Sqrt(values.Select(v => Math.Pow(v - mean, 2)).Sum() / values.Length);
                    
                    if (stdDev < 0.0000001) // Avoid division by zero
                    {
                        continue;
                    }
                    
                    // Get the most recent value and calculate its z-score
                    var latestValue = recentPoints.Last().Value;
                    var zScore = Math.Abs(latestValue - mean) / stdDev;
                    
                    // Add to result if it exceeds the threshold
                    if (zScore > zScoreThreshold)
                    {
                        result[metric.Key] = zScore;
                    }
                }
                catch (Exception ex)
                {
                    _logger.Error(ex, "Error analyzing metric {MetricName}", metric.Key);
                }
            }
            
            return result;
        }
        
        /// <summary>
        /// Analyzes trends in metrics over the specified time period.
        /// </summary>
        /// <param name="analysisPeriod">The period over which to analyze trends.</param>
        /// <returns>List of metric trends.</returns>
        public List<MetricTrend> AnalyzeMetricTrends(TimeSpan analysisPeriod)
        {
            var trends = new List<MetricTrend>();
            var cutoffTime = DateTime.Now - analysisPeriod;
            
            Dictionary<string, List<TimeSeriesMetricPoint>> metricHistoryCopy;
            lock (_metricHistoryLock)
            {
                // Create a copy of the dictionary to avoid holding the lock during analysis
                metricHistoryCopy = new Dictionary<string, List<TimeSeriesMetricPoint>>();
                foreach (var kvp in _metricHistory)
                {
                    metricHistoryCopy[kvp.Key] = new List<TimeSeriesMetricPoint>(kvp.Value);
                }
            }
            
            foreach (var metricEntry in metricHistoryCopy)
            {
                try
                {
                    var metricName = metricEntry.Key;
                    var points = metricEntry.Value
                        .Where(p => p.Timestamp >= cutoffTime)
                        .OrderBy(p => p.Timestamp)
                        .ToList();
                    
                    if (points.Count < 2) // Need at least 2 points to analyze trend
                    {
                        continue;
                    }
                    
                    // Simple linear regression
                    var n = points.Count;
                    var timestamps = points.Select(p => p.Timestamp.Ticks / (double)TimeSpan.TicksPerHour).ToList();
                    var values = points.Select(p => p.Value).ToList();
                    
                    double sumX = timestamps.Sum();
                    double sumY = values.Sum();
                    double sumXY = timestamps.Zip(values, (x, y) => x * y).Sum();
                    double sumX2 = timestamps.Sum(x => x * x);
                    
                    // Linear regression formula: y = slope * x + intercept
                    double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
                    double intercept = (sumY - slope * sumX) / n;
                    
                    // Calculate percentage change
                    double firstValue = points.First().Value;
                    double lastValue = points.Last().Value;
                    
                    double percentChange = firstValue != 0 ? 
                        ((lastValue - firstValue) / Math.Abs(firstValue)) * 100 : 0;
                    
                    // Determine trend direction
                    TrendDirection direction;
                    if (Math.Abs(percentChange) < 2) // Less than 2% change is considered stable
                    {
                        direction = TrendDirection.Stable;
                    }
                    else if (percentChange > 0)
                    {
                        direction = TrendDirection.Increasing;
                    }
                    else
                    {
                        direction = TrendDirection.Decreasing;
                    }
                    
                    // Create trend object
                    var trend = new MetricTrend
                    {
                        MetricName = metricName,
                        Direction = direction,
                        PercentChange = percentChange,
                        RateOfChange = slope, // Units per hour
                        AnalysisPeriod = analysisPeriod,
                        AnalyzedAt = DateTime.Now
                    };
                    
                    trends.Add(trend);
                }
                catch (Exception ex)
                {
                    _logger.Error(ex, "Error analyzing trends for metric {MetricName}", metricEntry.Key);
                }
            }
            
            return trends;
        }
        
        /// <summary>
        /// Predicts future values for a specified metric based on historical data.
        /// </summary>
        /// <param name="metricName">Name of the metric to predict.</param>
        /// <param name="predictionHorizon">How far into the future to predict.</param>
        /// <param name="historyWindow">How much historical data to use (null for all available).</param>
        /// <returns>Prediction for the metric or null if prediction failed.</returns>
        public MetricPrediction PredictMetricValue(
            string metricName, 
            TimeSpan predictionHorizon, 
            TimeSpan? historyWindow = null)
        {
            if (string.IsNullOrEmpty(metricName))
            {
                throw new ArgumentNullException(nameof(metricName));
            }
            
            try
            {
                List<TimeSeriesMetricPoint> dataPoints;
                lock (_metricHistoryLock)
                {
                    if (!_metricHistory.ContainsKey(metricName) || _metricHistory[metricName].Count < 5)
                    {
                        _logger.Debug("Not enough data to predict metric {MetricName}", metricName);
                        return null;
                    }
                    
                    // Create a copy of the data points to avoid holding the lock during processing
                    dataPoints = new List<TimeSeriesMetricPoint>(_metricHistory[metricName]);
                }
                
                // Filter data points based on history window
                if (historyWindow.HasValue)
                {
                    var cutoffTime = DateTime.Now - historyWindow.Value;
                    dataPoints = dataPoints.Where(p => p.Timestamp >= cutoffTime).ToList();
                }
                
                if (dataPoints.Count < 5)
                {
                    _logger.Debug("Not enough data points in history window for metric {MetricName}", metricName);
                    return null;
                }
                
                // Sort by timestamp
                dataPoints = dataPoints.OrderBy(p => p.Timestamp).ToList();
                
                // Extract time series values
                var timestamps = dataPoints.Select(p => p.Timestamp.Ticks / (double)TimeSpan.TicksPerHour).ToList();
                var values = dataPoints.Select(p => p.Value).ToList();
                
                // Simple linear regression for prediction
                var n = timestamps.Count;
                double sumX = timestamps.Sum();
                double sumY = values.Sum();
                double sumXY = timestamps.Zip(values, (x, y) => x * y).Sum();
                double sumX2 = timestamps.Sum(x => x * x);
                
                // Linear regression formula: y = slope * x + intercept
                double denominator = n * sumX2 - sumX * sumX;
                
                // Check if the denominator is zero (or very close to zero)
                if (Math.Abs(denominator) < 0.0000001)
                {
                    // If the line is vertical or data is constant, return the mean
                    double meanValue = values.Average();
                    return new MetricPrediction
                    {
                        MetricName = metricName,
                        PredictedValue = meanValue,
                        PredictionForTime = DateTime.Now + predictionHorizon,
                        PredictionCreatedAt = DateTime.Now,
                        Confidence = 0.5, // Medium confidence for constant data
                        LowerBound = meanValue,
                        UpperBound = meanValue,
                        AnomalyScore = 0
                    };
                }
                
                double slope = (n * sumXY - sumX * sumY) / denominator;
                double intercept = (sumY - slope * sumX) / n;
                
                // Calculate the prediction
                double futureTimepoint = (DateTime.Now + predictionHorizon).Ticks / (double)TimeSpan.TicksPerHour;
                double predictedValue = slope * futureTimepoint + intercept;
                
                // Calculate error measures
                var residuals = new List<double>();
                for (int i = 0; i < n; i++)
                {
                    double fitted = slope * timestamps[i] + intercept;
                    residuals.Add(values[i] - fitted);
                }
                
                double residualStdDev = Math.Sqrt(residuals.Select(r => r * r).Sum() / (n - 2));
                
                // Calculate confidence and prediction interval
                double confidence = 1.0 - (residualStdDev / Math.Abs(values.Average()));
                confidence = Math.Max(0, Math.Min(1, confidence)); // Clamp to [0, 1]
                
                // Simple prediction interval (approximate)
                double predictionInterval = residualStdDev * 1.96; // 95% confidence
                
                // Calculate anomaly score based on how much the prediction deviates from recent values
                double recentAvg = values.Skip(Math.Max(0, values.Count - 5)).Average();
                double anomalyScore = Math.Min(1, Math.Abs(predictedValue - recentAvg) / Math.Max(1, recentAvg));
                
                // Create the prediction object
                return new MetricPrediction
                {
                    MetricName = metricName,
                    PredictedValue = predictedValue,
                    PredictionForTime = DateTime.Now + predictionHorizon,
                    PredictionCreatedAt = DateTime.Now,
                    Confidence = confidence,
                    LowerBound = Math.Max(0, predictedValue - predictionInterval), // Ensure non-negative for metrics that can't be negative
                    UpperBound = predictedValue + predictionInterval,
                    AnomalyScore = anomalyScore
                };
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to predict metric value for {MetricName}", metricName);
                return null;
            }
        }
        
        /// <summary>
        /// Performs comprehensive predictive analysis on multiple metrics.
        /// </summary>
        /// <param name="predictionHorizon">How far into the future to predict.</param>
        /// <param name="analysisTimeWindow">Time window to use for analysis.</param>
        /// <param name="metricFilter">Optional filter to select specific metrics.</param>
        /// <param name="thresholds">Optional dictionary of thresholds for specific metrics.</param>
        /// <returns>Result of the predictive analysis.</returns>
        public PredictiveAnalysisResult PerformPredictiveAnalysis(
            TimeSpan predictionHorizon,
            TimeSpan analysisTimeWindow,
            Func<string, bool> metricFilter = null,
            Dictionary<string, double> thresholds = null)
        {
            _logger.Information("Starting predictive analysis with horizon {PredictionHorizon}", predictionHorizon);
            
            var result = new PredictiveAnalysisResult();
            
            try
            {
                // Filter metrics based on the provided filter
                List<string> metricsToAnalyze;
                lock (_metricHistoryLock)
                {
                    metricsToAnalyze = _metricHistory.Keys
                        .Where(m => metricFilter == null || metricFilter(m))
                        .ToList();
                }
                
                // Analyze trends first
                result.Trends = AnalyzeMetricTrends(analysisTimeWindow);
                
                // Predict values for each metric
                foreach (var metricName in metricsToAnalyze)
                {
                    var prediction = PredictMetricValue(metricName, predictionHorizon, analysisTimeWindow);
                    if (prediction != null)
                    {
                        result.Predictions.Add(prediction);
                        
                        // Get the most recent value for this metric
                        List<TimeSeriesMetricPoint> recentPoints;
                        lock (_metricHistoryLock)
                        {
                            if (!_metricHistory.ContainsKey(metricName))
                            {
                                continue;
                            }
                            recentPoints = _metricHistory[metricName]
                                .OrderByDescending(p => p.Timestamp)
                                .Take(1)
                                .ToList();
                        }
                        
                        if (recentPoints.Any())
                        {
                            double currentValue = recentPoints.First().Value;
                            double threshold = 0;
                            
                            // Determine threshold for alert
                            if (thresholds != null && thresholds.TryGetValue(metricName, out double specificThreshold))
                            {
                                threshold = specificThreshold;
                            }
                            else
                            {
                                // Use a default threshold based on metric type
                                if (metricName.Contains("CpuUsage") || metricName.Contains("Memory") || metricName.Contains("DiskUsage"))
                                {
                                    threshold = 90; // 90% for resource usage metrics
                                }
                                else if (metricName.Contains("ErrorRate"))
                                {
                                    threshold = 0.05; // 5% for error rates
                                }
                                else if (metricName.Contains("ResponseTime") || metricName.Contains("LatencyMs"))
                                {
                                    threshold = 1000; // 1000ms for response times
                                }
                            }
                            
                            // Check if the prediction exceeds the threshold and there's a concerning trend
                            var trend = result.Trends.FirstOrDefault(t => t.MetricName == metricName);
                            if (threshold > 0 && 
                                prediction.PredictedValue > threshold && 
                                prediction.Confidence > 0.6 &&
                                (trend == null || trend.Direction == TrendDirection.Increasing))
                            {
                                // Determine severity based on how much the threshold is exceeded and confidence
                                double severityFactor = (prediction.PredictedValue / threshold) * prediction.Confidence;
                                AlertSeverity severity = severityFactor >= 1.5 ? AlertSeverity.Critical :
                                                       severityFactor >= 1.2 ? AlertSeverity.High :
                                                       severityFactor >= 1.1 ? AlertSeverity.Medium :
                                                       AlertSeverity.Low;
                                
                                // Create a predictive alert
                                var alert = new PredictiveAlert
                                {
                                    Title = $"Predicted {metricName} Issue",
                                    Description = $"{metricName} is predicted to exceed threshold of {threshold} within {predictionHorizon.TotalMinutes} minutes",
                                    MetricName = metricName,
                                    CurrentValue = currentValue,
                                    PredictedValue = prediction.PredictedValue,
                                    Threshold = threshold,
                                    TimeToIssue = predictionHorizon,
                                    Confidence = prediction.Confidence,
                                    Severity = severity,
                                    ValidUntil = DateTime.Now + predictionHorizon + TimeSpan.FromHours(1)
                                };
                                
                                // Add suggested actions based on the metric type
                                if (metricName.Contains("CpuUsage"))
                                {
                                    alert.SuggestedActions.AddRange(new[] {
                                        "Check for CPU-intensive processes",
                                        "Consider scaling up CPU resources",
                                        "Optimize CPU-intensive operations"
                                    });
                                }
                                else if (metricName.Contains("Memory"))
                                {
                                    alert.SuggestedActions.AddRange(new[] {
                                        "Check for memory leaks",
                                        "Increase memory allocation",
                                        "Optimize memory-intensive operations"
                                    });
                                }
                                else if (metricName.Contains("DiskUsage"))
                                {
                                    alert.SuggestedActions.AddRange(new[] {
                                        "Free up disk space by removing unnecessary files",
                                        "Increase disk capacity",
                                        "Enable compression or archiving"
                                    });
                                }
                                else if (metricName.Contains("ErrorRate"))
                                {
                                    alert.SuggestedActions.AddRange(new[] {
                                        "Investigate error patterns in logs",
                                        "Check external dependencies",
                                        "Implement retry strategies"
                                    });
                                }
                                else if (metricName.Contains("ResponseTime") || metricName.Contains("LatencyMs"))
                                {
                                    alert.SuggestedActions.AddRange(new[] {
                                        "Check network connectivity",
                                        "Investigate database performance",
                                        "Review recent code changes affecting performance"
                                    });
                                }
                                
                                result.Alerts.Add(alert);
                            }
                        }
                    }
                }
                
                _logger.Information("Predictive analysis completed. Generated {AlertCount} alerts", result.Alerts.Count);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to perform predictive analysis");
            }
            
            return result;
        }
        
        /// <summary>
        /// Runs anomaly detection on metric time-series using the Python anomaly detection module.
        /// </summary>
        /// <param name="metricName">Name of the metric to analyze.</param>
        /// <param name="analysisTimeWindow">Time window to analyze.</param>
        /// <returns>A task containing the anomaly detection result.</returns>
        public async Task<dynamic> DetectAnomaliesAsync(string metricName, TimeSpan analysisTimeWindow)
        {
            if (string.IsNullOrEmpty(metricName))
            {
                throw new ArgumentNullException(nameof(metricName));
            }
            
            try
            {
                List<TimeSeriesMetricPoint> dataPoints;
                lock (_metricHistoryLock)
                {
                    if (!_metricHistory.ContainsKey(metricName) || _metricHistory[metricName].Count < 10)
                    {
                        return null;
                    }
                    
                    // Prepare data for the Python script
                    var cutoffTime = DateTime.Now - analysisTimeWindow;
                    dataPoints = _metricHistory[metricName]
                        .Where(p => p.Timestamp >= cutoffTime)
                        .OrderBy(p => p.Timestamp)
                        .ToList();
                }
                
                if (dataPoints.Count < 10) // Need sufficient data
                {
                    return null;
                }
                
                // Remove the redundant data preparation that was moved above
                
                // Convert to a format suitable for the anomaly detection script
                var timestamps = dataPoints.Select(p => p.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")).ToArray();
                var values = dataPoints.Select(p => p.Value).ToArray();
                
                // Prepare input for anomaly detection script
                var inputData = new
                {
                    data = new
                    {
                        timestamps,
                        values
                    },
                    metricName,
                    use_feature_engineering = true,
                    sensitivity = 1.0
                };
                
                // Create a temporary file to store results
                string tempInputFile = Path.Combine(_metricsStoragePath, $"anomaly_input_{Guid.NewGuid()}.json");
                string tempOutputFile = Path.Combine(_metricsStoragePath, $"anomaly_output_{Guid.NewGuid()}.json");
                
                try
                {
                    // Write input data to temp file
                    await File.WriteAllTextAsync(tempInputFile, JsonSerializer.Serialize(inputData));
                    
                    // Call Python anomaly detection script (assumes it's available in the path)
                    var pythonScript = Path.Combine(
                        AppDomain.CurrentDomain.BaseDirectory, 
                        "python", 
                        "anomaly_detection.py");
                    
                    if (!File.Exists(pythonScript))
                    {
                        _logger.Warning("Anomaly detection script not found at {ScriptPath}", pythonScript);
                        return null;
                    }
                    
                    // Execute the Python script
                    var startInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = $"\"{pythonScript}\" \"{tempInputFile}\" \"{tempOutputFile}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };
                    
                    using (var process = new System.Diagnostics.Process { StartInfo = startInfo })
                    {
                        process.Start();
                        string output = process.StandardOutput.ReadToEnd();
                        string error = process.StandardError.ReadToEnd();
                        process.WaitForExit();
                        
                        if (process.ExitCode != 0)
                        {
                            _logger.Error("Anomaly detection failed: {Error}", error);
                            return null;
                        }
                    }
                    
                    // Read the output file
                    if (File.Exists(tempOutputFile))
                    {
                        string resultJson = await File.ReadAllTextAsync(tempOutputFile);
                        var result = JsonSerializer.Deserialize<dynamic>(resultJson);
                        return result;
                    }
                }
                finally
                {
                    // Clean up temporary files
                    if (File.Exists(tempInputFile))
                    {
                        File.Delete(tempInputFile);
                    }
                    
                    if (File.Exists(tempOutputFile))
                    {
                        File.Delete(tempOutputFile);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to detect anomalies for metric {MetricName}", metricName);
            }
            
            return null;
        }
        
        /// <summary>
        /// Loads historical metric data from persistent storage.
        /// </summary>
        private void LoadHistoricalData()
        {
            if (!_enablePersistence || !Directory.Exists(_metricsStoragePath))
            {
                return;
            }
            
            try
            {
                var metricFiles = Directory.GetFiles(_metricsStoragePath, "metric_*.json");
                
                foreach (var file in metricFiles)
                {
                    try
                    {
                        var json = File.ReadAllText(file);
                        var timeSeries = JsonSerializer.Deserialize<MetricTimeSeries>(json);
                        
                        if (timeSeries != null && !string.IsNullOrEmpty(timeSeries.MetricName))
                        {
                            lock (_metricHistoryLock)
                            {
                                _metricHistory[timeSeries.MetricName] = timeSeries.DataPoints ?? new List<TimeSeriesMetricPoint>();
                            }
                            _logger.Debug("Loaded {Count} historical points for metric {MetricName}", 
                                timeSeries.DataPoints?.Count ?? 0, timeSeries.MetricName);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.Error(ex, "Failed to load metric data from {File}", file);
                    }
                }
                
                int metricCount;
                lock (_metricHistoryLock)
                {
                    metricCount = _metricHistory.Count;
                }
                _logger.Information("Loaded historical data for {Count} metrics", metricCount);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to load historical metrics data");
            }
        }
        
        /// <summary>
        /// Persists metric time-series data to storage.
        /// </summary>
        /// <param name="metricName">Name of the metric to persist.</param>
        private void PersistMetricHistory(string metricName)
        {
            if (!_enablePersistence || string.IsNullOrEmpty(metricName))
            {
                return;
            }
            
            try
            {
                MetricTimeSeries timeSeries;
                lock (_metricHistoryLock)
                {
                    if (!_metricHistory.ContainsKey(metricName))
                    {
                        return;
                    }
                    
                    timeSeries = new MetricTimeSeries
                    {
                        MetricName = metricName,
                        DataPoints = new List<TimeSeriesMetricPoint>(_metricHistory[metricName])
                    };
                }
                
                string json = JsonSerializer.Serialize(timeSeries);
                string fileName = Path.Combine(_metricsStoragePath, $"metric_{metricName.Replace('.', '_')}.json");
                
                File.WriteAllText(fileName, json);
                
                _logger.Debug("Persisted {Count} data points for metric {MetricName}", 
                    timeSeries.DataPoints.Count, metricName);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to persist metric history for {MetricName}", metricName);
            }
        }
        
        /// <summary>
        /// Persists all metric histories to storage.
        /// </summary>
        public void PersistAllMetricHistory()
        {
            if (!_enablePersistence)
            {
                return;
            }
            
            List<string> metricNames;
            lock (_metricHistoryLock)
            {
                metricNames = new List<string>(_metricHistory.Keys);
            }
            
            foreach (var metricName in metricNames)
            {
                PersistMetricHistory(metricName);
            }
            
            _logger.Information("Persisted all metric histories to {StoragePath}", _metricsStoragePath);
        }
    }
}