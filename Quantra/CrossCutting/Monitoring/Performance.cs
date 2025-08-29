using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.CrossCutting.Monitoring.Models;

namespace Quantra.CrossCutting.Monitoring
{
    /// <summary>
    /// Static helper for performance monitoring operations.
    /// </summary>
    public static class Performance
    {
        private static readonly IMonitoringManager _manager = MonitoringManager.Instance;
        
        /// <summary>
        /// Initializes the performance monitoring system.
        /// </summary>
        public static void Initialize()
        {
            _manager.Initialize();
        }
        
        /// <summary>
        /// Records a metric value.
        /// </summary>
        public static void RecordMetric(string name, double value, IDictionary<string, string> dimensions = null)
        {
            _manager.RecordMetric(name, value, dimensions);
        }
        
        /// <summary>
        /// Records the execution time of an operation.
        /// </summary>
        public static void Time(string operationName, Action action, IDictionary<string, string> dimensions = null)
        {
            _manager.RecordExecutionTime(operationName, action, dimensions);
        }
        
        /// <summary>
        /// Records the execution time of a function.
        /// </summary>
        public static T Time<T>(string operationName, Func<T> func, IDictionary<string, string> dimensions = null)
        {
            var (result, _) = _manager.RecordExecutionTime(operationName, func, dimensions);
            return result;
        }
        
        /// <summary>
        /// Records the execution time of an async action.
        /// </summary>
        public static Task TimeAsync(string operationName, Func<Task> action, IDictionary<string, string> dimensions = null)
        {
            return _manager.RecordExecutionTimeAsync(operationName, action, dimensions);
        }
        
        /// <summary>
        /// Records the execution time of an async function.
        /// </summary>
        public static async Task<T> TimeAsync<T>(string operationName, Func<Task<T>> func, IDictionary<string, string> dimensions = null)
        {
            var (result, _) = await _manager.RecordExecutionTimeAsync(operationName, func, dimensions);
            return result;
        }
        
        /// <summary>
        /// Records a successful operation.
        /// </summary>
        public static void RecordSuccess(string operationName, IDictionary<string, string> dimensions = null)
        {
            _manager.RecordSuccess(operationName, dimensions);
        }
        
        /// <summary>
        /// Records a failed operation.
        /// </summary>
        public static void RecordFailure(string operationName, Exception exception = null, IDictionary<string, string> dimensions = null)
        {
            _manager.RecordFailure(operationName, exception, dimensions);
        }
        
        /// <summary>
        /// Records memory usage at this point.
        /// </summary>
        public static void RecordMemoryUsage(string context = null)
        {
            _manager.RecordMemoryUsage(context);
        }
        
        /// <summary>
        /// Gets metrics for a specific operation.
        /// </summary>
        public static OperationMetrics GetMetrics(string operationName)
        {
            return _manager.GetMetrics(operationName);
        }
        
        /// <summary>
        /// Gets all recorded metrics.
        /// </summary>
        public static IReadOnlyDictionary<string, OperationMetrics> GetAllMetrics()
        {
            return _manager.GetAllMetrics();
        }
        
        /// <summary>
        /// Returns the health status of a specific component.
        /// </summary>
        public static Task<ComponentHealth> CheckComponentHealthAsync(string componentName)
        {
            return _manager.CheckComponentHealthAsync(componentName);
        }
        
        /// <summary>
        /// Returns the overall health status.
        /// </summary>
        public static Task<SystemHealth> CheckSystemHealthAsync()
        {
            return _manager.CheckOverallHealthAsync();
        }
        
        /// <summary>
        /// Returns current resource utilization metrics.
        /// </summary>
        public static ResourceUtilization GetResourceUtilization()
        {
            return _manager.GetResourceUtilization();
        }
        
        /// <summary>
        /// Gets the predictive monitoring service.
        /// </summary>
        public static PredictiveMonitor Predict => _manager.GetPredictiveMonitor();
        
        /// <summary>
        /// Records a time-series data point for predictive analysis.
        /// </summary>
        public static void RecordTimeSeriesDataPoint(string metricName, double value, IDictionary<string, string> dimensions = null)
        {
            _manager.RecordTimeSeriesDataPoint(metricName, value, dimensions);
        }
        
        /// <summary>
        /// Gets the time-series data for a specific metric.
        /// </summary>
        public static IEnumerable<TimeSeriesMetricPoint> GetMetricTimeSeries(string metricName, TimeSpan? timeWindow = null)
        {
            return _manager.GetMetricTimeSeries(metricName, timeWindow);
        }
        
        /// <summary>
        /// Predicts the future value of a metric.
        /// </summary>
        public static MetricPrediction PredictMetricValue(string metricName, TimeSpan predictionHorizon, TimeSpan? historyWindow = null)
        {
            return _manager.PredictMetricValue(metricName, predictionHorizon, historyWindow);
        }
        
        /// <summary>
        /// Performs comprehensive predictive analysis on system metrics.
        /// </summary>
        public static PredictiveAnalysisResult PerformPredictiveAnalysis(
            TimeSpan predictionHorizon,
            TimeSpan analysisTimeWindow,
            Func<string, bool> metricFilter = null,
            Dictionary<string, double> thresholds = null)
        {
            return _manager.PerformPredictiveAnalysis(predictionHorizon, analysisTimeWindow, metricFilter, thresholds);
        }
        
        /// <summary>
        /// Detects anomalies in a time-series metric using advanced algorithms.
        /// </summary>
        public static Task<dynamic> DetectAnomaliesAsync(string metricName, TimeSpan analysisTimeWindow)
        {
            return _manager.DetectAnomaliesAsync(metricName, analysisTimeWindow);
        }
    }
}