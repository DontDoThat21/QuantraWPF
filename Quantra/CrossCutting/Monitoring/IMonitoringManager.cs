using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.CrossCutting;
using Quantra.CrossCutting.Monitoring.Models;

namespace Quantra.CrossCutting.Monitoring
{
    /// <summary>
    /// Interface for the centralized monitoring system.
    /// </summary>
    public interface IMonitoringManager : ICrossCuttingModule
    {
        /// <summary>
        /// Records a metric value.
        /// </summary>
        void RecordMetric(string name, double value, IDictionary<string, string> dimensions = null);
        
        /// <summary>
        /// Records the execution time of an operation.
        /// </summary>
        TimeSpan RecordExecutionTime(string operationName, Action action, IDictionary<string, string> dimensions = null);
        
        /// <summary>
        /// Records the execution time of a function.
        /// </summary>
        (T Result, TimeSpan Duration) RecordExecutionTime<T>(string operationName, Func<T> func, IDictionary<string, string> dimensions = null);
        
        /// <summary>
        /// Records the execution time of an async action.
        /// </summary>
        Task<TimeSpan> RecordExecutionTimeAsync(string operationName, Func<Task> action, IDictionary<string, string> dimensions = null);
        
        /// <summary>
        /// Records the execution time of an async function.
        /// </summary>
        Task<(T Result, TimeSpan Duration)> RecordExecutionTimeAsync<T>(string operationName, Func<Task<T>> func, IDictionary<string, string> dimensions = null);
        
        /// <summary>
        /// Records a successful operation.
        /// </summary>
        void RecordSuccess(string operationName, IDictionary<string, string> dimensions = null);
        
        /// <summary>
        /// Records a failed operation.
        /// </summary>
        void RecordFailure(string operationName, Exception exception = null, IDictionary<string, string> dimensions = null);
        
        /// <summary>
        /// Records memory usage at this point.
        /// </summary>
        void RecordMemoryUsage(string context = null);
        
        /// <summary>
        /// Gets metrics for a specific operation.
        /// </summary>
        OperationMetrics GetMetrics(string operationName);
        
        /// <summary>
        /// Gets all recorded metrics.
        /// </summary>
        IReadOnlyDictionary<string, OperationMetrics> GetAllMetrics();
        
        /// <summary>
        /// Clears historical metrics beyond a certain time window.
        /// </summary>
        void ClearHistoricalMetrics(TimeSpan age);
        
        /// <summary>
        /// Returns the health status of a specific component.
        /// </summary>
        Task<ComponentHealth> CheckComponentHealthAsync(string componentName);
        
        /// <summary>
        /// Returns the overall health status.
        /// </summary>
        Task<SystemHealth> CheckOverallHealthAsync();
        
        /// <summary>
        /// Returns current resource utilization metrics.
        /// </summary>
        ResourceUtilization GetResourceUtilization();
        
        /// <summary>
        /// Gets the predictive monitor instance.
        /// </summary>
        PredictiveMonitor GetPredictiveMonitor();
        
        /// <summary>
        /// Records a time-series data point for predictive analysis.
        /// </summary>
        void RecordTimeSeriesDataPoint(string metricName, double value, IDictionary<string, string> dimensions = null);
        
        /// <summary>
        /// Gets the metric time series for a specific metric.
        /// </summary>
        IEnumerable<TimeSeriesMetricPoint> GetMetricTimeSeries(string metricName, TimeSpan? timeWindow = null);
        
        /// <summary>
        /// Predicts the future value of a metric.
        /// </summary>
        MetricPrediction PredictMetricValue(string metricName, TimeSpan predictionHorizon, TimeSpan? historyWindow = null);
        
        /// <summary>
        /// Performs comprehensive predictive analysis on system metrics.
        /// </summary>
        PredictiveAnalysisResult PerformPredictiveAnalysis(
            TimeSpan predictionHorizon,
            TimeSpan analysisTimeWindow,
            Func<string, bool> metricFilter = null,
            Dictionary<string, double> thresholds = null);
        
        /// <summary>
        /// Detects anomalies in a time-series metric using advanced algorithms.
        /// </summary>
        Task<dynamic> DetectAnomaliesAsync(string metricName, TimeSpan analysisTimeWindow);
    }
}