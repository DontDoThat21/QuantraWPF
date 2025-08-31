using System;
using System.Collections.Generic;

namespace Quantra.CrossCutting.Monitoring.Models
{
    /// <summary>
    /// Represents a time-series metric point with timestamp.
    /// </summary>
    public class TimeSeriesMetricPoint
    {
        /// <summary>
        /// Gets or sets the timestamp of the metric.
        /// </summary>
        public DateTime Timestamp { get; set; }
        
        /// <summary>
        /// Gets or sets the metric value.
        /// </summary>
        public double Value { get; set; }
        
        /// <summary>
        /// Gets or sets additional dimensions associated with this metric.
        /// </summary>
        public Dictionary<string, string> Dimensions { get; set; }
    }
    
    /// <summary>
    /// Represents a time-series of metric values.
    /// </summary>
    public class MetricTimeSeries
    {
        /// <summary>
        /// Gets or sets the name of the metric.
        /// </summary>
        public string MetricName { get; set; }
        
        /// <summary>
        /// Gets or sets the collection of metric data points.
        /// </summary>
        public List<TimeSeriesMetricPoint> DataPoints { get; set; } = new List<TimeSeriesMetricPoint>();
    }
    
    /// <summary>
    /// Represents a prediction for a specific metric.
    /// </summary>
    public class MetricPrediction
    {
        /// <summary>
        /// Gets or sets the metric name.
        /// </summary>
        public string MetricName { get; set; }
        
        /// <summary>
        /// Gets or sets the predicted value.
        /// </summary>
        public double PredictedValue { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp for which the prediction is made.
        /// </summary>
        public DateTime PredictionForTime { get; set; }
        
        /// <summary>
        /// Gets or sets the time when the prediction was created.
        /// </summary>
        public DateTime PredictionCreatedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the confidence level of the prediction (0-1).
        /// </summary>
        public double Confidence { get; set; }
        
        /// <summary>
        /// Gets or sets the lower boundary of the prediction interval.
        /// </summary>
        public double LowerBound { get; set; }
        
        /// <summary>
        /// Gets or sets the upper boundary of the prediction interval.
        /// </summary>
        public double UpperBound { get; set; }
        
        /// <summary>
        /// Gets or sets the anomaly score (0-1, higher means more anomalous).
        /// </summary>
        public double AnomalyScore { get; set; }
    }
    
    /// <summary>
    /// Represents the trend analysis for a metric.
    /// </summary>
    public class MetricTrend
    {
        /// <summary>
        /// Gets or sets the metric name.
        /// </summary>
        public string MetricName { get; set; }
        
        /// <summary>
        /// Gets or sets the trend direction.
        /// </summary>
        public TrendDirection Direction { get; set; }
        
        /// <summary>
        /// Gets or sets the percentage change over the analyzed period.
        /// </summary>
        public double PercentChange { get; set; }
        
        /// <summary>
        /// Gets or sets the rate of change (units per time period).
        /// </summary>
        public double RateOfChange { get; set; }
        
        /// <summary>
        /// Gets or sets the time window over which the trend was calculated.
        /// </summary>
        public TimeSpan AnalysisPeriod { get; set; }
        
        /// <summary>
        /// Gets or sets the time when the trend analysis was performed.
        /// </summary>
        public DateTime AnalyzedAt { get; set; }
    }
    
    /// <summary>
    /// Represents the direction of a trend.
    /// </summary>
    public enum TrendDirection
    {
        /// <summary>
        /// Trend is increasing.
        /// </summary>
        Increasing,
        
        /// <summary>
        /// Trend is decreasing.
        /// </summary>
        Decreasing,
        
        /// <summary>
        /// Trend is stable or fluctuating without clear direction.
        /// </summary>
        Stable,
        
        /// <summary>
        /// Trend is cyclical or periodic.
        /// </summary>
        Cyclical
    }
    
    /// <summary>
    /// Represents a predictive alert for a potential future issue.
    /// </summary>
    public class PredictiveAlert
    {
        /// <summary>
        /// Gets or sets the unique identifier for the alert.
        /// </summary>
        public Guid Id { get; set; } = Guid.NewGuid();
        
        /// <summary>
        /// Gets or sets the alert title.
        /// </summary>
        public string Title { get; set; }
        
        /// <summary>
        /// Gets or sets the alert description.
        /// </summary>
        public string Description { get; set; }
        
        /// <summary>
        /// Gets or sets the metric that triggered the alert.
        /// </summary>
        public string MetricName { get; set; }
        
        /// <summary>
        /// Gets or sets the current value of the metric.
        /// </summary>
        public double CurrentValue { get; set; }
        
        /// <summary>
        /// Gets or sets the predicted value that triggered the alert.
        /// </summary>
        public double PredictedValue { get; set; }
        
        /// <summary>
        /// Gets or sets the threshold value that was exceeded.
        /// </summary>
        public double Threshold { get; set; }
        
        /// <summary>
        /// Gets or sets the time range in which the issue is predicted to occur.
        /// </summary>
        public TimeSpan TimeToIssue { get; set; }
        
        /// <summary>
        /// Gets or sets the prediction confidence level (0-1).
        /// </summary>
        public double Confidence { get; set; }
        
        /// <summary>
        /// Gets or sets the severity level of the alert.
        /// </summary>
        public AlertSeverity Severity { get; set; }
        
        /// <summary>
        /// Gets or sets the time when the alert was created.
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.Now;
        
        /// <summary>
        /// Gets or sets the time until which this alert is valid.
        /// </summary>
        public DateTime ValidUntil { get; set; }
        
        /// <summary>
        /// Gets or sets suggested actions to address the predicted issue.
        /// </summary>
        public List<string> SuggestedActions { get; set; } = new List<string>();
    }
    
    /// <summary>
    /// Represents the severity level of a predictive alert.
    /// </summary>
    public enum AlertSeverity
    {
        /// <summary>
        /// Informational alert, no immediate action required.
        /// </summary>
        Information = 0,
        
        /// <summary>
        /// Low severity, should be addressed in normal operational workflow.
        /// </summary>
        Low = 1,
        
        /// <summary>
        /// Medium severity, should be addressed soon.
        /// </summary>
        Medium = 2,
        
        /// <summary>
        /// High severity, requires prompt attention.
        /// </summary>
        High = 3,
        
        /// <summary>
        /// Critical severity, requires immediate attention.
        /// </summary>
        Critical = 4
    }
    
    /// <summary>
    /// Represents the result of a predictive analysis process.
    /// </summary>
    public class PredictiveAnalysisResult
    {
        /// <summary>
        /// Gets or sets the time when the analysis was performed.
        /// </summary>
        public DateTime AnalyzedAt { get; set; } = DateTime.Now;
        
        /// <summary>
        /// Gets or sets the collection of metric predictions.
        /// </summary>
        public List<MetricPrediction> Predictions { get; set; } = new List<MetricPrediction>();
        
        /// <summary>
        /// Gets or sets the collection of detected trends.
        /// </summary>
        public List<MetricTrend> Trends { get; set; } = new List<MetricTrend>();
        
        /// <summary>
        /// Gets or sets the collection of generated predictive alerts.
        /// </summary>
        public List<PredictiveAlert> Alerts { get; set; } = new List<PredictiveAlert>();
    }
}