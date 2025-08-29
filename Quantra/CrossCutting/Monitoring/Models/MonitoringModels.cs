using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace Quantra.CrossCutting.Monitoring.Models
{
    /// <summary>
    /// Represents metrics for a specific operation.
    /// </summary>
    public class OperationMetrics
    {
        /// <summary>
        /// Operation name.
        /// </summary>
        public string Name { get; set; }
        
        /// <summary>
        /// Total number of executions.
        /// </summary>
        public int ExecutionCount { get; set; }
        
        /// <summary>
        /// Number of successful executions.
        /// </summary>
        public int SuccessCount { get; set; }
        
        /// <summary>
        /// Number of failed executions.
        /// </summary>
        public int FailureCount { get; set; }
        
        /// <summary>
        /// Average execution time in milliseconds.
        /// </summary>
        public double AverageExecutionTimeMs { get; set; }
        
        /// <summary>
        /// Minimum execution time in milliseconds.
        /// </summary>
        public double MinExecutionTimeMs { get; set; }
        
        /// <summary>
        /// Maximum execution time in milliseconds.
        /// </summary>
        public double MaxExecutionTimeMs { get; set; }
        
        /// <summary>
        /// Recent execution times in milliseconds.
        /// </summary>
        public List<double> RecentExecutionTimesMs { get; set; } = new List<double>();
        
        /// <summary>
        /// Time of the last execution.
        /// </summary>
        public DateTime LastExecutionTime { get; set; }
        
        /// <summary>
        /// Most recent failures with details.
        /// </summary>
        public List<FailureRecord> RecentFailures { get; set; } = new List<FailureRecord>();
    }
    
    /// <summary>
    /// Represents a failure record.
    /// </summary>
    public class FailureRecord
    {
        /// <summary>
        /// Time when the failure occurred.
        /// </summary>
        public DateTime OccurredAt { get; set; }
        
        /// <summary>
        /// Exception type.
        /// </summary>
        public string ExceptionType { get; set; }
        
        /// <summary>
        /// Error message.
        /// </summary>
        public string Message { get; set; }
        
        /// <summary>
        /// Additional dimensions for the failure.
        /// </summary>
        public Dictionary<string, string> Dimensions { get; set; }
    }
    
    /// <summary>
    /// Represents the health status of a component.
    /// </summary>
    public class ComponentHealth
    {
        /// <summary>
        /// Component name.
        /// </summary>
        public string Name { get; set; }
        
        /// <summary>
        /// Status of the component.
        /// </summary>
        public HealthStatus Status { get; set; }
        
        /// <summary>
        /// Description of the health status.
        /// </summary>
        public string Description { get; set; }
        
        /// <summary>
        /// Time of the check.
        /// </summary>
        public DateTime CheckedAt { get; set; }
        
        /// <summary>
        /// Error message if any.
        /// </summary>
        public string Error { get; set; }
        
        /// <summary>
        /// Additional health metrics for the component.
        /// </summary>
        public Dictionary<string, object> Metrics { get; set; }
    }
    
    /// <summary>
    /// Represents the overall system health.
    /// </summary>
    public class SystemHealth
    {
        /// <summary>
        /// Overall status.
        /// </summary>
        public HealthStatus Status { get; set; }
        
        /// <summary>
        /// Time of the check.
        /// </summary>
        public DateTime CheckedAt { get; set; }
        
        /// <summary>
        /// Health statuses of individual components.
        /// </summary>
        public ConcurrentDictionary<string, ComponentHealth> Components { get; set; }
        
        /// <summary>
        /// System uptime.
        /// </summary>
        public TimeSpan Uptime { get; set; }
        
        /// <summary>
        /// Resource utilization metrics.
        /// </summary>
        public ResourceUtilization Resources { get; set; }
    }
    
    /// <summary>
    /// Health status enumeration.
    /// </summary>
    public enum HealthStatus
    {
        /// <summary>
        /// The component is functioning normally.
        /// </summary>
        Healthy,
        
        /// <summary>
        /// The component has some issues but is still operational.
        /// </summary>
        Degraded,
        
        /// <summary>
        /// The component is not functioning.
        /// </summary>
        Unhealthy,
        
        /// <summary>
        /// The status of the component is unknown.
        /// </summary>
        Unknown
    }
    
    /// <summary>
    /// Represents resource utilization metrics.
    /// </summary>
    public class ResourceUtilization
    {
        /// <summary>
        /// CPU usage percentage.
        /// </summary>
        public double CpuUsagePercent { get; set; }
        
        /// <summary>
        /// Memory usage in bytes.
        /// </summary>
        public long MemoryUsageBytes { get; set; }
        
        /// <summary>
        /// Disk space usage in bytes.
        /// </summary>
        public long DiskUsageBytes { get; set; }
        
        /// <summary>
        /// Total available disk space in bytes.
        /// </summary>
        public long DiskTotalBytes { get; set; }
        
        /// <summary>
        /// Network throughput in bytes per second.
        /// </summary>
        public long NetworkThroughputBytesPerSecond { get; set; }
        
        /// <summary>
        /// Time of the measurement.
        /// </summary>
        public DateTime MeasuredAt { get; set; }
        
        /// <summary>
        /// Memory usage percentage.
        /// </summary>
        public double MemoryUsagePercent
        {
            get
            {
                long totalMemory = Environment.WorkingSet;
                return totalMemory > 0 ? (double)MemoryUsageBytes / totalMemory * 100 : 0;
            }
        }
        
        /// <summary>
        /// Disk usage percentage.
        /// </summary>
        public double DiskUsagePercent => DiskTotalBytes > 0 ? (double)DiskUsageBytes / DiskTotalBytes * 100 : 0;
    }
}