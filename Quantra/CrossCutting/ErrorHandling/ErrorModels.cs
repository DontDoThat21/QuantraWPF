using System;

namespace Quantra.CrossCutting.ErrorHandling
{
    /// <summary>
    /// Categorization of exceptions for appropriate handling.
    /// </summary>
    public enum ErrorCategory
    {
        /// <summary>
        /// Transient error that can be retried.
        /// </summary>
        Transient,
        
        /// <summary>
        /// User error requiring user correction.
        /// </summary>
        UserError,
        
        /// <summary>
        /// System failure requiring administrative action.
        /// </summary>
        SystemFailure,
        
        /// <summary>
        /// API error with an external service.
        /// </summary>
        ApiError,
        
        /// <summary>
        /// Database error.
        /// </summary>
        DatabaseError,
        
        /// <summary>
        /// Network connectivity issue.
        /// </summary>
        NetworkError,
        
        /// <summary>
        /// Trading-specific error.
        /// </summary>
        TradingError,
        
        /// <summary>
        /// Security-related error.
        /// </summary>
        SecurityError,
        
        /// <summary>
        /// Configuration error.
        /// </summary>
        ConfigurationError,
        
        /// <summary>
        /// Unknown error type.
        /// </summary>
        Unknown
    }
    
    /// <summary>
    /// Retry options for resilient operations.
    /// </summary>
    public class RetryOptions
    {
        /// <summary>
        /// Maximum number of retry attempts.
        /// </summary>
        public int MaxRetries { get; set; } = 3;
        
        /// <summary>
        /// Initial delay between retries in milliseconds.
        /// </summary>
        public int InitialDelayMs { get; set; } = 100;
        
        /// <summary>
        /// Factor by which the delay increases with each retry.
        /// </summary>
        public double BackoffFactor { get; set; } = 2.0;
        
        /// <summary>
        /// Maximum delay between retries in milliseconds.
        /// </summary>
        public int MaxDelayMs { get; set; } = 10000;
        
        /// <summary>
        /// Whether to add jitter to retry delays.
        /// </summary>
        public bool UseJitter { get; set; } = true;
        
        /// <summary>
        /// Types of exceptions that should trigger retries.
        /// </summary>
        public Type[] RetryableExceptions { get; set; }
        
        /// <summary>
        /// Custom function to determine if an exception is retryable.
        /// </summary>
        public Func<Exception, bool> RetryPredicate { get; set; }
        
        /// <summary>
        /// Gets retry options suitable for critical operations.
        /// </summary>
        public static RetryOptions ForCriticalOperation()
        {
            return new RetryOptions
            {
                MaxRetries = 5,
                InitialDelayMs = 200,
                BackoffFactor = 2.0,
                MaxDelayMs = 30000,
                UseJitter = true
            };
        }
        
        /// <summary>
        /// Gets retry options suitable for user-facing operations.
        /// </summary>
        public static RetryOptions ForUserFacingOperation()
        {
            return new RetryOptions
            {
                MaxRetries = 2,
                InitialDelayMs = 100,
                BackoffFactor = 1.5,
                MaxDelayMs = 3000,
                UseJitter = true
            };
        }
        
        /// <summary>
        /// Gets retry options suitable for background operations.
        /// </summary>
        public static RetryOptions ForBackgroundOperation()
        {
            return new RetryOptions
            {
                MaxRetries = 10,
                InitialDelayMs = 500,
                BackoffFactor = 2.0,
                MaxDelayMs = 60000,
                UseJitter = true
            };
        }
    }
    
    /// <summary>
    /// Circuit breaker options for resilient operations.
    /// </summary>
    public class CircuitBreakerOptions
    {
        /// <summary>
        /// Number of consecutive failures required to trip the circuit.
        /// </summary>
        public int FailureThreshold { get; set; } = 5;
        
        /// <summary>
        /// Time in milliseconds to keep the circuit open before allowing a trial request.
        /// </summary>
        public int DurationOfBreakMs { get; set; } = 30000;
        
        /// <summary>
        /// Time window in milliseconds over which to count failures.
        /// </summary>
        public int SamplingDurationMs { get; set; } = 60000;
        
        /// <summary>
        /// Types of exceptions that should be counted as failures.
        /// </summary>
        public Type[] FailureExceptions { get; set; }
        
        /// <summary>
        /// Custom function to determine if an exception should count as a failure.
        /// </summary>
        public Func<Exception, bool> FailurePredicate { get; set; }
        
        /// <summary>
        /// Action to perform when the circuit transitions to open.
        /// </summary>
        public Action<string, Exception> OnCircuitOpened { get; set; }
        
        /// <summary>
        /// Action to perform when the circuit transitions to closed.
        /// </summary>
        public Action<string> OnCircuitClosed { get; set; }
        
        /// <summary>
        /// Action to perform when the circuit enters half-open state.
        /// </summary>
        public Action<string> OnCircuitHalfOpen { get; set; }
    }
    
    /// <summary>
    /// Status of a circuit breaker.
    /// </summary>
    public enum CircuitBreakerStatus
    {
        /// <summary>
        /// Circuit is closed and functioning normally.
        /// </summary>
        Closed,
        
        /// <summary>
        /// Circuit is open and not allowing requests.
        /// </summary>
        Open,
        
        /// <summary>
        /// Circuit is allowing a trial request.
        /// </summary>
        HalfOpen,
        
        /// <summary>
        /// Circuit does not exist or has not been configured.
        /// </summary>
        NotRegistered
    }
}