using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data.SQLite;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using Polly;
using Polly.CircuitBreaker;
using Polly.Retry;
using Polly.Wrap;
using Quantra.CrossCutting.Logging;
using Quantra.Models; // For AlertModel, etc.

namespace Quantra.CrossCutting.ErrorHandling
{
    /// <summary>
    /// Manager for centralized error handling and resilience policies.
    /// </summary>
    public class ErrorHandlingManager : IErrorHandlingManager
    {
        private static readonly Lazy<ErrorHandlingManager> _instance = new Lazy<ErrorHandlingManager>(() => new ErrorHandlingManager());
        private readonly ILogger _logger;
        private readonly ConcurrentDictionary<string, AsyncCircuitBreakerPolicy> _circuitBreakers;
        private readonly ConcurrentDictionary<string, CircuitBreakerStatus> _circuitStatuses;
        private bool _initialized;

        /// <summary>
        /// Gets the singleton instance of the ErrorHandlingManager.
        /// </summary>
        public static ErrorHandlingManager Instance => _instance.Value;

        /// <inheritdoc />
        public string ModuleName => "ErrorHandling";

        /// <summary>
        /// Private constructor to enforce singleton pattern.
        /// </summary>
        private ErrorHandlingManager()
        {
            _circuitBreakers = new ConcurrentDictionary<string, AsyncCircuitBreakerPolicy>();
            _circuitStatuses = new ConcurrentDictionary<string, CircuitBreakerStatus>();
            _logger = Log.ForType<ErrorHandlingManager>();
            
            // Register standard circuit breakers during construction to avoid circular dependencies
            RegisterCircuitBreaker("AlphaVantage");
            RegisterCircuitBreaker("Database");
            RegisterCircuitBreaker("WebullAPI");
            RegisterCircuitBreaker("MLInference");
            
            _logger.Information("ErrorHandlingManager initialized with standard circuit breakers");
            _initialized = true;
        }

        /// <inheritdoc />
        public void Initialize(string configurationSection = null)
        {
            // Initialization is now done in constructor to prevent circular dependencies
            // This method remains for interface compatibility
            if (!_initialized)
            {
                _logger.Warning("Initialize called but initialization should have been completed in constructor");
            }
        }

        /// <inheritdoc />
        public ErrorCategory CategorizeException(Exception exception)
        {
            if (exception == null)
            {
                return ErrorCategory.Unknown;
            }

            // Check for specific exception types
            foreach (var ex in exception.GetAllExceptions())
            {
                // Check exception type and any relevant properties
                if (ex is TimeoutException || ex is OperationCanceledException)
                {
                    return ErrorCategory.Transient;
                }
                
                if (ex is SocketException || ex is WebException || ex is HttpRequestException)
                {
                    return ErrorCategory.NetworkError;
                }
                
                if (ex is SQLiteException || ex.Message.Contains("database"))
                {
                    return ErrorCategory.DatabaseError;
                }
                
                if (ex is UnauthorizedAccessException || ex is System.Security.SecurityException)
                {
                    return ErrorCategory.SecurityError;
                }
                
                if (ex is ArgumentException || ex is FormatException || ex is InvalidOperationException)
                {
                    return ErrorCategory.UserError;
                }
                
                if (ex is IOException || ex is System.Runtime.InteropServices.COMException)
                {
                    return ErrorCategory.SystemFailure;
                }
                
                // Check for trading specific errors
                if (ex.Message.Contains("trading") || ex.Message.Contains("position") || 
                    ex.Message.Contains("order") || ex.Message.Contains("execution"))
                {
                    return ErrorCategory.TradingError;
                }
                
                // Check for API specific errors
                if (ex.Message.Contains("API") || ex.Message.Contains("quota") || 
                    ex.Message.Contains("rate limit") || ex.Message.Contains("throttle"))
                {
                    return ErrorCategory.ApiError;
                }
                
                if (ex.Message.Contains("config") || ex.Message.Contains("setting"))
                {
                    return ErrorCategory.ConfigurationError;
                }
            }
            
            // If we cannot categorize specifically, check if it's transient
            if (exception.IsTransient())
            {
                return ErrorCategory.Transient;
            }
            
            // Default to unknown if we cannot determine a specific category
            return ErrorCategory.Unknown;
        }

        /// <inheritdoc />
        public void HandleException(Exception exception, string context = null)
        {
            if (exception == null)
            {
                return;
            }

            var category = CategorizeException(exception);
            var contextLogger = string.IsNullOrEmpty(context) 
                ? _logger 
                : _logger.ForContext("ErrorContext", context);
            
            // Log the exception with appropriate level and context
            contextLogger.ForContext("ErrorCategory", category.ToString())
                .Error(exception, "Error occurred: {ErrorMessage}", exception.GetFullErrorMessage());
            
            // Take additional actions based on the category
            switch (category)
            {
                case ErrorCategory.Transient:
                    // For transient errors, just log and let retry policies handle it
                    break;
                    
                case ErrorCategory.UserError:
                    // User errors should be displayed to the user
                    RaiseAlert(exception, "User Input Error", AlertCategory.Standard, 2);
                    break;
                    
                case ErrorCategory.SystemFailure:
                    // System failures require immediate attention
                    RaiseAlert(exception, "System Failure", AlertCategory.SystemHealth, 3);
                    break;
                    
                case ErrorCategory.ApiError:
                    // API errors might require throttling or circuit breaking
                    RaiseAlert(exception, "API Error", AlertCategory.Global, 2);
                    break;
                    
                case ErrorCategory.DatabaseError:
                    // Database errors are critical
                    RaiseAlert(exception, "Database Error", AlertCategory.SystemHealth, 3);
                    break;
                    
                case ErrorCategory.NetworkError:
                    // Network errors might be temporary
                    RaiseAlert(exception, "Network Connectivity Error", AlertCategory.SystemHealth, 2);
                    break;
                    
                case ErrorCategory.TradingError:
                    // Trading errors are very important for a trading system
                    RaiseAlert(exception, "Trading System Error", AlertCategory.Opportunity, 3);
                    break;
                    
                case ErrorCategory.SecurityError:
                    // Security errors require immediate attention
                    RaiseAlert(exception, "Security Alert", AlertCategory.SystemHealth, 3);
                    break;
                    
                case ErrorCategory.ConfigurationError:
                    // Configuration errors might prevent correct operation
                    RaiseAlert(exception, "Configuration Error", AlertCategory.SystemHealth, 2);
                    break;
                    
                default:
                    RaiseAlert(exception, "Unexpected Error", AlertCategory.Global, 2);
                    break;
            }
        }

        /// <inheritdoc />
        public void ExecuteWithRetry(Action action, RetryOptions options = null)
        {
            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }
            
            options ??= RetryOptions.ForUserFacingOperation();
            
            // Create retry policy
            var policy = CreateRetryPolicy(options);
            
            try
            {
                policy.Execute(action);
            }
            catch (Exception ex)
            {
                // If we reach here, all retries have been exhausted
                HandleException(ex, "RetryExhausted");
                throw; // Re-throw after handling
            }
        }

        /// <inheritdoc />
        public T ExecuteWithRetry<T>(Func<T> func, RetryOptions options = null)
        {
            if (func == null)
            {
                throw new ArgumentNullException(nameof(func));
            }
            
            options ??= RetryOptions.ForUserFacingOperation();
            
            // Create retry policy
            var policy = CreateRetryPolicy(options);
            
            try
            {
                return policy.Execute(func);
            }
            catch (Exception ex)
            {
                // If we reach here, all retries have been exhausted
                HandleException(ex, "RetryExhausted");
                throw; // Re-throw after handling
            }
        }

        /// <inheritdoc />
        public async Task<T> ExecuteWithRetryAsync<T>(Func<Task<T>> func, RetryOptions options = null)
        {
            if (func == null)
            {
                throw new ArgumentNullException(nameof(func));
            }
            
            options ??= RetryOptions.ForUserFacingOperation();
            
            // Create retry policy
            var policy = CreateAsyncRetryPolicy(options);
            
            try
            {
                return await policy.ExecuteAsync(func).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                // If we reach here, all retries have been exhausted
                HandleException(ex, "RetryExhausted");
                throw; // Re-throw after handling
            }
        }

        /// <inheritdoc />
        public async Task ExecuteWithRetryAsync(Func<Task> action, RetryOptions options = null)
        {
            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }
            
            options ??= RetryOptions.ForUserFacingOperation();
            
            // Create retry policy
            var policy = CreateAsyncRetryPolicy(options);
            
            try
            {
                await policy.ExecuteAsync(action).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                // If we reach here, all retries have been exhausted
                HandleException(ex, "RetryExhausted");
                throw; // Re-throw after handling
            }
        }

        /// <inheritdoc />
        public void RegisterCircuitBreaker(string serviceName, CircuitBreakerOptions options = null)
        {
            if (string.IsNullOrEmpty(serviceName))
            {
                throw new ArgumentNullException(nameof(serviceName));
            }
            
            options ??= new CircuitBreakerOptions();
            
            // Set default handlers if not provided
            if (options.OnCircuitOpened == null)
            {
                options.OnCircuitOpened = (name, ex) => 
                    _logger.Warning(ex.Message, "Circuit breaker for {ServiceName} opened due to failures", name);
            }
                
            if (options.OnCircuitClosed == null)
            {
                options.OnCircuitClosed = name => 
                    _logger.Information("Circuit breaker for {ServiceName} closed - service recovered", name);
            }
                
            if (options.OnCircuitHalfOpen == null)
            {
                options.OnCircuitHalfOpen = name => 
                    _logger.Information("Circuit breaker for {ServiceName} half-open - testing service", name);
            }

            // Create the failure predicate
            Func<Exception, bool> failurePredicate = options.FailurePredicate ?? (ex => true);
            if (options.FailureExceptions != null && options.FailureExceptions.Length > 0)
            {
                var exceptionTypes = options.FailureExceptions;
                failurePredicate = ex => exceptionTypes.Contains(ex.GetType()) || 
                                        exceptionTypes.Any(t => t.IsInstanceOfType(ex));
            }

            // Create the circuit breaker policy
            var policy = Policy
                .Handle<Exception>(failurePredicate)
                .AdvancedCircuitBreakerAsync(
                    failureThreshold: 0.5, // 50% failure threshold
                    samplingDuration: TimeSpan.FromMilliseconds(options.SamplingDurationMs),
                    minimumThroughput: options.FailureThreshold,
                    durationOfBreak: TimeSpan.FromMilliseconds(options.DurationOfBreakMs),
                    onBreak: (ex, ts) => options.OnCircuitOpened(serviceName, ex),
                    onReset: () => options.OnCircuitClosed(serviceName),
                    onHalfOpen: () => options.OnCircuitHalfOpen(serviceName)
                );

            // Store the circuit breaker
            _circuitBreakers[serviceName] = policy;
            _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
            _logger.Debug("Registered circuit breaker for {ServiceName}", serviceName);
        }

        /// <inheritdoc />
        public void ExecuteWithCircuitBreaker(string serviceName, Action action)
        {
            if (string.IsNullOrEmpty(serviceName))
            {
                throw new ArgumentNullException(nameof(serviceName));
            }

            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }
            
            if (!_circuitBreakers.TryGetValue(serviceName, out var circuitBreaker))
            {
                RegisterCircuitBreaker(serviceName);
                circuitBreaker = _circuitBreakers[serviceName];
            }
            
            try
            {
                circuitBreaker.ExecuteAsync(() => 
                {
                    action();
                    return Task.CompletedTask;
                }).GetAwaiter().GetResult();
                
                // Update circuit status
                _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
            }
            catch (BrokenCircuitException)
            {
                _circuitStatuses[serviceName] = CircuitBreakerStatus.Open;
                _logger.Warning("Circuit for {ServiceName} is open - rejecting request", serviceName);
                throw;
            }
            catch (Exception ex)
            {
                // Non-circuit exception, just handle it
                HandleException(ex, $"CircuitBreaker:{serviceName}");
                throw;
            }
        }

        /// <inheritdoc />
        public T ExecuteWithCircuitBreaker<T>(string serviceName, Func<T> func)
        {
            if (string.IsNullOrEmpty(serviceName))
            {
                throw new ArgumentNullException(nameof(serviceName));
            }

            if (func == null)
            {
                throw new ArgumentNullException(nameof(func));
            }
            
            if (!_circuitBreakers.TryGetValue(serviceName, out var circuitBreaker))
            {
                RegisterCircuitBreaker(serviceName);
                circuitBreaker = _circuitBreakers[serviceName];
            }
            
            try
            {
                T result = circuitBreaker.ExecuteAsync(() => 
                {
                    return Task.FromResult(func());
                }).GetAwaiter().GetResult();
                
                // Update circuit status
                _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
                return result;
            }
            catch (BrokenCircuitException)
            {
                _circuitStatuses[serviceName] = CircuitBreakerStatus.Open;
                _logger.Warning("Circuit for {ServiceName} is open - rejecting request", serviceName);
                throw;
            }
            catch (Exception ex)
            {
                // Non-circuit exception, just handle it
                HandleException(ex, $"CircuitBreaker:{serviceName}");
                throw;
            }
        }

        /// <inheritdoc />
        public async Task<T> ExecuteWithCircuitBreakerAsync<T>(string serviceName, Func<Task<T>> func)
        {
            if (string.IsNullOrEmpty(serviceName))
            {
                throw new ArgumentNullException(nameof(serviceName));
            }

            if (func == null)
            {
                throw new ArgumentNullException(nameof(func));
            }
            
            if (!_circuitBreakers.TryGetValue(serviceName, out var circuitBreaker))
            {
                RegisterCircuitBreaker(serviceName);
                circuitBreaker = _circuitBreakers[serviceName];
            }
            
            try
            {
                T result = await circuitBreaker.ExecuteAsync(func).ConfigureAwait(false);
                
                // Update circuit status
                _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
                return result;
            }
            catch (BrokenCircuitException)
            {
                _circuitStatuses[serviceName] = CircuitBreakerStatus.Open;
                _logger.Warning("Circuit for {ServiceName} is open - rejecting request", serviceName);
                throw;
            }
            catch (Exception ex)
            {
                // Non-circuit exception, just handle it
                HandleException(ex, $"CircuitBreaker:{serviceName}");
                throw;
            }
        }

        /// <inheritdoc />
        public CircuitBreakerStatus GetCircuitStatus(string serviceName)
        {
            if (string.IsNullOrEmpty(serviceName))
            {
                return CircuitBreakerStatus.NotRegistered;
            }

            return _circuitStatuses.TryGetValue(serviceName, out var status) ? status : CircuitBreakerStatus.NotRegistered;
        }

        /// <summary>
        /// Raises an alert for an exception.
        /// </summary>
        private void RaiseAlert(Exception exception, string alertName, AlertCategory category, int priority)
        {
            try
            {
                var alert = new AlertModel
                {
                    Name = alertName,
                    Condition = exception.GetFullErrorMessage(),
                    AlertType = "Error",
                    IsActive = true,
                    Priority = priority,
                    CreatedDate = DateTime.Now,
                    Category = category,
                    Notes = exception.ToString()
                };
                Controls.AlertsControl.EmitGlobalAlert(alert);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to emit error alert");
            }
        }
        
        /// <summary>
        /// Creates a synchronous retry policy.
        /// </summary>
        private Policy CreateRetryPolicy(RetryOptions options)
        {
            return Policy
                .Handle<Exception>(ex => ShouldRetry(ex, options))
                .WaitAndRetry(
                    retryCount: options.MaxRetries,
                    sleepDurationProvider: (attempt, context) => 
                        CalculateRetryDelay(attempt, options),
                    onRetry: (ex, timespan, attempt, context) => 
                        _logger.Warning(ex.Message, "Retry attempt {RetryAttempt}/{MaxRetries} after {RetryDelayMs}ms. Reason: {ErrorMessage}", 
                            attempt, options.MaxRetries, timespan.TotalMilliseconds, ex.Message));
        }
        
        /// <summary>
        /// Creates an asynchronous retry policy.
        /// </summary>
        private AsyncPolicy CreateAsyncRetryPolicy(RetryOptions options)
        {
            return Policy
                .Handle<Exception>(ex => ShouldRetry(ex, options))
                .WaitAndRetryAsync(
                    retryCount: options.MaxRetries,
                    sleepDurationProvider: (attempt, context) => 
                        CalculateRetryDelay(attempt, options),
                    onRetryAsync: (ex, timespan, attempt, context) =>
                    {
                        _logger.Warning(ex.Message, "Retry attempt {RetryAttempt}/{MaxRetries} after {RetryDelayMs}ms. Reason: {ErrorMessage}", 
                            attempt, options.MaxRetries, timespan.TotalMilliseconds, ex.Message);
                        return Task.CompletedTask;
                    });
        }
        
        /// <summary>
        /// Determines whether an exception should trigger a retry.
        /// </summary>
        private bool ShouldRetry(Exception exception, RetryOptions options)
        {
            if (options.RetryPredicate != null)
            {
                return options.RetryPredicate(exception);
            }
            
            if (options.RetryableExceptions != null && options.RetryableExceptions.Length > 0)
            {
                return options.RetryableExceptions.Any(type => 
                    type.IsInstanceOfType(exception) || exception.ContainsExceptionType(type));
            }
            
            // Default to checking if it's a transient exception
            return exception.IsTransient();
        }
        
        /// <summary>
        /// Calculates the delay for a retry attempt.
        /// </summary>
        private TimeSpan CalculateRetryDelay(int attempt, RetryOptions options)
        {
            var delay = options.InitialDelayMs * Math.Pow(options.BackoffFactor, attempt - 1);
            var jitter = options.UseJitter ? new Random().NextDouble() * 0.2 - 0.1 : 0; // ±10% jitter
            
            var finalDelay = delay * (1 + jitter);
            return TimeSpan.FromMilliseconds(Math.Min(finalDelay, options.MaxDelayMs));
        }
    }
}