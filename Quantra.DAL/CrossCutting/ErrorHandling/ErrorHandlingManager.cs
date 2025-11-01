using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Threading.Tasks;
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
        private readonly ConcurrentDictionary<string, SimpleCircuit> _circuitBreakers;
        private readonly ConcurrentDictionary<string, CircuitBreakerStatus> _circuitStatuses;
        private bool _initialized;

        /// <summary>
        /// Gets the singleton instance of the ErrorHandlingManager.
        /// </summary>
        public static ErrorHandlingManager Instance => _instance.Value;

        /// <inheritdoc />
        public string ModuleName => "ErrorHandling";

        /// <summary>
        /// Simple circuit breaker implementation (no external dependencies).
        /// </summary>
        private sealed class SimpleCircuit
        {
            public CircuitBreakerOptions Options { get; }
            public int ConsecutiveFailures { get; set; }
            public CircuitBreakerStatus Status { get; set; } = CircuitBreakerStatus.Closed;
            public DateTime LastBreakUtc { get; set; } = DateTime.MinValue;

            public SimpleCircuit(CircuitBreakerOptions options)
            {
                Options = options ?? new CircuitBreakerOptions();
            }

            public bool IsOpen(TimeSpan? nowOffset = null)
            {
                if (Status != CircuitBreakerStatus.Open) return false;
                var elapsed = DateTime.UtcNow - LastBreakUtc;
                return elapsed.TotalMilliseconds < Options.DurationOfBreakMs;
            }
        }

        /// <summary>
        /// Private constructor to enforce singleton pattern.
        /// </summary>
        private ErrorHandlingManager()
        {
            _circuitBreakers = new ConcurrentDictionary<string, SimpleCircuit>();
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
                
                // Avoid direct reference to external packages; match by type name
                if (string.Equals(ex.GetType().Name, "SqlException", StringComparison.Ordinal) || 
                    string.Equals(ex.GetType().Name, "SQLiteException", StringComparison.Ordinal) ||
                    ex.Message.Contains("database", StringComparison.OrdinalIgnoreCase))
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
                if (ex.Message.Contains("trading", StringComparison.OrdinalIgnoreCase) || ex.Message.Contains("position", StringComparison.OrdinalIgnoreCase) || 
                    ex.Message.Contains("order", StringComparison.OrdinalIgnoreCase) || ex.Message.Contains("execution", StringComparison.OrdinalIgnoreCase))
                {
                    return ErrorCategory.TradingError;
                }
                
                // Check for API specific errors
                if (ex.Message.Contains("API", StringComparison.OrdinalIgnoreCase) || ex.Message.Contains("quota", StringComparison.OrdinalIgnoreCase) || 
                    ex.Message.Contains("rate limit", StringComparison.OrdinalIgnoreCase) || ex.Message.Contains("throttle", StringComparison.OrdinalIgnoreCase))
                {
                    return ErrorCategory.ApiError;
                }
                
                if (ex.Message.Contains("config", StringComparison.OrdinalIgnoreCase) || ex.Message.Contains("setting", StringComparison.OrdinalIgnoreCase))
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
                    RaiseAlert(exception, "User Input Error", AlertCategory.Standard, 2);
                    break;
                    
                case ErrorCategory.SystemFailure:
                    RaiseAlert(exception, "System Failure", AlertCategory.SystemHealth, 3);
                    break;
                    
                case ErrorCategory.ApiError:
                    RaiseAlert(exception, "API Error", AlertCategory.Global, 2);
                    break;
                    
                case ErrorCategory.DatabaseError:
                    RaiseAlert(exception, "Database Error", AlertCategory.SystemHealth, 3);
                    break;
                    
                case ErrorCategory.NetworkError:
                    RaiseAlert(exception, "Network Connectivity Error", AlertCategory.SystemHealth, 2);
                    break;
                    
                case ErrorCategory.TradingError:
                    RaiseAlert(exception, "Trading System Error", AlertCategory.Opportunity, 3);
                    break;
                    
                case ErrorCategory.SecurityError:
                    RaiseAlert(exception, "Security Alert", AlertCategory.SystemHealth, 3);
                    break;
                    
                case ErrorCategory.ConfigurationError:
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

            int attempt = 1;
            while (true)
            {
                try
                {
                    action();
                    return;
                }
                catch (Exception ex)
                {
                    if (!ShouldRetry(ex, options) || attempt > options.MaxRetries)
                    {
                        HandleException(ex, "RetryExhausted");
                        throw;
                    }

                    var delay = CalculateRetryDelay(attempt, options);
                    _logger.Warning(ex.Message, "Retry attempt {RetryAttempt}/{MaxRetries} after {RetryDelayMs}ms. Reason: {ErrorMessage}", 
                        attempt, options.MaxRetries, delay.TotalMilliseconds, ex.Message);
                    Task.Delay(delay).GetAwaiter().GetResult();
                    attempt++;
                }
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

            int attempt = 1;
            while (true)
            {
                try
                {
                    return func();
                }
                catch (Exception ex)
                {
                    if (!ShouldRetry(ex, options) || attempt > options.MaxRetries)
                    {
                        HandleException(ex, "RetryExhausted");
                        throw;
                    }

                    var delay = CalculateRetryDelay(attempt, options);
                    _logger.Warning(ex.Message, "Retry attempt {RetryAttempt}/{MaxRetries} after {RetryDelayMs}ms. Reason: {ErrorMessage}", 
                        attempt, options.MaxRetries, delay.TotalMilliseconds, ex.Message);
                    Task.Delay(delay).GetAwaiter().GetResult();
                    attempt++;
                }
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

            int attempt = 1;
            while (true)
            {
                try
                {
                    return await func().ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    if (!ShouldRetry(ex, options) || attempt > options.MaxRetries)
                    {
                        HandleException(ex, "RetryExhausted");
                        throw;
                    }

                    var delay = CalculateRetryDelay(attempt, options);
                    _logger.Warning(ex.Message, "Retry attempt {RetryAttempt}/{MaxRetries} after {RetryDelayMs}ms. Reason: {ErrorMessage}", 
                        attempt, options.MaxRetries, delay.TotalMilliseconds, ex.Message);
                    await Task.Delay(delay).ConfigureAwait(false);
                    attempt++;
                }
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

            int attempt = 1;
            while (true)
            {
                try
                {
                    await action().ConfigureAwait(false);
                    return;
                }
                catch (Exception ex)
                {
                    if (!ShouldRetry(ex, options) || attempt > options.MaxRetries)
                    {
                        HandleException(ex, "RetryExhausted");
                        throw;
                    }

                    var delay = CalculateRetryDelay(attempt, options);
                    _logger.Warning(ex.Message, "Retry attempt {RetryAttempt}/{MaxRetries} after {RetryDelayMs}ms. Reason: {ErrorMessage}", 
                        attempt, options.MaxRetries, delay.TotalMilliseconds, ex.Message);
                    await Task.Delay(delay).ConfigureAwait(false);
                    attempt++;
                }
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

            var circuit = new SimpleCircuit(options);

            _circuitBreakers[serviceName] = circuit;
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
            
            if (!_circuitBreakers.TryGetValue(serviceName, out var circuit))
            {
                RegisterCircuitBreaker(serviceName);
                circuit = _circuitBreakers[serviceName];
            }
            
            try
            {
                // Check circuit state
                if (circuit.IsOpen())
                {
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.Open;
                    _logger.Warning("Circuit for {ServiceName} is open - rejecting request", serviceName);
                    throw new InvalidOperationException($"Circuit for {serviceName} is open");
                }

                if (circuit.Status == CircuitBreakerStatus.Open && !circuit.IsOpen())
                {
                    circuit.Status = CircuitBreakerStatus.HalfOpen;
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.HalfOpen;
                    circuit.Options.OnCircuitHalfOpen?.Invoke(serviceName);
                }

                // Execute guarded action
                action();

                // Success - close circuit
                circuit.ConsecutiveFailures = 0;
                if (circuit.Status != CircuitBreakerStatus.Closed)
                {
                    circuit.Status = CircuitBreakerStatus.Closed;
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
                    circuit.Options.OnCircuitClosed?.Invoke(serviceName);
                }
                else
                {
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
                }
            }
            catch (Exception ex)
            {
                // Determine if this failure counts
                bool countsAsFailure = true;
                if (circuit.Options.FailurePredicate != null)
                {
                    countsAsFailure = circuit.Options.FailurePredicate(ex);
                }
                else if (circuit.Options.FailureExceptions != null && circuit.Options.FailureExceptions.Length > 0)
                {
                    countsAsFailure = circuit.Options.FailureExceptions.Any(t => t.IsInstanceOfType(ex));
                }

                if (countsAsFailure)
                {
                    circuit.ConsecutiveFailures++;
                    if (circuit.Status == CircuitBreakerStatus.HalfOpen || circuit.ConsecutiveFailures >= circuit.Options.FailureThreshold)
                    {
                        circuit.Status = CircuitBreakerStatus.Open;
                        circuit.LastBreakUtc = DateTime.UtcNow;
                        _circuitStatuses[serviceName] = CircuitBreakerStatus.Open;
                        circuit.Options.OnCircuitOpened?.Invoke(serviceName, ex);
                    }
                }

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
            
            if (!_circuitBreakers.TryGetValue(serviceName, out var circuit))
            {
                RegisterCircuitBreaker(serviceName);
                circuit = _circuitBreakers[serviceName];
            }
            
            try
            {
                if (circuit.IsOpen())
                {
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.Open;
                    _logger.Warning("Circuit for {ServiceName} is open - rejecting request", serviceName);
                    throw new InvalidOperationException($"Circuit for {serviceName} is open");
                }

                if (circuit.Status == CircuitBreakerStatus.Open && !circuit.IsOpen())
                {
                    circuit.Status = CircuitBreakerStatus.HalfOpen;
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.HalfOpen;
                    circuit.Options.OnCircuitHalfOpen?.Invoke(serviceName);
                }

                var result = func();

                circuit.ConsecutiveFailures = 0;
                if (circuit.Status != CircuitBreakerStatus.Closed)
                {
                    circuit.Status = CircuitBreakerStatus.Closed;
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
                    circuit.Options.OnCircuitClosed?.Invoke(serviceName);
                }
                else
                {
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
                }

                return result;
            }
            catch (Exception ex)
            {
                bool countsAsFailure = true;
                if (circuit.Options.FailurePredicate != null)
                {
                    countsAsFailure = circuit.Options.FailurePredicate(ex);
                }
                else if (circuit.Options.FailureExceptions != null && circuit.Options.FailureExceptions.Length > 0)
                {
                    countsAsFailure = circuit.Options.FailureExceptions.Any(t => t.IsInstanceOfType(ex));
                }

                if (countsAsFailure)
                {
                    circuit.ConsecutiveFailures++;
                    if (circuit.Status == CircuitBreakerStatus.HalfOpen || circuit.ConsecutiveFailures >= circuit.Options.FailureThreshold)
                    {
                        circuit.Status = CircuitBreakerStatus.Open;
                        circuit.LastBreakUtc = DateTime.UtcNow;
                        _circuitStatuses[serviceName] = CircuitBreakerStatus.Open;
                        circuit.Options.OnCircuitOpened?.Invoke(serviceName, ex);
                    }
                }

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
            
            if (!_circuitBreakers.TryGetValue(serviceName, out var circuit))
            {
                RegisterCircuitBreaker(serviceName);
                circuit = _circuitBreakers[serviceName];
            }
            
            try
            {
                if (circuit.IsOpen())
                {
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.Open;
                    _logger.Warning("Circuit for {ServiceName} is open - rejecting request", serviceName);
                    throw new InvalidOperationException($"Circuit for {serviceName} is open");
                }

                if (circuit.Status == CircuitBreakerStatus.Open && !circuit.IsOpen())
                {
                    circuit.Status = CircuitBreakerStatus.HalfOpen;
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.HalfOpen;
                    circuit.Options.OnCircuitHalfOpen?.Invoke(serviceName);
                }

                var result = await func().ConfigureAwait(false);

                circuit.ConsecutiveFailures = 0;
                if (circuit.Status != CircuitBreakerStatus.Closed)
                {
                    circuit.Status = CircuitBreakerStatus.Closed;
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
                    circuit.Options.OnCircuitClosed?.Invoke(serviceName);
                }
                else
                {
                    _circuitStatuses[serviceName] = CircuitBreakerStatus.Closed;
                }

                return result;
            }
            catch (Exception ex)
            {
                bool countsAsFailure = true;
                if (circuit.Options.FailurePredicate != null)
                {
                    countsAsFailure = circuit.Options.FailurePredicate(ex);
                }
                else if (circuit.Options.FailureExceptions != null && circuit.Options.FailureExceptions.Length > 0)
                {
                    countsAsFailure = circuit.Options.FailureExceptions.Any(t => t.IsInstanceOfType(ex));
                }

                if (countsAsFailure)
                {
                    circuit.ConsecutiveFailures++;
                    if (circuit.Status == CircuitBreakerStatus.HalfOpen || circuit.ConsecutiveFailures >= circuit.Options.FailureThreshold)
                    {
                        circuit.Status = CircuitBreakerStatus.Open;
                        circuit.LastBreakUtc = DateTime.UtcNow;
                        _circuitStatuses[serviceName] = CircuitBreakerStatus.Open;
                        circuit.Options.OnCircuitOpened?.Invoke(serviceName, ex);
                    }
                }

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

                // Avoid cross-project dependencies from DAL. Log the alert; UI layer can subscribe and display.
                DatabaseMonolith.Log(category.ToString(), alert.Name, alert.Notes);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to emit error alert");
            }
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