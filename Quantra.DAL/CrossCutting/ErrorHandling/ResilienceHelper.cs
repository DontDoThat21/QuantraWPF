using System;
using System.Threading.Tasks;

namespace Quantra.CrossCutting.ErrorHandling
{
    /// <summary>
    /// Static helper for resilient operations.
    /// </summary>
    public static class ResilienceHelper
    {
        private static readonly IErrorHandlingManager _manager = ErrorHandlingManager.Instance;
        
        /// <summary>
        /// Initializes the error handling system.
        /// </summary>
        public static void Initialize()
        {
            _manager.Initialize();
        }
        
        /// <summary>
        /// Executes an action with retry policies.
        /// </summary>
        public static void Retry(Action action, RetryOptions options = null)
        {
            _manager.ExecuteWithRetry(action, options);
        }
        
        /// <summary>
        /// Executes a function with retry policies.
        /// </summary>
        public static T Retry<T>(Func<T> func, RetryOptions options = null)
        {
            return _manager.ExecuteWithRetry(func, options);
        }
        
        /// <summary>
        /// Executes an async function with retry policies.
        /// </summary>
        public static Task<T> RetryAsync<T>(Func<Task<T>> func, RetryOptions options = null)
        {
            return _manager.ExecuteWithRetryAsync(func, options);
        }
        
        /// <summary>
        /// Executes an async action with retry policies.
        /// </summary>
        public static Task RetryAsync(Func<Task> action, RetryOptions options = null)
        {
            return _manager.ExecuteWithRetryAsync(action, options);
        }
        
        /// <summary>
        /// Executes an action through a circuit breaker.
        /// </summary>
        public static void WithCircuitBreaker(string serviceName, Action action)
        {
            _manager.ExecuteWithCircuitBreaker(serviceName, action);
        }
        
        /// <summary>
        /// Executes a function through a circuit breaker.
        /// </summary>
        public static T WithCircuitBreaker<T>(string serviceName, Func<T> func)
        {
            return _manager.ExecuteWithCircuitBreaker(serviceName, func);
        }
        
        /// <summary>
        /// Executes an async function through a circuit breaker.
        /// </summary>
        public static Task<T> WithCircuitBreakerAsync<T>(string serviceName, Func<Task<T>> func)
        {
            return _manager.ExecuteWithCircuitBreakerAsync(serviceName, func);
        }
        
        /// <summary>
        /// Gets the circuit breaker status for a service.
        /// </summary>
        public static CircuitBreakerStatus GetCircuitStatus(string serviceName)
        {
            return _manager.GetCircuitStatus(serviceName);
        }
        
        /// <summary>
        /// Handles an exception with appropriate actions based on its category.
        /// </summary>
        public static void HandleException(Exception exception, string context = null)
        {
            _manager.HandleException(exception, context);
        }
        
        /// <summary>
        /// Determines the category of an exception.
        /// </summary>
        public static ErrorCategory CategorizeException(Exception exception)
        {
            return _manager.CategorizeException(exception);
        }
        
        /// <summary>
        /// Makes a resilient call to an external API with retry and circuit breaker.
        /// </summary>
        public static T ExternalApiCall<T>(string apiName, Func<T> apiCall)
        {
            // Combine retry and circuit breaker for maximum resilience
            return WithCircuitBreaker(apiName, () => 
                Retry(apiCall, RetryOptions.ForUserFacingOperation()));
        }
        
        /// <summary>
        /// Makes a resilient async call to an external API with retry and circuit breaker.
        /// </summary>
        public static Task<T> ExternalApiCallAsync<T>(string apiName, Func<Task<T>> apiCall)
        {
            // Combine retry and circuit breaker for maximum resilience
            return WithCircuitBreakerAsync(apiName, () => 
                RetryAsync(apiCall, RetryOptions.ForUserFacingOperation()));
        }
    }
}