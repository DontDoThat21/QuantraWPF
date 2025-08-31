using System;
using System.Threading.Tasks;
using Quantra.CrossCutting;

namespace Quantra.CrossCutting.ErrorHandling
{
    /// <summary>
    /// Centralized error handling management interface.
    /// </summary>
    public interface IErrorHandlingManager : ICrossCuttingModule
    {
        /// <summary>
        /// Categorizes an exception.
        /// </summary>
        ErrorCategory CategorizeException(Exception exception);
        
        /// <summary>
        /// Handles an exception with appropriate logic based on its category.
        /// </summary>
        void HandleException(Exception exception, string context = null);
        
        /// <summary>
        /// Executes an action with retry policies based on the operation's criticality.
        /// </summary>
        void ExecuteWithRetry(Action action, RetryOptions options = null);
        
        /// <summary>
        /// Executes a function with retry policies based on the operation's criticality.
        /// </summary>
        T ExecuteWithRetry<T>(Func<T> func, RetryOptions options = null);
        
        /// <summary>
        /// Executes an async function with retry policies based on the operation's criticality.
        /// </summary>
        Task<T> ExecuteWithRetryAsync<T>(Func<Task<T>> func, RetryOptions options = null);
        
        /// <summary>
        /// Executes an async action with retry policies based on the operation's criticality.
        /// </summary>
        Task ExecuteWithRetryAsync(Func<Task> action, RetryOptions options = null);
        
        /// <summary>
        /// Registers a circuit breaker for a specific service.
        /// </summary>
        void RegisterCircuitBreaker(string serviceName, CircuitBreakerOptions options = null);
        
        /// <summary>
        /// Executes an action through a circuit breaker for a service.
        /// </summary>
        void ExecuteWithCircuitBreaker(string serviceName, Action action);
        
        /// <summary>
        /// Executes a function through a circuit breaker for a service.
        /// </summary>
        T ExecuteWithCircuitBreaker<T>(string serviceName, Func<T> func);
        
        /// <summary>
        /// Executes an async function through a circuit breaker for a service.
        /// </summary>
        Task<T> ExecuteWithCircuitBreakerAsync<T>(string serviceName, Func<Task<T>> func);
        
        /// <summary>
        /// Gets the current status of a circuit breaker.
        /// </summary>
        CircuitBreakerStatus GetCircuitStatus(string serviceName);
    }
}