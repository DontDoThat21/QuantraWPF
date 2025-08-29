using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Quantra.Utilities
{
    /// <summary>
    /// Utility class for throttling concurrent background tasks to prevent thread pool exhaustion and UI congestion
    /// </summary>
    public class ConcurrentTaskThrottler : IDisposable
    {
        private readonly SemaphoreSlim _semaphore;
        private readonly int _maxDegreeOfParallelism;
        private bool _disposed = false;

        /// <summary>
        /// Initializes a new instance of ConcurrentTaskThrottler
        /// </summary>
        /// <param name="maxDegreeOfParallelism">Maximum number of concurrent tasks (default: 6)</param>
        public ConcurrentTaskThrottler(int maxDegreeOfParallelism = 6)
        {
            if (maxDegreeOfParallelism <= 0)
                throw new ArgumentException("Max degree of parallelism must be greater than zero", nameof(maxDegreeOfParallelism));

            _maxDegreeOfParallelism = maxDegreeOfParallelism;
            _semaphore = new SemaphoreSlim(maxDegreeOfParallelism, maxDegreeOfParallelism);
        }

        /// <summary>
        /// Gets the maximum degree of parallelism
        /// </summary>
        public int MaxDegreeOfParallelism => _maxDegreeOfParallelism;

        /// <summary>
        /// Gets the current number of available slots
        /// </summary>
        public int AvailableCount => _semaphore.CurrentCount;

        /// <summary>
        /// Executes a collection of tasks with throttling
        /// </summary>
        /// <typeparam name="T">The type of the task result</typeparam>
        /// <param name="taskFactories">Functions that create tasks to execute</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Array of results from all tasks</returns>
        public async Task<T[]> ExecuteThrottledAsync<T>(
            IEnumerable<Func<Task<T>>> taskFactories,
            CancellationToken cancellationToken = default)
        {
            if (taskFactories == null)
                throw new ArgumentNullException(nameof(taskFactories));

            var factories = taskFactories.ToList();
            if (factories.Count == 0)
                return Array.Empty<T>();

            var throttledTasks = factories.Select(factory => ExecuteThrottledTaskAsync(factory, cancellationToken));
            return await Task.WhenAll(throttledTasks);
        }

        /// <summary>
        /// Executes a collection of tasks with throttling and returns results as they complete
        /// </summary>
        /// <typeparam name="T">The type of the task result</typeparam>
        /// <param name="taskFactories">Functions that create tasks to execute</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Results as they complete</returns>
        public async IAsyncEnumerable<T> ExecuteThrottledAsyncEnumerable<T>(
            IEnumerable<Func<Task<T>>> taskFactories,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (taskFactories == null)
                throw new ArgumentNullException(nameof(taskFactories));

            var factories = taskFactories.ToList();
            if (factories.Count == 0)
                yield break;

            var throttledTasks = factories.Select(factory => ExecuteThrottledTaskAsync(factory, cancellationToken)).ToList();

            while (throttledTasks.Count > 0)
            {
                var completedTask = await Task.WhenAny(throttledTasks);
                throttledTasks.Remove(completedTask);
                
                T result;
                try
                {
                    result = await completedTask;
                }
                catch (Exception ex)
                {
                    // Log the error but continue processing other tasks
                    DatabaseMonolith.Log("Warning", "Throttled task failed", ex.ToString());
                    continue;
                }
                
                yield return result;
            }
        }

        /// <summary>
        /// Executes a single task with throttling
        /// </summary>
        /// <typeparam name="T">The type of the task result</typeparam>
        /// <param name="taskFactory">Function that creates the task to execute</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>The task result</returns>
        public async Task<T> ExecuteThrottledTaskAsync<T>(
            Func<Task<T>> taskFactory,
            CancellationToken cancellationToken = default)
        {
            if (taskFactory == null)
                throw new ArgumentNullException(nameof(taskFactory));

            await _semaphore.WaitAsync(cancellationToken);
            try
            {
                return await taskFactory();
            }
            finally
            {
                _semaphore.Release();
            }
        }

        /// <summary>
        /// Executes a single task with throttling (no return value)
        /// </summary>
        /// <param name="taskFactory">Function that creates the task to execute</param>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task ExecuteThrottledTaskAsync(
            Func<Task> taskFactory,
            CancellationToken cancellationToken = default)
        {
            if (taskFactory == null)
                throw new ArgumentNullException(nameof(taskFactory));

            await _semaphore.WaitAsync(cancellationToken);
            try
            {
                await taskFactory();
            }
            finally
            {
                _semaphore.Release();
            }
        }

        /// <summary>
        /// Executes a collection of input items through a throttled operation
        /// </summary>
        /// <typeparam name="TInput">The input type</typeparam>
        /// <typeparam name="TOutput">The output type</typeparam>
        /// <param name="items">Items to process</param>
        /// <param name="operation">Operation to perform on each item</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Dictionary mapping input items to their results</returns>
        public async Task<Dictionary<TInput, TOutput>> ExecuteThrottledBatchAsync<TInput, TOutput>(
            IEnumerable<TInput> items,
            Func<TInput, Task<TOutput>> operation,
            CancellationToken cancellationToken = default)
        {
            if (items == null)
                throw new ArgumentNullException(nameof(items));
            if (operation == null)
                throw new ArgumentNullException(nameof(operation));

            var itemList = items.ToList();
            var results = new Dictionary<TInput, TOutput>();

            if (itemList.Count == 0)
                return results;

            var taskFactories = itemList.Select<TInput, Func<Task<KeyValuePair<TInput, TOutput>>>>(item =>
                async () =>
                {
                    try
                    {
                        var result = await operation(item);
                        return new KeyValuePair<TInput, TOutput>(item, result);
                    }
                    catch (Exception ex)
                    {
                        DatabaseMonolith.Log("Warning", $"Throttled batch operation failed for item: {item}", ex.ToString());
                        return new KeyValuePair<TInput, TOutput>(item, default(TOutput));
                    }
                });

            var batchResults = await ExecuteThrottledAsync(taskFactories, cancellationToken);
            
            foreach (var result in batchResults)
            {
                results[result.Key] = result.Value;
            }

            return results;
        }

        /// <summary>
        /// Disposes the throttler and releases resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the throttler and releases resources
        /// </summary>
        /// <param name="disposing">True if disposing managed resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                _semaphore?.Dispose();
                _disposed = true;
            }
        }
    }
}