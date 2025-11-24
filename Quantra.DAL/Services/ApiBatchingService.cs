using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Utilities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for batching API operations to improve performance and respect rate limits
    /// </summary>
    public class ApiBatchingService : IDisposable
    {
        private readonly Dictionary<string, DateTime> _lastApiCall;
        private readonly Dictionary<string, TimeSpan> _apiIntervals;
        private readonly object _lock = new object();
        private readonly ConcurrentTaskThrottler _taskThrottler;
        private bool _disposed = false;

        public ApiBatchingService()
        {
            _lastApiCall = new Dictionary<string, DateTime>();
            _apiIntervals = new Dictionary<string, TimeSpan>
            {
                ["AlphaVantage"] = TimeSpan.FromSeconds(12), // Alpha Vantage free tier limit
                ["Default"] = TimeSpan.FromSeconds(1)
            };

            // Initialize throttler with a reasonable default for API operations
            _taskThrottler = new ConcurrentTaskThrottler(4);
        }

        /// <summary>
        /// Executes a batch of API operations with rate limiting and error handling
        /// </summary>
        /// <typeparam name="TInput">Input type for batch operation</typeparam>
        /// <typeparam name="TOutput">Output type for batch operation</typeparam>
        /// <param name="items">Items to process</param>
        /// <param name="operation">Operation to perform on each item</param>
        /// <param name="batchSize">Size of each batch</param>
        /// <param name="apiProvider">API provider name for rate limiting</param>
        /// <param name="delayBetweenBatches">Additional delay between batches</param>
        /// <returns>Dictionary mapping input items to their results</returns>
        public async Task<Dictionary<TInput, TOutput>> ExecuteBatch<TInput, TOutput>(
            IEnumerable<TInput> items,
            Func<TInput, Task<TOutput>> operation,
            int batchSize = 5,
            string apiProvider = "Default",
            TimeSpan? delayBetweenBatches = null)
        {
            var itemList = items.ToList();
            var results = new Dictionary<TInput, TOutput>();

            if (itemList.Count == 0)
                return results;

            //DatabaseMonolith.Log("Info", $"Starting batch operation for {itemList.Count} items (batch size: {batchSize})");

            var delay = delayBetweenBatches ?? _apiIntervals.GetValueOrDefault(apiProvider, _apiIntervals["Default"]);

            for (int i = 0; i < itemList.Count; i += batchSize)
            {
                var batch = itemList.Skip(i).Take(batchSize).ToList();

                // Apply rate limiting
                await WaitForRateLimit(apiProvider);

                // Process batch concurrently with throttling to prevent thread pool exhaustion
                var batchTaskFactories = batch.Select(item =>
                    new Func<Task<KeyValuePair<TInput, TOutput>>>(async () =>
                    {
                        try
                        {
                            var result = await operation(item);
                            return new KeyValuePair<TInput, TOutput>(item, result);
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Warning", $"Batch operation failed for item: {item}", ex.ToString());
                            return new KeyValuePair<TInput, TOutput>(item, default(TOutput));
                        }
                    }));

                var batchResults = await _taskThrottler.ExecuteThrottledAsync(batchTaskFactories);

                // Collect results
                foreach (var result in batchResults)
                {
                    results[result.Key] = result.Value;
                }

                // Delay between batches if more items remain
                if (i + batchSize < itemList.Count)
                {
                    await Task.Delay(delay);
                }

                //DatabaseMonolith.Log("Debug", $"Completed batch {i / batchSize + 1}/{(itemList.Count + batchSize - 1) / batchSize}");
            }

            //DatabaseMonolith.Log("Info", $"Batch operation completed. Processed {results.Count}/{itemList.Count} items successfully");
            return results;
        }

        /// <summary>
        /// Waits for API rate limit if necessary
        /// </summary>
        /// <param name="apiProvider">API provider name</param>
        private async Task WaitForRateLimit(string apiProvider)
        {
            lock (_lock)
            {
                if (_lastApiCall.TryGetValue(apiProvider, out DateTime lastCall))
                {
                    var interval = _apiIntervals.GetValueOrDefault(apiProvider, _apiIntervals["Default"]);
                    var timeSinceLastCall = DateTime.UtcNow - lastCall;

                    if (timeSinceLastCall < interval)
                    {
                        var waitTime = interval - timeSinceLastCall;
                        Task.Delay(waitTime).Wait();
                    }
                }

                _lastApiCall[apiProvider] = DateTime.UtcNow;
            }
        }

        /// <summary>
        /// Sets a custom rate limit for an API provider
        /// </summary>
        /// <param name="apiProvider">API provider name</param>
        /// <param name="interval">Minimum interval between calls</param>
        public void SetRateLimit(string apiProvider, TimeSpan interval)
        {
            lock (_lock)
            {
                _apiIntervals[apiProvider] = interval;
            }
        }

        /// <summary>
        /// Disposes the service and releases resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the service and releases resources
        /// </summary>
        /// <param name="disposing">True if disposing managed resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                _taskThrottler?.Dispose();
                _disposed = true;
            }
        }
    }
}