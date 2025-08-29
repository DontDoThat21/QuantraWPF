using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Utilities;
using Xunit;

namespace Quantra.Tests.Utilities
{
    public class ConcurrentTaskThrottlerTests
    {
        [Fact]
        public void Constructor_WithValidMaxDegree_InitializesCorrectly()
        {
            // Arrange & Act
            using var throttler = new ConcurrentTaskThrottler(4);

            // Assert
            Assert.Equal(4, throttler.MaxDegreeOfParallelism);
            Assert.Equal(4, throttler.AvailableCount);
        }

        [Fact]
        public void Constructor_WithInvalidMaxDegree_ThrowsArgumentException()
        {
            // Assert
            Assert.Throws<ArgumentException>(() => new ConcurrentTaskThrottler(0));
            Assert.Throws<ArgumentException>(() => new ConcurrentTaskThrottler(-1));
        }

        [Fact]
        public async Task ExecuteThrottledAsync_WithEmptyTasks_ReturnsEmptyArray()
        {
            // Arrange
            using var throttler = new ConcurrentTaskThrottler(2);
            var taskFactories = new List<Func<Task<int>>>();

            // Act
            var results = await throttler.ExecuteThrottledAsync(taskFactories);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public async Task ExecuteThrottledAsync_WithNullTasks_ThrowsArgumentNullException()
        {
            // Arrange
            using var throttler = new ConcurrentTaskThrottler(2);

            // Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() => 
                throttler.ExecuteThrottledAsync<int>(null));
        }

        [Fact]
        public async Task ExecuteThrottledAsync_ThrottlesConcurrentTasks()
        {
            // Arrange
            const int maxDegree = 2;
            const int taskCount = 6;
            const int taskDurationMs = 100;
            
            using var throttler = new ConcurrentTaskThrottler(maxDegree);
            var concurrentExecutions = 0;
            var maxConcurrentExecutions = 0;
            var lockObject = new object();

            var taskFactories = Enumerable.Range(0, taskCount).Select(i => 
                new Func<Task<int>>(async () =>
                {
                    lock (lockObject)
                    {
                        concurrentExecutions++;
                        maxConcurrentExecutions = Math.Max(maxConcurrentExecutions, concurrentExecutions);
                    }

                    await Task.Delay(taskDurationMs);

                    lock (lockObject)
                    {
                        concurrentExecutions--;
                    }

                    return i;
                })).ToList();

            // Act
            var stopwatch = Stopwatch.StartNew();
            var results = await throttler.ExecuteThrottledAsync(taskFactories);
            stopwatch.Stop();

            // Assert
            Assert.Equal(taskCount, results.Length);
            Assert.Equal(Enumerable.Range(0, taskCount), results.OrderBy(x => x));
            Assert.True(maxConcurrentExecutions <= maxDegree, 
                $"Max concurrent executions ({maxConcurrentExecutions}) exceeded max degree ({maxDegree})");
            
            // Verify that throttling actually occurred (should take longer than if all tasks ran concurrently)
            var expectedMinTime = (taskCount / maxDegree - 1) * taskDurationMs;
            Assert.True(stopwatch.ElapsedMilliseconds >= expectedMinTime * 0.8, // Allow 20% tolerance
                $"Execution time ({stopwatch.ElapsedMilliseconds}ms) suggests tasks weren't properly throttled");
        }

        [Fact]
        public async Task ExecuteThrottledBatchAsync_ProcessesAllItems()
        {
            // Arrange
            using var throttler = new ConcurrentTaskThrottler(3);
            var items = Enumerable.Range(1, 10).ToList();
            
            async Task<int> SquareOperation(int x)
            {
                await Task.Delay(10); // Simulate work
                return x * x;
            }

            // Act
            var results = await throttler.ExecuteThrottledBatchAsync(items, SquareOperation);

            // Assert
            Assert.Equal(items.Count, results.Count);
            foreach (var item in items)
            {
                Assert.True(results.ContainsKey(item));
                Assert.Equal(item * item, results[item]);
            }
        }

        [Fact]
        public async Task ExecuteThrottledBatchAsync_HandlesExceptions()
        {
            // Arrange
            using var throttler = new ConcurrentTaskThrottler(2);
            var items = new[] { 1, 2, 3, 4, 5 };
            
            async Task<int> FaultyOperation(int x)
            {
                await Task.Delay(10);
                if (x == 3) throw new InvalidOperationException("Test exception");
                return x * 2;
            }

            // Act
            var results = await throttler.ExecuteThrottledBatchAsync(items, FaultyOperation);

            // Assert
            Assert.Equal(5, results.Count);
            Assert.Equal(2, results[1]);
            Assert.Equal(4, results[2]);
            Assert.Equal(0, results[3]); // Default value for failed operation
            Assert.Equal(8, results[4]);
            Assert.Equal(10, results[5]);
        }

        [Fact]
        public async Task ExecuteThrottledTaskAsync_SingleTask_ExecutesCorrectly()
        {
            // Arrange
            using var throttler = new ConcurrentTaskThrottler(1);
            const int expectedResult = 42;

            // Act
            var result = await throttler.ExecuteThrottledTaskAsync(async () =>
            {
                await Task.Delay(10);
                return expectedResult;
            });

            // Assert
            Assert.Equal(expectedResult, result);
        }

        [Fact]
        public async Task ExecuteThrottledTaskAsync_NoReturnValue_ExecutesCorrectly()
        {
            // Arrange
            using var throttler = new ConcurrentTaskThrottler(1);
            var executed = false;

            // Act
            await throttler.ExecuteThrottledTaskAsync(async () =>
            {
                await Task.Delay(10);
                executed = true;
            });

            // Assert
            Assert.True(executed);
        }

        [Fact]
        public async Task AvailableCount_ReflectsCurrentUsage()
        {
            // Arrange
            const int maxDegree = 3;
            using var throttler = new ConcurrentTaskThrottler(maxDegree);
            var taskStarted = new TaskCompletionSource<bool>();
            var releaseTask = new TaskCompletionSource<bool>();

            // Start a task that will block
            var blockingTask = throttler.ExecuteThrottledTaskAsync(async () =>
            {
                taskStarted.SetResult(true);
                await releaseTask.Task;
                return 1;
            });

            // Wait for the task to start
            await taskStarted.Task;

            // Assert
            Assert.Equal(maxDegree - 1, throttler.AvailableCount);

            // Release the blocking task
            releaseTask.SetResult(true);
            await blockingTask;

            // Assert count is back to full
            Assert.Equal(maxDegree, throttler.AvailableCount);
        }
    }
}