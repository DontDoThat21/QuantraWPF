using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;
using Quantra.CrossCutting.Monitoring;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Tests
{
    /// <summary>
    /// Tests to verify performance timing functionality is working correctly
    /// </summary>
    [TestFixture]
    public class PerformanceTimingTests
    {
        private IMonitoringManager _monitoringManager;

        [SetUp]
        public void Setup()
        {
            _monitoringManager = MonitoringManager.Instance;
        }

        /// <summary>
        /// Test that execution timing recording works correctly
        /// </summary>
        [Test]
        public void RecordExecutionTime_ShouldLogTiming()
        {
            // Arrange
            var operationName = "TestOperation";
            var expectedDelay = TimeSpan.FromMilliseconds(50);

            // Act
            var actualDuration = _monitoringManager.RecordExecutionTime(operationName, () =>
            {
                System.Threading.Thread.Sleep(expectedDelay);
            });

            // Assert
            Assert.That(actualDuration, Is.GreaterThan(TimeSpan.FromMilliseconds(40)));
            Assert.That(actualDuration, Is.LessThan(TimeSpan.FromMilliseconds(200)));
            
            // Verify metrics were recorded
            var metrics = _monitoringManager.GetMetrics(operationName);
            Assert.That(metrics, Is.Not.Null);
            Assert.That(metrics.ExecutionCount, Is.EqualTo(1));
            Assert.That(metrics.SuccessCount, Is.EqualTo(1));
        }

        /// <summary>
        /// Test that async execution timing recording works correctly
        /// </summary>
        [Test]
        public async Task RecordExecutionTimeAsync_ShouldLogTiming()
        {
            // Arrange
            var operationName = "TestAsyncOperation";
            var expectedDelay = TimeSpan.FromMilliseconds(50);

            // Act
            var (result, actualDuration) = await _monitoringManager.RecordExecutionTimeAsync(operationName, async () =>
            {
                await Task.Delay(expectedDelay);
                return "test result";
            });

            // Assert
            Assert.That(result, Is.EqualTo("test result"));
            Assert.That(actualDuration, Is.GreaterThan(TimeSpan.FromMilliseconds(40)));
            Assert.That(actualDuration, Is.LessThan(TimeSpan.FromMilliseconds(200)));
            
            // Verify metrics were recorded
            var metrics = _monitoringManager.GetMetrics(operationName);
            Assert.That(metrics, Is.Not.Null);
            Assert.That(metrics.ExecutionCount, Is.EqualTo(1));
            Assert.That(metrics.SuccessCount, Is.EqualTo(1));
        }

        /// <summary>
        /// Test that timing logs are created for indicator operations
        /// This test verifies the logging infrastructure without requiring a full UI
        /// </summary>
        [Test]
        public void IndicatorTiming_ShouldCreateTimingLogs()
        {
            // Arrange
            var operationName = "GetRSI_AAPL_1day";
            
            // Act - Simulate an indicator calculation
            var (result, duration) = _monitoringManager.RecordExecutionTime(operationName, () =>
            {
                // Simulate RSI calculation time
                System.Threading.Thread.Sleep(10);
                return 65.5; // Mock RSI value
            });

            // Assert
            Assert.That(result, Is.EqualTo(65.5));
            Assert.That(duration, Is.GreaterThan(TimeSpan.Zero));
            
            // Verify the operation was logged
            var metrics = _monitoringManager.GetMetrics(operationName);
            Assert.That(metrics, Is.Not.Null);
            Assert.That(metrics.Name, Is.EqualTo(operationName));
            Assert.That(metrics.AverageExecutionTimeMs, Is.GreaterThan(0));
        }

        /// <summary>
        /// Test that UI update timing logs are created
        /// This test verifies the logging infrastructure for UI operations
        /// </summary>
        [Test]
        public async Task UIUpdateTiming_ShouldCreateTimingLogs()
        {
            // Arrange
            var operationName = "UIUpdate_SetIndicators_AAPL";
            
            // Act - Simulate a UI update operation
            var duration = await _monitoringManager.RecordExecutionTimeAsync(operationName, async () =>
            {
                // Simulate UI update time
                await Task.Delay(5);
            });

            // Assert
            Assert.That(duration, Is.GreaterThan(TimeSpan.Zero));
            
            // Verify the operation was logged
            var metrics = _monitoringManager.GetMetrics(operationName);
            Assert.That(metrics, Is.Not.Null);
            Assert.That(metrics.Name, Is.EqualTo(operationName));
            Assert.That(metrics.ExecutionCount, Is.EqualTo(1));
        }
    }
}