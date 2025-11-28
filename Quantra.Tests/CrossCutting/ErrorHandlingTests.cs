using System;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.CrossCutting;
using Quantra.CrossCutting.ErrorHandling;

namespace Quantra.Tests.CrossCutting
{
    [TestClass]
    public class ErrorHandlingTests
    {
        [TestInitialize]
        public void Setup()
        {
            // Initialize the cross-cutting framework
            CrossCuttingRegistry.Initialize();
        }

        [TestMethod]
        public void ErrorHandlingManager_Initialization_ShouldSucceed()
        {
            // Arrange
            var manager = ErrorHandlingManager.Instance;

            // Act
            manager.Initialize();

            // Assert
            Assert.IsNotNull(manager);
        }

        [TestMethod]
        public void ResilienceHelper_RetrySuccessful_ShouldNotThrow()
        {
            // Arrange
            int counter = 0;
            Action testAction = () =>
            {
                counter++;
                if (counter < 3)
                {
                    throw new TimeoutException("Simulated timeout");
                }
            };

            // Act
            ResilienceHelper.Retry(testAction);

            // Assert
            Assert.AreEqual(3, counter); // Should have tried 3 times before success
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void ResilienceHelper_RetryExhausted_ShouldThrowOriginalException()
        {
            // Arrange
            var options = new RetryOptions { MaxRetries = 2 };

            // Act - should throw after retries are exhausted
            ResilienceHelper.Retry(() => throw new InvalidOperationException("Simulated error"), options);
        }

        [TestMethod]
        public async Task ResilienceHelper_AsyncRetry_ShouldWork()
        {
            // Arrange
            int counter = 0;
            Func<Task<string>> testFunc = async () =>
            {
                counter++;
                if (counter < 2)
                {
                    await Task.Delay(10); // Small delay to simulate async work
                    throw new HttpRequestException("Simulated network error");
                }
                return "Success";
            };

            // Act
            string result = await ResilienceHelper.RetryAsync(testFunc);

            // Assert
            Assert.AreEqual("Success", result);
            Assert.AreEqual(2, counter); // Should have tried 2 times before success
        }

        [TestMethod]
        public void ExceptionExtensions_IsTransient_ShouldIdentifyTransientErrors()
        {
            // Arrange
            var timeout = new TimeoutException();
            var http = new HttpRequestException("503 Service Unavailable");
            var invalidOp = new InvalidOperationException("Business logic error");

            // Act
            bool isTimeoutTransient = timeout.IsTransient();
            bool isHttpTransient = http.IsTransient();
            bool isInvalidOpTransient = invalidOp.IsTransient();

            // Assert
            Assert.IsTrue(isTimeoutTransient, "Timeout should be considered transient");
            Assert.IsTrue(isHttpTransient, "HTTP errors should be considered transient");
            Assert.IsFalse(isInvalidOpTransient, "InvalidOperation should not be considered transient");
        }

        [TestMethod]
        public void ResilienceHelper_CircuitBreaker_ShouldWorkForSuccessfulCalls()
        {
            // Arrange
            string serviceName = "TestService";

            // Register a circuit breaker
            ResilienceHelper.Initialize();

            // Act
            var status1 = ResilienceHelper.GetCircuitStatus(serviceName);

            // Make a successful call
            string result = ResilienceHelper.WithCircuitBreaker(serviceName, () => "Success");

            // Check status after successful call
            var status2 = ResilienceHelper.GetCircuitStatus(serviceName);

            // Assert
            Assert.AreEqual(CircuitBreakerStatus.NotRegistered, status1);
            Assert.AreEqual("Success", result);
            Assert.AreEqual(CircuitBreakerStatus.Closed, status2);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void ErrorHandling_CategorizeException_ShouldCategorizeCorrectly()
        {
            // Arrange
            var timeout = new TimeoutException("Operation timed out");
            var http = new HttpRequestException("Connection failed");
            var invalidOp = new InvalidOperationException("Invalid operation");

            // Act
            var timeoutCategory = ResilienceHelper.CategorizeException(timeout);
            var httpCategory = ResilienceHelper.CategorizeException(http);
            var invalidOpCategory = ResilienceHelper.CategorizeException(invalidOp);

            // Assert
            Assert.AreEqual(ErrorCategory.Transient, timeoutCategory);
            Assert.AreEqual(ErrorCategory.NetworkError, httpCategory);
            Assert.AreEqual(ErrorCategory.UserError, invalidOpCategory);

            // Verify exception handling works
            ResilienceHelper.HandleException(invalidOp);

            // This should throw after being handled
            throw invalidOp;
        }
    }
}