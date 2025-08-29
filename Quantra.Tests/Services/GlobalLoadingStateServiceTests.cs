using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Threading.Tasks;
using Quantra.Services;

namespace Quantra.Tests.Services
{
    [TestClass]
    public class GlobalLoadingStateServiceTests
    {
        [TestInitialize]
        public void TestInitialize()
        {
            // Ensure clean state before each test
            GlobalLoadingStateService.SetLoadingState(false);
        }

        [TestMethod]
        public void SetLoadingState_InitialState_IsFalse()
        {
            // Arrange & Act & Assert
            Assert.IsFalse(GlobalLoadingStateService.IsLoading);
        }

        [TestMethod]
        public void SetLoadingState_ToTrue_SetsLoadingState()
        {
            // Arrange & Act
            GlobalLoadingStateService.SetLoadingState(true);

            // Assert
            Assert.IsTrue(GlobalLoadingStateService.IsLoading);
        }

        [TestMethod]
        public void SetLoadingState_ChangesState_FiresEvent()
        {
            // Arrange
            bool eventFired = false;
            bool receivedState = false;

            GlobalLoadingStateService.LoadingStateChanged += (isLoading) =>
            {
                eventFired = true;
                receivedState = isLoading;
            };

            // Act
            GlobalLoadingStateService.SetLoadingState(true);

            // Assert
            Assert.IsTrue(eventFired);
            Assert.IsTrue(receivedState);
        }

        [TestMethod]
        public void SetLoadingState_SameState_DoesNotFireEvent()
        {
            // Arrange
            GlobalLoadingStateService.SetLoadingState(false); // Ensure initial state
            bool eventFired = false;

            GlobalLoadingStateService.LoadingStateChanged += (isLoading) =>
            {
                eventFired = true;
            };

            // Act
            GlobalLoadingStateService.SetLoadingState(false); // Set to same state

            // Assert
            Assert.IsFalse(eventFired);
        }

        [TestMethod]
        public async Task WithLoadingState_Task_ManagesStateCorrectly()
        {
            // Arrange
            bool loadingStateDuringTask = false;
            var taskStarted = new TaskCompletionSource<bool>();
            var testTask = Task.Run(async () =>
            {
                taskStarted.SetResult(true);
                loadingStateDuringTask = GlobalLoadingStateService.IsLoading;
                await Task.Delay(50);
            });

            // Act
            var wrappedTask = GlobalLoadingStateService.WithLoadingState(testTask);
            await taskStarted.Task; // Wait for task to start
            await wrappedTask;

            // Assert
            Assert.IsTrue(loadingStateDuringTask, "Loading state should be true during task execution");
            Assert.IsFalse(GlobalLoadingStateService.IsLoading, "Loading state should be false after task completion");
        }

        [TestMethod]
        public async Task WithLoadingState_TaskWithResult_ManagesStateCorrectly()
        {
            // Arrange
            const string expectedResult = "test result";
            bool loadingStateDuringTask = false;
            var taskStarted = new TaskCompletionSource<bool>();
            
            var testTask = Task.Run(async () =>
            {
                taskStarted.SetResult(true);
                loadingStateDuringTask = GlobalLoadingStateService.IsLoading;
                await Task.Delay(50);
                return expectedResult;
            });

            // Act
            var wrappedTask = GlobalLoadingStateService.WithLoadingState(testTask);
            await taskStarted.Task; // Wait for task to start
            var result = await wrappedTask;

            // Assert
            Assert.AreEqual(expectedResult, result);
            Assert.IsTrue(loadingStateDuringTask, "Loading state should be true during task execution");
            Assert.IsFalse(GlobalLoadingStateService.IsLoading, "Loading state should be false after task completion");
        }

        [TestMethod]
        public async Task WithLoadingState_TaskThrowsException_StillResetsLoadingState()
        {
            // Arrange
            var testTask = Task.Run(async () =>
            {
                await Task.Delay(50);
                throw new InvalidOperationException("Test exception");
            });

            // Act & Assert
            try
            {
                await GlobalLoadingStateService.WithLoadingState(testTask);
                Assert.Fail("Expected exception was not thrown");
            }
            catch (InvalidOperationException)
            {
                // Expected exception
            }

            // Assert
            Assert.IsFalse(GlobalLoadingStateService.IsLoading, "Loading state should be false even when task throws exception");
        }

        [TestMethod]
        public void LoadingStateChanged_MultipleSubscribers_AllReceiveEvents()
        {
            // Arrange
            bool subscriber1Called = false;
            bool subscriber2Called = false;
            bool subscriber1State = false;
            bool subscriber2State = false;

            GlobalLoadingStateService.LoadingStateChanged += (isLoading) =>
            {
                subscriber1Called = true;
                subscriber1State = isLoading;
            };

            GlobalLoadingStateService.LoadingStateChanged += (isLoading) =>
            {
                subscriber2Called = true;
                subscriber2State = isLoading;
            };

            // Act
            GlobalLoadingStateService.SetLoadingState(true);

            // Assert
            Assert.IsTrue(subscriber1Called);
            Assert.IsTrue(subscriber2Called);
            Assert.IsTrue(subscriber1State);
            Assert.IsTrue(subscriber2State);
        }
    }
}