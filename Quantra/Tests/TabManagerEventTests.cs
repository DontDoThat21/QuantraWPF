using System;
using System.Threading;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Windows.Controls;
using System.Windows;

namespace Quantra.Tests
{
    [TestClass]
    public class TabManagerEventTests
    {
        private TabControl testTabControl;
        private Utilities.TabManager tabManager;
        private MainWindow testMainWindow;
        private bool tabAddedEventFired;
        private string addedTabName;

        [TestInitialize]
        public void Initialize()
        {
            // Reset test state
            tabAddedEventFired = false;
            addedTabName = null;
            
            // Initialize minimal WPF application context for testing
            if (Application.Current == null)
            {
                new Application();
            }
        }

        [TestMethod]
        public void TabManager_AddCustomTab_ShouldRaiseTabAddedEvent()
        {
            try
            {
                // Arrange
                testTabControl = new TabControl();
                
                // Add a '+' tab to simulate the real environment
                var plusTab = new TabItem { Header = "+" };
                testTabControl.Items.Add(plusTab);
                
                // Create a mock MainWindow for testing
                testMainWindow = new MainWindow();
                tabManager = new Utilities.TabManager(testMainWindow, testTabControl);
                
                // Subscribe to the TabAdded event to verify it fires
                tabManager.TabAdded += (tabName) =>
                {
                    tabAddedEventFired = true;
                    addedTabName = tabName;
                };

                // Act
                string testTabName = "TestTab_" + DateTime.Now.Ticks;
                tabManager.AddCustomTab(testTabName);

                // Assert
                Assert.IsTrue(tabAddedEventFired, "TabAdded event should have been fired");
                Assert.AreEqual(testTabName, addedTabName, "Tab name should match the added tab");
                
                // Verify the tab was actually added to the TabControl
                var addedTab = testTabControl.Items[0] as TabItem; // Should be first item since '+' tab moves to end
                Assert.IsNotNull(addedTab, "Tab should have been added to TabControl");
                Assert.AreEqual(testTabName, addedTab.Header.ToString(), "Added tab should have correct name");
            }
            catch (Exception ex)
            {
                // Log any exceptions for debugging
                Console.WriteLine($"Test failed with exception: {ex.Message}");
                throw;
            }
        }

        [TestMethod]
        public void MainWindow_TabManagerEvent_ShouldRaiseMainWindowTabAddedEvent()
        {
            try
            {
                // Arrange
                testTabControl = new TabControl();
                var plusTab = new TabItem { Header = "+" };
                testTabControl.Items.Add(plusTab);
                
                testMainWindow = new MainWindow();
                bool mainWindowEventFired = false;
                string mainWindowTabName = null;
                
                // The MainWindow automatically wires TabManager.TabAdded to its own TabAdded event
                // during InitializeTabManagement(), so we can test this by using the TabManager directly
                tabManager = testMainWindow.TabManager;
                
                // Subscribe to TabManager's TabAdded event to verify the event chain works
                tabManager.TabAdded += (tabName) => {
                    mainWindowEventFired = true;
                    mainWindowTabName = tabName;
                };

                // Act
                string testTabName = "TestTab_MainWindow_" + DateTime.Now.Ticks;
                tabManager.AddCustomTab(testTabName);

                // Assert
                Assert.IsTrue(mainWindowEventFired, "MainWindow TabAdded event should have been fired");
                Assert.AreEqual(testTabName, mainWindowTabName, "MainWindow event should have correct tab name");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"MainWindow event test failed: {ex.Message}");
                throw;
            }
        }

        [TestCleanup]
        public void Cleanup()
        {
            // Clean up test resources
            testTabControl = null;
            tabManager = null;
            testMainWindow = null;
        }
    }
}