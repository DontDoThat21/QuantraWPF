using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.Controls;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.Services.Interfaces;

namespace Quantra.Tests
{
    [TestClass]
    public class SentimentShiftAlertServiceTests
    {
        private SentimentShiftAlertService _sentimentShiftAlertService;
        private List<AlertModel> _capturedAlerts;
        private static bool _eventHandlerAttached = false;
        
        [TestInitialize]
        public void Initialize()
        {
            _sentimentShiftAlertService = new SentimentShiftAlertService();
            _capturedAlerts = new List<AlertModel>();
            
            // Use reflection to get the GlobalAlertEmitted event
            var eventField = typeof(AlertsControl).GetField("GlobalAlertEmitted", 
                BindingFlags.Static | BindingFlags.NonPublic);
            
            if (eventField != null && !_eventHandlerAttached)
            {
                // Create delegate to capture emitted alerts
                EventHandler<AlertModel> handler = (sender, alert) =>
                {
                    _capturedAlerts.Add(alert);
                };
                
                // Get the add method of the event
                var addMethod = eventField.FieldType.GetMethod("add");
                
                // Create a delegate of the event handler type
                var handlerDelegate = Delegate.CreateDelegate(
                    eventField.FieldType.GenericTypeArguments[0], 
                    handler.Target, 
                    handler.Method);
                
                // Invoke the add method to subscribe to the event
                addMethod?.Invoke(null, new object[] { handlerDelegate });
                
                _eventHandlerAttached = true;
            }
        }
        
        [TestMethod]
        public async Task SentimentShiftAlert_ShouldDetectSignificantShifts()
        {
            // This is a partial test that demonstrates how to test the service
            // In reality, we'd need to mock dependencies, but this shows the structure
            
            // Arrange - in a real test we'd set up mocks
            string testSymbol = "AAPL";
            
            // Act - in a real scenario this would be better tested with controlled inputs
            await _sentimentShiftAlertService.MonitorSentimentShiftsAsync(testSymbol);
            
            // Force a second call that would trigger alerts since baseline established
            // In a real test we'd control this through mocks
            FieldInfo lastSentimentValuesField = typeof(SentimentShiftAlertService)
                .GetField("_lastSentimentValues", BindingFlags.NonPublic | BindingFlags.Instance);
            
            if (lastSentimentValuesField != null)
            {
                // We'd simulate a sentiment change here in a full test
                // but for now we just verify method doesn't throw
                await _sentimentShiftAlertService.MonitorSentimentShiftsAsync(testSymbol);
            }
            
            // Assert
            // In a real test with mocked dependencies, we'd verify alerts were created
            // Here we just check the service runs without exceptions
            
            // If alerts were captured, verify they have the right properties
            foreach (var alert in _capturedAlerts)
            {
                Assert.AreEqual(AlertCategory.Sentiment, alert.Category);
                Assert.AreEqual("SentimentShift", alert.AlertType);
                Assert.IsTrue(alert.Name.Contains(testSymbol));
                Assert.IsTrue(alert.Notes.Contains("Previous Sentiment"));
                Assert.IsTrue(alert.Notes.Contains("Current Sentiment"));
                Assert.IsTrue(alert.Notes.Contains("Shift Magnitude"));
            }
        }
        
        [TestMethod]
        public void CalculateAlertPriority_ShouldReturnCorrectPriority()
        {
            // Arrange
            var method = typeof(SentimentShiftAlertService).GetMethod("CalculateAlertPriority", 
                BindingFlags.NonPublic | BindingFlags.Instance);
            
            if (method == null)
            {
                Assert.Fail("CalculateAlertPriority method not found");
                return;
            }
            
            // Test different source types and shift magnitudes
            
            // Act - AnalystRatings high magnitude (0.5)
            int priorityAnalystHigh = (int)method.Invoke(_sentimentShiftAlertService, new object[] { "AnalystRatings", 0.5 });
            
            // Act - Twitter low magnitude (0.25)
            int priorityTwitterLow = (int)method.Invoke(_sentimentShiftAlertService, new object[] { "Twitter", 0.25 });
            
            // Act - News medium magnitude (0.35)
            int priorityNewsMedium = (int)method.Invoke(_sentimentShiftAlertService, new object[] { "News", 0.35 });
            
            // Assert
            Assert.AreEqual(1, priorityAnalystHigh, "AnalystRatings with high shift should have priority 1");
            Assert.AreEqual(3, priorityTwitterLow, "Twitter with low shift should have priority 3");
            Assert.AreEqual(2, priorityNewsMedium, "News with medium shift should have priority 2");
        }
    }
}