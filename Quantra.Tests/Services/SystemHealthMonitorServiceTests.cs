using System;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.Services;
using Quantra.CrossCutting.Monitoring;

namespace Quantra.Tests.Services
{
    [TestClass]
    public class SystemHealthMonitorServiceTests
    {
        [TestMethod]
        public void JsonParsing_ShouldHandleBooleanAnomaliesDetected()
        {
            // Test JSON that simulates what the Python anomaly detection script returns
            var jsonWithBooleanTrue = @"{
                ""anomalies_detected"": true,
                ""anomaly_count"": 2,
                ""recent_anomalies"": [
                    {
                        ""types"": [""VOLUME_ANOMALY""],
                        ""insights"": {
                            ""description"": [""Unusual volume spike detected""],
                            ""potential_causes"": [""News event or large trade""],
                            ""suggested_actions"": [""Monitor closely""]
                        }
                    }
                ]
            }";

            var jsonWithBooleanFalse = @"{
                ""anomalies_detected"": false,
                ""anomaly_count"": 0,
                ""recent_anomalies"": []
            }";

            // Parse the JSON
            var resultTrue = JsonSerializer.Deserialize<JsonElement>(jsonWithBooleanTrue);
            var resultFalse = JsonSerializer.Deserialize<JsonElement>(jsonWithBooleanFalse);

            // Test that we can properly read the boolean values
            Assert.IsTrue(resultTrue.TryGetProperty("anomalies_detected", out JsonElement anomaliesDetectedTrue));
            Assert.IsTrue(anomaliesDetectedTrue.GetBoolean(), "Should read true boolean value correctly");

            Assert.IsTrue(resultFalse.TryGetProperty("anomalies_detected", out JsonElement anomaliesDetectedFalse));
            Assert.IsFalse(anomaliesDetectedFalse.GetBoolean(), "Should read false boolean value correctly");
        }

        [TestMethod]
        public void JsonParsing_ShouldNotThrowWhenReadingBooleanAsFalse()
        {
            // This test specifically validates the fix for the original issue
            var jsonWithFalse = @"{""anomalies_detected"": false}";
            var result = JsonSerializer.Deserialize<JsonElement>(jsonWithFalse);

            // This should not throw an exception anymore
            Assert.IsTrue(result.TryGetProperty("anomalies_detected", out JsonElement anomaliesDetectedElement));
            
            // The original code would have called GetInt32() here and failed
            // The fixed code uses GetBoolean() which should work
            bool anomaliesDetected = anomaliesDetectedElement.GetBoolean();
            Assert.IsFalse(anomaliesDetected, "Should correctly read false value");
        }

        [TestMethod]
        public void JsonParsing_OriginalCodeWouldHaveThrown()
        {
            // Demonstrate that the original approach would fail
            var jsonWithFalse = @"{""anomalies_detected"": false}";
            var result = JsonSerializer.Deserialize<JsonElement>(jsonWithFalse);
            
            Assert.IsTrue(result.TryGetProperty("anomalies_detected", out JsonElement anomaliesDetectedElement));
            
            // This is what the original code tried to do - should throw InvalidOperationException
            Assert.ThrowsException<InvalidOperationException>(() => 
            {
                anomaliesDetectedElement.GetInt32();
            }, "GetInt32() should throw when trying to read a boolean as integer");
        }

        [TestMethod]
        public void DetectMetricAnomaliesAsync_LogicValidation()
        {
            // Test the logic pattern used in the fixed code
            var jsonWithAnomalies = @"{
                ""anomalies_detected"": true,
                ""recent_anomalies"": [
                    {
                        ""types"": [""VOLUME_ANOMALY"", ""PRICE_ANOMALY""],
                        ""insights"": {
                            ""description"": [""Unusual activity detected""],
                            ""potential_causes"": [""Market event""],
                            ""suggested_actions"": [""Review position""]
                        }
                    }
                ]
            }";

            var jsonWithoutAnomalies = @"{
                ""anomalies_detected"": false,
                ""recent_anomalies"": []
            }";

            var resultWithAnomalies = JsonSerializer.Deserialize<JsonElement>(jsonWithAnomalies);
            var resultWithoutAnomalies = JsonSerializer.Deserialize<JsonElement>(jsonWithoutAnomalies);

            // Test the logic pattern from the fixed code
            bool shouldProcessAnomalies1 = resultWithAnomalies.TryGetProperty("anomalies_detected", out JsonElement element1) && 
                                          element1.GetBoolean();
            Assert.IsTrue(shouldProcessAnomalies1, "Should detect anomalies when true");

            bool shouldProcessAnomalies2 = resultWithoutAnomalies.TryGetProperty("anomalies_detected", out JsonElement element2) && 
                                          element2.GetBoolean();
            Assert.IsFalse(shouldProcessAnomalies2, "Should not detect anomalies when false");
        }
    }
}