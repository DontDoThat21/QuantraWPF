using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Models;
using Quantra.Models;

namespace Quantra.Tests.Models
{
    /// <summary>
    /// Tests for MarketChatMessage chart-related properties (MarketChat story 8)
    /// </summary>
    [TestClass]
    public class MarketChatMessageChartTests
    {
        #region HasChartData Tests

        [TestMethod]
        public void MarketChatMessage_HasChartData_ReturnsFalse_WhenChartDataIsNull()
        {
            // Arrange
            var message = new MarketChatMessage
            {
                ChartData = null
            };

            // Act & Assert
            Assert.IsFalse(message.HasChartData);
        }

        [TestMethod]
        public void MarketChatMessage_HasChartData_ReturnsFalse_WhenChartDataHasEmptyHistoricalPrices()
        {
            // Arrange
            var chartData = new ProjectionChartData
            {
                Symbol = "AAPL",
                HistoricalPrices = new List<double>()
            };
            var message = new MarketChatMessage
            {
                ChartData = chartData
            };

            // Act & Assert
            Assert.IsFalse(message.HasChartData);
        }

        [TestMethod]
        public void MarketChatMessage_HasChartData_ReturnsTrue_WhenChartDataIsValid()
        {
            // Arrange
            var chartData = new ProjectionChartData
            {
                Symbol = "AAPL",
                HistoricalPrices = new List<double> { 100.0, 101.0, 102.0 }
            };
            var message = new MarketChatMessage
            {
                ChartData = chartData
            };

            // Act & Assert
            Assert.IsTrue(message.HasChartData);
        }

        #endregion

        #region ShowChart Tests

        [TestMethod]
        public void MarketChatMessage_ShowChart_ReturnsFalse_WhenFromUser()
        {
            // Arrange
            var chartData = new ProjectionChartData
            {
                Symbol = "AAPL",
                HistoricalPrices = new List<double> { 100.0, 101.0 }
            };
            var message = new MarketChatMessage
            {
                IsFromUser = true,
                ChartData = chartData,
                IsLoading = false
            };

            // Act & Assert
            Assert.IsFalse(message.ShowChart);
        }

        [TestMethod]
        public void MarketChatMessage_ShowChart_ReturnsFalse_WhenLoading()
        {
            // Arrange
            var chartData = new ProjectionChartData
            {
                Symbol = "AAPL",
                HistoricalPrices = new List<double> { 100.0, 101.0 }
            };
            var message = new MarketChatMessage
            {
                IsFromUser = false,
                ChartData = chartData,
                IsLoading = true
            };

            // Act & Assert
            Assert.IsFalse(message.ShowChart);
        }

        [TestMethod]
        public void MarketChatMessage_ShowChart_ReturnsFalse_WhenNoChartData()
        {
            // Arrange
            var message = new MarketChatMessage
            {
                IsFromUser = false,
                ChartData = null,
                IsLoading = false
            };

            // Act & Assert
            Assert.IsFalse(message.ShowChart);
        }

        [TestMethod]
        public void MarketChatMessage_ShowChart_ReturnsTrue_WhenValidAssistantMessageWithChartData()
        {
            // Arrange
            var chartData = new ProjectionChartData
            {
                Symbol = "AAPL",
                HistoricalPrices = new List<double> { 100.0, 101.0, 102.0 }
            };
            var message = new MarketChatMessage
            {
                IsFromUser = false,
                ChartData = chartData,
                IsLoading = false
            };

            // Act & Assert
            Assert.IsTrue(message.ShowChart);
        }

        #endregion

        #region ChartStatusDisplay Tests

        [TestMethod]
        public void MarketChatMessage_ChartStatusDisplay_ReturnsNull_WhenNoChartData()
        {
            // Arrange
            var message = new MarketChatMessage
            {
                ChartData = null
            };

            // Act & Assert
            Assert.IsNull(message.ChartStatusDisplay);
        }

        [TestMethod]
        public void MarketChatMessage_ChartStatusDisplay_ReturnsFormattedString_WhenHasValidChartData()
        {
            // Arrange
            var chartData = new ProjectionChartData
            {
                Symbol = "AAPL",
                PredictedAction = "BUY",
                Confidence = 0.85,
                HistoricalPrices = new List<double> { 100.0, 101.0 }
            };
            var message = new MarketChatMessage
            {
                ChartData = chartData
            };

            // Act
            var status = message.ChartStatusDisplay;

            // Assert
            Assert.IsNotNull(status);
            Assert.IsTrue(status.Contains("AAPL"));
            Assert.IsTrue(status.Contains("BUY"));
        }

        #endregion

        #region ChartData PropertyChanged Tests

        [TestMethod]
        public void MarketChatMessage_PropertyChanged_FiresOnChartDataChange()
        {
            // Arrange
            var message = new MarketChatMessage();
            bool propertyChangedFired = false;
            string changedPropertyName = null;

            message.PropertyChanged += (sender, args) =>
            {
                propertyChangedFired = true;
                changedPropertyName = args.PropertyName;
            };

            // Act
            message.ChartData = new ProjectionChartData
            {
                Symbol = "AAPL",
                HistoricalPrices = new List<double> { 100.0 }
            };

            // Assert
            Assert.IsTrue(propertyChangedFired);
            Assert.AreEqual("ChartData", changedPropertyName);
        }

        [TestMethod]
        public void MarketChatMessage_PropertyChanged_NotifiesHasChartData_WhenChartDataChanges()
        {
            // Arrange
            var message = new MarketChatMessage();
            var notifiedProperties = new List<string>();

            message.PropertyChanged += (sender, args) =>
            {
                notifiedProperties.Add(args.PropertyName);
            };

            // Act
            message.ChartData = new ProjectionChartData
            {
                Symbol = "AAPL",
                HistoricalPrices = new List<double> { 100.0 }
            };

            // Assert
            Assert.IsTrue(notifiedProperties.Contains("ChartData"));
            Assert.IsTrue(notifiedProperties.Contains("HasChartData"));
        }

        #endregion

        #region ChartData Defaults Tests

        [TestMethod]
        public void MarketChatMessage_ChartData_DefaultsToNull()
        {
            // Arrange
            var message = new MarketChatMessage();

            // Act & Assert
            Assert.IsNull(message.ChartData);
        }

        [TestMethod]
        public void MarketChatMessage_HasChartData_DefaultsToFalse()
        {
            // Arrange
            var message = new MarketChatMessage();

            // Act & Assert
            Assert.IsFalse(message.HasChartData);
        }

        #endregion

        #region MessageType Tests

        [TestMethod]
        public void MarketChatMessage_MessageType_CanBeSetToChartMessage()
        {
            // Arrange
            var message = new MarketChatMessage
            {
                MessageType = MessageType.ChartMessage
            };

            // Act & Assert
            Assert.AreEqual(MessageType.ChartMessage, message.MessageType);
        }

        #endregion
    }
}
