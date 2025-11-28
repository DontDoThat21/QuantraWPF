using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.Models;

namespace Quantra.Tests.Models
{
    /// <summary>
    /// Tests for MarketChatMessage cache-related properties (MarketChat story 3)
    /// </summary>
    [TestClass]
    public class MarketChatMessageCacheTests
    {
        [TestMethod]
        public void MarketChatMessage_ShowCacheStatus_ReturnsFalse_WhenFromUser()
        {
            // Arrange
            var message = new MarketChatMessage
            {
                IsFromUser = true,
                CacheStatusDisplay = "Based on prediction from 5 minutes ago",
                IsLoading = false
            };

            // Act & Assert
            Assert.IsFalse(message.ShowCacheStatus);
        }

        [TestMethod]
        public void MarketChatMessage_ShowCacheStatus_ReturnsFalse_WhenLoading()
        {
            // Arrange
            var message = new MarketChatMessage
            {
                IsFromUser = false,
                CacheStatusDisplay = "Based on prediction from 5 minutes ago",
                IsLoading = true
            };

            // Act & Assert
            Assert.IsFalse(message.ShowCacheStatus);
        }

        [TestMethod]
        public void MarketChatMessage_ShowCacheStatus_ReturnsFalse_WhenCacheStatusDisplayIsNull()
        {
            // Arrange
            var message = new MarketChatMessage
            {
                IsFromUser = false,
                CacheStatusDisplay = null,
                IsLoading = false
            };

            // Act & Assert
            Assert.IsFalse(message.ShowCacheStatus);
        }

        [TestMethod]
        public void MarketChatMessage_ShowCacheStatus_ReturnsFalse_WhenCacheStatusDisplayIsEmpty()
        {
            // Arrange
            var message = new MarketChatMessage
            {
                IsFromUser = false,
                CacheStatusDisplay = string.Empty,
                IsLoading = false
            };

            // Act & Assert
            Assert.IsFalse(message.ShowCacheStatus);
        }

        [TestMethod]
        public void MarketChatMessage_ShowCacheStatus_ReturnsTrue_WhenValidAssistantMessageWithCacheStatus()
        {
            // Arrange
            var message = new MarketChatMessage
            {
                IsFromUser = false,
                CacheStatusDisplay = "Based on prediction from 23 minutes ago",
                IsLoading = false
            };

            // Act & Assert
            Assert.IsTrue(message.ShowCacheStatus);
        }

        [TestMethod]
        public void MarketChatMessage_UsesCachedData_DefaultsToFalse()
        {
            // Arrange
            var message = new MarketChatMessage();

            // Act & Assert
            Assert.IsFalse(message.UsesCachedData);
        }

        [TestMethod]
        public void MarketChatMessage_CacheAge_DefaultsToNull()
        {
            // Arrange
            var message = new MarketChatMessage();

            // Act & Assert
            Assert.IsNull(message.CacheAge);
        }

        [TestMethod]
        public void MarketChatMessage_CacheStatusDisplay_DefaultsToNull()
        {
            // Arrange
            var message = new MarketChatMessage();

            // Act & Assert
            Assert.IsNull(message.CacheStatusDisplay);
        }

        [TestMethod]
        public void MarketChatMessage_PropertyChanged_FiresOnCachePropertyChange()
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
            message.UsesCachedData = true;

            // Assert
            Assert.IsTrue(propertyChangedFired);
            Assert.AreEqual("UsesCachedData", changedPropertyName);
        }

        [TestMethod]
        public void MarketChatMessage_PropertyChanged_FiresOnCacheStatusDisplayChange()
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
            message.CacheStatusDisplay = "Based on prediction from 10 minutes ago";

            // Assert
            Assert.IsTrue(propertyChangedFired);
            Assert.AreEqual("CacheStatusDisplay", changedPropertyName);
        }

        [TestMethod]
        public void MarketChatMessage_PropertyChanged_FiresOnCacheAgeChange()
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
            message.CacheAge = TimeSpan.FromMinutes(30);

            // Assert
            Assert.IsTrue(propertyChangedFired);
            Assert.AreEqual("CacheAge", changedPropertyName);
        }

        [TestMethod]
        public void MarketChatMessage_CacheProperties_CanBeSetAndRetrieved()
        {
            // Arrange
            var cacheAge = TimeSpan.FromMinutes(15);
            var cacheStatus = "Based on prediction from 15 minutes ago";

            var message = new MarketChatMessage
            {
                UsesCachedData = true,
                CacheStatusDisplay = cacheStatus,
                CacheAge = cacheAge
            };

            // Assert
            Assert.IsTrue(message.UsesCachedData);
            Assert.AreEqual(cacheStatus, message.CacheStatusDisplay);
            Assert.AreEqual(cacheAge, message.CacheAge);
        }
    }
}
