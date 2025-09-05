using System;
using NUnit.Framework;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Tests
{
    [TestFixture]
    public class OrderServiceTests
    {
        private OrderService _orderService;

        [SetUp]
        public void Setup()
        {
            _orderService = new OrderService();
        }

        [Test]
        public void SetupDollarCostAveraging_ValidParameters_ReturnsTrue()
        {
            // Arrange
            string symbol = "AAPL";
            int totalShares = 100;
            int numberOfOrders = 4;
            int intervalDays = 7;

            // Act
            bool result = _orderService.SetupDollarCostAveraging(symbol, totalShares, numberOfOrders, intervalDays);

            // Assert
            Assert.That(result, Is.True, "SetupDollarCostAveraging should return true for valid parameters");
        }

        [Test]
        public void SetupDollarCostAveraging_InvalidSymbol_ReturnsFalse()
        {
            // Arrange
            string symbol = "";
            int totalShares = 100;
            int numberOfOrders = 4;
            int intervalDays = 7;

            // Act
            bool result = _orderService.SetupDollarCostAveraging(symbol, totalShares, numberOfOrders, intervalDays);

            // Assert
            Assert.That(result, Is.False, "SetupDollarCostAveraging should return false for invalid symbol");
        }

        [Test]
        public void SetupDollarCostAveraging_InvalidShares_ReturnsFalse()
        {
            // Arrange
            string symbol = "AAPL";
            int totalShares = 0;
            int numberOfOrders = 4;
            int intervalDays = 7;

            // Act
            bool result = _orderService.SetupDollarCostAveraging(symbol, totalShares, numberOfOrders, intervalDays);

            // Assert
            Assert.That(result, Is.False, "SetupDollarCostAveraging should return false for zero shares");
        }
    }
}