using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Tests
{
    [TestClass]
    public class ServiceLocatorTests
    {
        [TestInitialize]
        public void Initialize()
        {
            // Clear any existing registrations to ensure clean test state
            // Note: This would require adding a Clear method to ServiceLocator in a real scenario
        }

        [TestMethod]
        public void Resolve_WithUnknownServiceName_ThrowsException()
        {
            // Act & Assert
            Assert.ThrowsException<InvalidOperationException>(() =>
                ServiceLocator.Resolve<ISocialMediaSentimentService>("UnknownService"));
        }

        [TestMethod]
        public void Resolve_WithWrongType_ThrowsException()
        {
            // Arrange
            ServiceLocator.RegisterService("TestService", "StringValue");

            // Act & Assert
            Assert.ThrowsException<InvalidOperationException>(() =>
                ServiceLocator.Resolve<ISocialMediaSentimentService>("TestService"));
        }
    }
}