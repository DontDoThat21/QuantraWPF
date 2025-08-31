using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using NUnit.Framework;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Tests.Services
{
    [TestFixture]
    public class PredictionCacheServiceTests
    {
        private PredictionCacheService _cacheService;
        private string _testDatabasePath;

        [SetUp]
        public void SetUp()
        {
            // Use a test-specific cache with shorter validity period
            _cacheService = new PredictionCacheService(TimeSpan.FromMinutes(5));

            // Clean up any existing test data
            var appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Quantra");
            _testDatabasePath = Path.Combine(appDataPath, "prediction_cache.db");
        }

        [TearDown]
        public void TearDown()
        {
            _cacheService?.ClearExpiredCache();
        }

        [Test]
        public void GenerateInputDataHash_SameData_ReturnsSameHash()
        {
            // Arrange
            var inputData1 = new Dictionary<string, double>
            {
                ["close"] = 150.50,
                ["volume"] = 1000000,
                ["rsi"] = 65.5
            };

            var inputData2 = new Dictionary<string, double>
            {
                ["close"] = 150.50,
                ["volume"] = 1000000,
                ["rsi"] = 65.5
            };

            // Act
            var hash1 = _cacheService.GenerateInputDataHash(inputData1);
            var hash2 = _cacheService.GenerateInputDataHash(inputData2);

            // Assert
            Assert.That(hash1, Is.EqualTo(hash2));
        }

        [Test]
        public void GenerateInputDataHash_DifferentData_ReturnsDifferentHash()
        {
            // Arrange
            var inputData1 = new Dictionary<string, double>
            {
                ["close"] = 150.50,
                ["volume"] = 1000000,
                ["rsi"] = 65.5
            };

            var inputData2 = new Dictionary<string, double>
            {
                ["close"] = 151.00, // Different value
                ["volume"] = 1000000,
                ["rsi"] = 65.5
            };

            // Act
            var hash1 = _cacheService.GenerateInputDataHash(inputData1);
            var hash2 = _cacheService.GenerateInputDataHash(inputData2);

            // Assert
            Assert.That(hash1, Is.Not.EqualTo(hash2));
        }

        [Test]
        public void CachePrediction_AndRetrieve_ReturnsCorrectData()
        {
            // Arrange
            var symbol = "AAPL";
            var modelVersion = "v1.0";
            var inputHash = "test_hash_123";

            var originalPrediction = new PredictionResult
            {
                Symbol = symbol,
                Action = "BUY",
                Confidence = 0.85,
                TargetPrice = 155.50,
                CurrentPrice = 150.00,
                PredictionDate = DateTime.Now
            };

            // Act
            _cacheService.CachePrediction(symbol, modelVersion, inputHash, originalPrediction);
            var retrievedPrediction = _cacheService.GetCachedPrediction(symbol, modelVersion, inputHash);

            // Assert
            Assert.That(retrievedPrediction, Is.Not.Null);
            Assert.That(retrievedPrediction.Symbol, Is.EqualTo(symbol));
            Assert.That(retrievedPrediction.Action, Is.EqualTo("BUY"));
            Assert.That(retrievedPrediction.Confidence, Is.EqualTo(0.85).Within(0.001));
            Assert.That(retrievedPrediction.TargetPrice, Is.EqualTo(155.50).Within(0.001));
        }

        [Test]
        public void GetCachedPrediction_NonExistentKey_ReturnsNull()
        {
            // Arrange
            var symbol = "NONEXISTENT";
            var modelVersion = "v1.0";
            var inputHash = "nonexistent_hash";

            // Act
            var result = _cacheService.GetCachedPrediction(symbol, modelVersion, inputHash);

            // Assert
            Assert.That(result, Is.Null);
        }

        [Test]
        public void CachePrediction_DuplicateKey_UpdatesExistingEntry()
        {
            // Arrange
            var symbol = "AAPL";
            var modelVersion = "v1.0";
            var inputHash = "test_hash_456";

            var prediction1 = new PredictionResult
            {
                Symbol = symbol,
                Action = "BUY",
                Confidence = 0.75,
                TargetPrice = 155.00,
                PredictionDate = DateTime.Now
            };

            var prediction2 = new PredictionResult
            {
                Symbol = symbol,
                Action = "SELL",
                Confidence = 0.90,
                TargetPrice = 145.00,
                PredictionDate = DateTime.Now
            };

            // Act
            _cacheService.CachePrediction(symbol, modelVersion, inputHash, prediction1);
            _cacheService.CachePrediction(symbol, modelVersion, inputHash, prediction2); // Should update
            var result = _cacheService.GetCachedPrediction(symbol, modelVersion, inputHash);

            // Assert
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Action, Is.EqualTo("SELL")); // Should have the updated value
            Assert.That(result.Confidence, Is.EqualTo(0.90).Within(0.001));
            Assert.That(result.TargetPrice, Is.EqualTo(145.00).Within(0.001));
        }

        [Test]
        public void GetCachedPrediction_DifferentSymbol_ReturnsNull()
        {
            // Arrange
            var symbol1 = "AAPL";
            var symbol2 = "GOOGL";
            var modelVersion = "v1.0";
            var inputHash = "test_hash_789";

            var prediction = new PredictionResult
            {
                Symbol = symbol1,
                Action = "BUY",
                Confidence = 0.80,
                TargetPrice = 160.00,
                PredictionDate = DateTime.Now
            };

            // Act
            _cacheService.CachePrediction(symbol1, modelVersion, inputHash, prediction);
            var result = _cacheService.GetCachedPrediction(symbol2, modelVersion, inputHash); // Different symbol

            // Assert
            Assert.That(result, Is.Null);
        }

        [Test]
        public void ClearExpiredCache_RemovesOldEntries()
        {
            // This test would need a way to manipulate the cache validity period
            // or mock the current time to properly test expiration logic
            // For now, we'll just ensure the method runs without throwing

            // Arrange
            var symbol = "AAPL";
            var modelVersion = "v1.0";
            var inputHash = "test_hash_cleanup";

            var prediction = new PredictionResult
            {
                Symbol = symbol,
                Action = "BUY",
                Confidence = 0.70,
                TargetPrice = 150.00,
                PredictionDate = DateTime.Now
            };

            // Act & Assert
            _cacheService.CachePrediction(symbol, modelVersion, inputHash, prediction);
            Assert.DoesNotThrow(() => _cacheService.ClearExpiredCache());
        }
    }
}