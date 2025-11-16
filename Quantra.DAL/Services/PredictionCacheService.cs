using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Quantra.Models;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for caching ML prediction results to minimize repeated model execution
    /// </summary>
    public class PredictionCacheService
    {
        private readonly TimeSpan _cacheValidityPeriod;
        private readonly LoggingService _loggingService;

        public PredictionCacheService(TimeSpan? cacheValidityPeriod = null, LoggingService loggingService)
        {
            _cacheValidityPeriod = cacheValidityPeriod ?? TimeSpan.FromHours(1); // Default 1 hour cache
            _loggingService = loggingService;
            EnsureCacheTableExists();
        }

        /// <summary>
        /// Ensures the prediction cache table exists using Entity Framework Core
        /// </summary>
        private void EnsureCacheTableExists()
        {
            try
            {
                // Create DbContext using default configuration
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlite("Data Source=Quantra.db;Journal Mode=WAL;Busy Timeout=30000;");

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Ensure the database and tables are created
                    // This will create all tables defined in the DbContext including PredictionCache
                    dbContext.Database.EnsureCreated();

                    _loggingService.Log("Info", "Prediction cache table created or verified using EF Core");
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to ensure prediction cache table exists", ex.ToString());
            }
        }

        /// <summary>
        /// Get cached prediction result if available and valid using Entity Framework
        /// </summary>
        public PredictionResult GetCachedPrediction(string symbol, string modelVersion, string inputDataHash)
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlite("Data Source=Quantra.db;Journal Mode=WAL;Busy Timeout=30000;");

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Query using LINQ - EF Core will translate to appropriate SQL (works with both SQLite and SQL Server)
                    var cachedEntry = dbContext.PredictionCache
                        .Where(pc => pc.Symbol == symbol
                                  && pc.ModelVersion == modelVersion
                                  && pc.InputDataHash == inputDataHash)
                        .OrderByDescending(pc => pc.CreatedAt)
                        .FirstOrDefault();

                    if (cachedEntry != null)
                    {
                        // Check if cache is still valid
                        if (DateTime.Now - cachedEntry.CreatedAt <= _cacheValidityPeriod)
                        {
                            return new PredictionResult
                            {
                                Symbol = symbol,
                                CurrentPrice = 0, // Will be set by caller
                                TargetPrice = cachedEntry.PredictedPrice ?? 0,
                                Action = cachedEntry.PredictedAction ?? string.Empty,
                                Confidence = cachedEntry.Confidence ?? 0,
                                PredictionDate = cachedEntry.PredictionTimestamp ?? DateTime.Now,
                                ModelType = "cached"
                            };
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error retrieving cached prediction for {symbol}", ex.ToString());
            }

            return null;
        }

        /// <summary>
        /// Cache a prediction result using Entity Framework
        /// </summary>
        public void CachePrediction(string symbol, string modelVersion, string inputDataHash, PredictionResult prediction)
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlite("Data Source=Quantra.db;Journal Mode=WAL;Busy Timeout=30000;");

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Check if entry already exists
                    var existingEntry = dbContext.PredictionCache
                        .FirstOrDefault(pc => pc.Symbol == symbol
                                           && pc.ModelVersion == modelVersion
                                           && pc.InputDataHash == inputDataHash);

                    if (existingEntry != null)
                    {
                        // Update existing entry
                        existingEntry.PredictedPrice = prediction.TargetPrice;
                        existingEntry.PredictedAction = prediction.Action;
                        existingEntry.Confidence = prediction.Confidence;
                        existingEntry.PredictionTimestamp = prediction.PredictionDate;
                        existingEntry.CreatedAt = DateTime.Now;
                    }
                    else
                    {
                        // Insert new entry
                        var newEntry = new Data.Entities.PredictionCacheEntity
                        {
                            Symbol = symbol,
                            ModelVersion = modelVersion,
                            InputDataHash = inputDataHash,
                            PredictedPrice = prediction.TargetPrice,
                            PredictedAction = prediction.Action,
                            Confidence = prediction.Confidence,
                            PredictionTimestamp = prediction.PredictionDate,
                            CreatedAt = DateTime.Now
                        };
                        dbContext.PredictionCache.Add(newEntry);
                    }

                    dbContext.SaveChanges();
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error caching prediction for {symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Generate a hash of input data for cache key
        /// </summary>
        public string GenerateInputDataHash(Dictionary<string, double> inputData)
        {
            if (inputData == null || inputData.Count == 0)
                return string.Empty;

            // Normalize: sort keys and format doubles in invariant culture with fixed precision
            var sb = new StringBuilder();
            foreach (var kvp in inputData.OrderBy(k => k.Key, StringComparer.Ordinal))
            {
                sb.Append(kvp.Key);
                sb.Append(':');
                sb.Append(kvp.Value.ToString("F6", CultureInfo.InvariantCulture));
                sb.Append('|');
            }
            var normalized = sb.ToString();

            // Stable cryptographic hash (SHA256) to avoid platform-specific GetHashCode differences
            using var sha = SHA256.Create();
            var bytes = Encoding.UTF8.GetBytes(normalized);
            var hashBytes = sha.ComputeHash(bytes);
            var hash = Convert.ToHexString(hashBytes); // .NET 5+ uppercase hex
            return hash;
        }

        /// <summary>
        /// Clear old cache entries using Entity Framework
        /// </summary>
        public void ClearExpiredCache()
        {
            try
            {
                var expiryDate = DateTime.Now - _cacheValidityPeriod;
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlite("Data Source=Quantra.db;Journal Mode=WAL;Busy Timeout=30000;");

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Find expired entries using LINQ
                    var expiredEntries = dbContext.PredictionCache
                        .Where(pc => pc.CreatedAt < expiryDate)
                        .ToList();

                    if (expiredEntries.Any())
                    {
                        dbContext.PredictionCache.RemoveRange(expiredEntries);
                        dbContext.SaveChanges();
                        _loggingService.Log("Info", $"Cleared {expiredEntries.Count} expired cache entries");
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error clearing expired cache", ex.ToString());
            }
        }
    }
}