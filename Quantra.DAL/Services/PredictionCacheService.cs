using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for caching ML prediction results to minimize repeated model execution
    /// </summary>
    public class PredictionCacheService
    {
        private readonly string _databasePath;
        private readonly TimeSpan _cacheValidityPeriod;

        public PredictionCacheService(TimeSpan? cacheValidityPeriod = null)
        {
            _databasePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Quantra", "prediction_cache.db");
            _cacheValidityPeriod = cacheValidityPeriod ?? TimeSpan.FromHours(1); // Default 1 hour cache
            EnsureCacheTableExists();
        }

        private void EnsureCacheTableExists()
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(_databasePath));

                using (var connection = new SQLiteConnection($"Data Source={_databasePath};Version=3;"))
                {
                    connection.Open();
                    var command = new SQLiteCommand(@"
                        CREATE TABLE IF NOT EXISTS PredictionCache (
                            Id INTEGER PRIMARY KEY AUTOINCREMENT,
                            Symbol TEXT NOT NULL,
                            ModelVersion TEXT NOT NULL,
                            InputDataHash TEXT NOT NULL,
                            PredictedPrice REAL,
                            PredictedAction TEXT,
                            Confidence REAL,
                            PredictionTimestamp TEXT,
                            CreatedAt TEXT,
                            UNIQUE(Symbol, ModelVersion, InputDataHash)
                        )", connection);
                    command.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error creating prediction cache table", ex.ToString());
            }
        }

        /// <summary>
        /// Get cached prediction result if available and valid
        /// </summary>
        public PredictionResult GetCachedPrediction(string symbol, string modelVersion, string inputDataHash)
        {
            try
            {
                using (var connection = new SQLiteConnection($"Data Source={_databasePath};Version=3;"))
                {
                    connection.Open();
                    var command = new SQLiteCommand(@"
                        SELECT PredictedPrice, PredictedAction, Confidence, PredictionTimestamp, CreatedAt
                        FROM PredictionCache 
                        WHERE Symbol = @Symbol AND ModelVersion = @ModelVersion AND InputDataHash = @InputDataHash
                        ORDER BY CreatedAt DESC
                        LIMIT 1", connection);
                    
                    command.Parameters.AddWithValue("@Symbol", symbol);
                    command.Parameters.AddWithValue("@ModelVersion", modelVersion);
                    command.Parameters.AddWithValue("@InputDataHash", inputDataHash);

                    using (var reader = command.ExecuteReader())
                    {
                        if (reader.Read())
                        {
                            var createdAtStr = reader["CreatedAt"].ToString();
                            var predictionTsStr = reader["PredictionTimestamp"].ToString();

                            // Parse with fixed format, fallback to general parse
                            var format = "yyyy-MM-dd HH:mm:ss";
                            var createdAt = DateTime.TryParseExact(createdAtStr, format, CultureInfo.InvariantCulture, DateTimeStyles.None, out var ca)
                                ? ca
                                : DateTime.Parse(createdAtStr, CultureInfo.InvariantCulture);
                            var predictionTs = DateTime.TryParseExact(predictionTsStr, format, CultureInfo.InvariantCulture, DateTimeStyles.None, out var pt)
                                ? pt
                                : DateTime.Parse(predictionTsStr, CultureInfo.InvariantCulture);
                            
                            // Check if cache is still valid
                            if (DateTime.Now - createdAt <= _cacheValidityPeriod)
                            {
                                return new PredictionResult
                                {
                                    Symbol = symbol,
                                    CurrentPrice = 0, // Will be set by caller
                                    TargetPrice = Convert.ToDouble(reader["PredictedPrice"], CultureInfo.InvariantCulture),
                                    Action = reader["PredictedAction"].ToString(),
                                    Confidence = Convert.ToDouble(reader["Confidence"], CultureInfo.InvariantCulture),
                                    PredictionDate = predictionTs,
                                    ModelType = "cached"
                                };
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error retrieving cached prediction for {symbol}", ex.ToString());
            }

            return null;
        }

        /// <summary>
        /// Cache a prediction result
        /// </summary>
        public void CachePrediction(string symbol, string modelVersion, string inputDataHash, PredictionResult prediction)
        {
            try
            {
                using (var connection = new SQLiteConnection($"Data Source={_databasePath};Version=3;"))
                {
                    connection.Open();
                    var command = new SQLiteCommand(@"
                        INSERT OR REPLACE INTO PredictionCache 
                        (Symbol, ModelVersion, InputDataHash, PredictedPrice, PredictedAction, Confidence, PredictionTimestamp, CreatedAt)
                        VALUES (@Symbol, @ModelVersion, @InputDataHash, @PredictedPrice, @PredictedAction, @Confidence, @PredictionTimestamp, @CreatedAt)", connection);

                    command.Parameters.AddWithValue("@Symbol", symbol);
                    command.Parameters.AddWithValue("@ModelVersion", modelVersion);
                    command.Parameters.AddWithValue("@InputDataHash", inputDataHash);
                    command.Parameters.AddWithValue("@PredictedPrice", prediction.TargetPrice);
                    command.Parameters.AddWithValue("@PredictedAction", prediction.Action);
                    command.Parameters.AddWithValue("@Confidence", prediction.Confidence);
                    command.Parameters.AddWithValue("@PredictionTimestamp", prediction.PredictionDate.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
                    command.Parameters.AddWithValue("@CreatedAt", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));

                    command.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error caching prediction for {symbol}", ex.ToString());
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
        /// Clear old cache entries
        /// </summary>
        public void ClearExpiredCache()
        {
            try
            {
                var expiryDate = DateTime.Now - _cacheValidityPeriod;
                using (var connection = new SQLiteConnection($"Data Source={_databasePath};Version=3;"))
                {
                    connection.Open();
                    var command = new SQLiteCommand(@"
                        DELETE FROM PredictionCache 
                        WHERE datetime(CreatedAt) < datetime(@ExpiryDate)", connection);
                    
                    command.Parameters.AddWithValue("@ExpiryDate", expiryDate.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
                    command.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error clearing expired cache", ex.ToString());
            }
        }
    }
}