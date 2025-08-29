using System;
using System.Collections.Generic;
using System.Data.SQLite;
using Quantra.Models;
using Quantra;

namespace Quantra.Repositories
{
    public class IndicatorSettingsRepository
    {
        // Ensure the indicator settings table exists
        public static void EnsureIndicatorSettingsTableExists()
        {
            try
            {
                string createTableQuery = @"
                    CREATE TABLE IF NOT EXISTS IndicatorSettings (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ControlId INTEGER NOT NULL,
                        IndicatorName TEXT NOT NULL,
                        IsEnabled INTEGER NOT NULL,
                        LastUpdated DATETIME NOT NULL,
                        UNIQUE(ControlId, IndicatorName)
                    )";

                // Fix: Use fully qualified namespace to ensure correct method resolution
                Quantra.DatabaseMonolith.ExecuteNonQuery(createTableQuery);
                Quantra.DatabaseMonolith.Log("Info", "IndicatorSettings table created or verified");
            }
            catch (Exception ex)
            {
                Quantra.DatabaseMonolith.Log("Error", "Failed to create IndicatorSettings table", ex.ToString());
                throw;
            }
        }

        // Save or update a single indicator setting
        public static void SaveIndicatorSetting(IndicatorSettingsModel setting)
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();

                    using (var transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            string upsertQuery = @"
                                    INSERT INTO IndicatorSettings (ControlId, IndicatorName, IsEnabled, LastUpdated)
                                    VALUES (@ControlId, @IndicatorName, @IsEnabled, @LastUpdated)
                                    ON CONFLICT(ControlId, IndicatorName) 
                                    DO UPDATE SET IsEnabled = @IsEnabled, LastUpdated = @LastUpdated";

                            using (var command = new SQLiteCommand(upsertQuery, connection))
                            {
                                command.Parameters.AddWithValue("@ControlId", setting.ControlId);
                                command.Parameters.AddWithValue("@IndicatorName", setting.IndicatorName);
                                command.Parameters.AddWithValue("@IsEnabled", setting.IsEnabled ? 1 : 0);
                                command.Parameters.AddWithValue("@LastUpdated", DateTime.Now);
                                command.ExecuteNonQuery();
                            }

                            transaction.Commit();
                        }
                        catch (Exception)
                        {
                            transaction.Rollback();
                            throw;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to save indicator setting for {setting.IndicatorName}", ex.ToString());
                throw;
            }
        }

        // Save multiple indicator settings at once (atomically)
        public static void SaveIndicatorSettings(List<IndicatorSettingsModel> settings)
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();

                    using (var transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            string upsertQuery = @"
                                    INSERT INTO IndicatorSettings (ControlId, IndicatorName, IsEnabled, LastUpdated)
                                    VALUES (@ControlId, @IndicatorName, @IsEnabled, @LastUpdated)
                                    ON CONFLICT(ControlId, IndicatorName) 
                                    DO UPDATE SET IsEnabled = @IsEnabled, LastUpdated = @LastUpdated";

                            using (var command = new SQLiteCommand(upsertQuery, connection))
                            {
                                foreach (var setting in settings)
                                {
                                    command.Parameters.Clear();
                                    command.Parameters.AddWithValue("@ControlId", setting.ControlId);
                                    command.Parameters.AddWithValue("@IndicatorName", setting.IndicatorName);
                                    command.Parameters.AddWithValue("@IsEnabled", setting.IsEnabled ? 1 : 0);
                                    command.Parameters.AddWithValue("@LastUpdated", DateTime.Now);
                                    command.ExecuteNonQuery();
                                }
                            }

                            transaction.Commit();
                        }
                        catch (Exception)
                        {
                            transaction.Rollback();
                            throw;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to save multiple indicator settings", ex.ToString());
                throw;
            }
        }

        // Get all indicator settings for a specific control
        public static List<IndicatorSettingsModel> GetIndicatorSettingsForControl(string controlId)
        {
            List<IndicatorSettingsModel> settings = new List<IndicatorSettingsModel>();

            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();

                    string query = @"
                            SELECT Id, ControlId, IndicatorName, IsEnabled, LastUpdated 
                            FROM IndicatorSettings 
                            WHERE ControlId = @ControlId";

                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@ControlId", controlId);

                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                settings.Add(new IndicatorSettingsModel
                                {
                                    Id = Convert.ToInt32(reader["Id"]),
                                    ControlId = Convert.ToInt32(reader["ControlId"]),
                                    IndicatorName = reader["IndicatorName"].ToString(),
                                    IsEnabled = Convert.ToBoolean(reader["IsEnabled"]),
                                    LastUpdated = Convert.ToDateTime(reader["LastUpdated"])
                                });
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to retrieve indicator settings for control {controlId}", ex.ToString());
                throw;
            }

            return settings;
        }

        public bool Exists(int controlId)
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();

                    string query = @"
                        SELECT COUNT(*) 
                        FROM IndicatorSettings 
                        WHERE ControlId = @ControlId";

                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@ControlId", controlId);
                        int count = Convert.ToInt32(command.ExecuteScalar());
                        return count > 0;
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to check existence of indicator settings for control {controlId}", ex.ToString());
                throw;
            }
        }

        public IndicatorSettingsModel GetByControlId(int controlId)
        {
            try
            {
                var settings = new IndicatorSettingsModel
                {
                    ControlId = controlId
                };

                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();

                    string query = @"
                        SELECT Id, ControlId, IndicatorName, IsEnabled, LastUpdated 
                        FROM IndicatorSettings 
                        WHERE ControlId = @ControlId";

                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@ControlId", controlId);

                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                // Set the first record's ID to the model
                                if (settings.Id == 0)
                                {
                                    settings.Id = Convert.ToInt32(reader["Id"]);
                                }

                                // Map indicators to properties
                                string indicatorName = reader["IndicatorName"].ToString();
                                bool isEnabled = Convert.ToBoolean(reader["IsEnabled"]);

                                switch (indicatorName)
                                {
                                    case "VWAP":
                                        settings.UseVwap = isEnabled;
                                        break;
                                    case "MACD":
                                        settings.UseMacd = isEnabled;
                                        break;
                                    case "RSI":
                                        settings.UseRsi = isEnabled;
                                        break;
                                    case "Bollinger":
                                        settings.UseBollinger = isEnabled;
                                        break;
                                    case "MA":
                                        settings.UseMa = isEnabled;
                                        break;
                                    case "Volume":
                                        settings.UseVolume = isEnabled;
                                        break;
                                    case "BreadthThrust":
                                        settings.UseBreadthThrust = isEnabled;
                                        break;
                                }
                            }

                            // Check if any records were found
                            if (settings.Id == 0)
                            {
                                return null; // No settings found for this control ID
                            }
                        }
                    }
                }

                return settings;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to retrieve indicator settings for control {controlId}", ex.ToString());
                throw;
            }
        }

        public void Save(IndicatorSettingsModel settings)
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();

                    using (var transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            // Save each indicator setting individually
                            SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "VWAP", settings.UseVwap));
                            SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "MACD", settings.UseMacd));
                            SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "RSI", settings.UseRsi));
                            SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "Bollinger", settings.UseBollinger));
                            SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "MA", settings.UseMa));
                            SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "Volume", settings.UseVolume));
                            SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "BreadthThrust", settings.UseBreadthThrust));
                            
                            transaction.Commit();
                        }
                        catch (Exception ex)
                        {
                            transaction.Rollback();
                            throw;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to save indicator settings for control {settings.ControlId}", ex.ToString());
                throw;
            }
        }

        public void Update(IndicatorSettingsModel settings)
        {
            try
            {
                // Since the SaveIndicatorSetting uses UPSERT logic, we can reuse the same Save method
                Save(settings);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to update indicator settings for control {settings.ControlId}", ex.ToString());
                throw;
            }
        }
    }
}
