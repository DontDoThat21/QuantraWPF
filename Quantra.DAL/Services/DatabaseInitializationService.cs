using System;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using System.Linq;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.DAL.Services.Interfaces;
using Microsoft.Data.SqlClient;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for initializing and ensuring database tables exist using Entity Framework Core
    /// </summary>
    public class DatabaseInitializationService : IDatabaseInitializationService
    {
        private readonly QuantraDbContext _dbContext;
        private readonly LoggingService _loggingService;

        public DatabaseInitializationService(QuantraDbContext dbContext, LoggingService loggingService)
        {
            _dbContext = dbContext;
            _loggingService = loggingService;
        }

        /// <summary>
        /// Ensures all required database tables exist using Entity Framework Core
        /// </summary>
        public void EnsureAllTablesExist()
        {
            ResilienceHelper.Retry(() =>
            {
                // Use EF Core to create all tables based on the DbContext model
                _dbContext.Database.EnsureCreated();

                _loggingService.Log("Info", "Ensured all database tables exist");
            }, RetryOptions.ForCriticalOperation());
        }

        /// <summary>
        /// Ensures UserAppSettings table exists and has required structure
        /// </summary>
        public void EnsureUserAppSettingsTable()
        {
            ResilienceHelper.Retry(() =>
            {
                // EF Core will create the table if it doesn't exist
                _dbContext.Database.EnsureCreated();

                // Verify the table exists by attempting a simple query
                var tableExists = _dbContext.UserAppSettings.Any();

                _loggingService.Log("Info", "UserAppSettings table ensured");
            }, RetryOptions.ForCriticalOperation());
        }

        /// <summary>
        /// Ensures UserCredentials table exists and has required structure
        /// </summary>
        public void EnsureUserCredentialsTable()
        {
            ResilienceHelper.Retry(() =>
            {
                _dbContext.Database.EnsureCreated();

                // Verify the table exists
                var tableExists = _dbContext.UserCredentials.Any();

                _loggingService.Log("Info", "UserCredentials table ensured");
            }, RetryOptions.ForCriticalOperation());
        }

        /// <summary>
        /// Ensures Logs table exists and has required structure
        /// </summary>
        public void EnsureLogsTable()
        {
            ResilienceHelper.Retry(() =>
            {
                _dbContext.Database.EnsureCreated();

                // Verify the table exists
                var tableExists = _dbContext.Logs.Any();

                _loggingService.Log("Info", "Logs table ensured");
            }, RetryOptions.ForCriticalOperation());
        }

        /// <summary>
        /// Ensures TabConfigs table exists and has required structure
        /// </summary>
        public void EnsureTabConfigsTable()
        {
            ResilienceHelper.Retry(() =>
            {
                _dbContext.Database.EnsureCreated();

                // Verify the table exists
                var tableExists = _dbContext.TabConfigs.Any();

                _loggingService.Log("Info", "TabConfigs table ensured");
            }, RetryOptions.ForCriticalOperation());
        }

        /// <summary>
        /// Ensures Settings table exists and has required structure
        /// </summary>
        public void EnsureSettingsTable()
        {
            ResilienceHelper.Retry(() =>
            {
                _dbContext.Database.EnsureCreated();

                // Check if any settings exist, if not create a default one
                if (!_dbContext.Settings.Any())
                {
                    var defaultSettings = new SettingsEntity
                    {
                        EnableApiModalChecks = true,
                        ApiTimeoutSeconds = 30,
                        CacheDurationMinutes = 15,
                        EnableHistoricalDataCache = true,
                        EnableDarkMode = true,
                        ChartUpdateIntervalSeconds = 2,
                        EnablePriceAlerts = true,
                        EnableTradeNotifications = true,
                        EnablePaperTrading = true,
                        RiskLevel = "Low",
                        DefaultGridRows = 4,
                        DefaultGridColumns = 4,
                        GridBorderColor = "#FF00FFFF",
                        AlertEmail = "test@gmail.com"
                    };

                    _dbContext.Settings.Add(defaultSettings);
                    _dbContext.SaveChanges();

                    _loggingService.Log("Info", "Default settings created");
                }

                _loggingService.Log("Info", "Settings table ensured");
            }, RetryOptions.ForCriticalOperation());
        }

        /// <summary>
        /// Fixes the UserPreferences.Value column to allow unlimited text storage (NVARCHAR(MAX))
        /// This fixes the string truncation issue when saving UserSettings
        /// </summary>
        public void FixUserPreferencesValueColumn()
        {
            try
            {
                _loggingService.Log("Info", "Checking UserPreferences.Value column size");

                using (var connection = _dbContext.Database.GetDbConnection())
                {
                    connection.Open();
                    
                    // Check if column needs to be altered
                    var checkSql = @"
                        SELECT c.max_length
                        FROM sys.columns c
                        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
                        WHERE c.object_id = OBJECT_ID('dbo.UserPreferences')
                        AND c.name = 'Value'
                        AND t.name LIKE '%varchar%'";

                    using (var command = connection.CreateCommand())
                    {
                        command.CommandText = checkSql;
                        var maxLength = command.ExecuteScalar();

                        // If max_length is -1, it's already NVARCHAR(MAX)
                        // If it's anything else (like 1000 or 500), we need to alter it
                        if (maxLength != null && Convert.ToInt32(maxLength) != -1)
                        {
                            _loggingService.Log("Info", $"UserPreferences.Value column is VARCHAR({maxLength}), updating to NVARCHAR(MAX)");

                            // Alter the column to NVARCHAR(MAX)
                            var alterSql = "ALTER TABLE dbo.UserPreferences ALTER COLUMN [Value] NVARCHAR(MAX) NULL";
                            
                            using (var alterCommand = connection.CreateCommand())
                            {
                                alterCommand.CommandText = alterSql;
                                alterCommand.ExecuteNonQuery();
                            }

                            _loggingService.Log("Info", "Successfully updated UserPreferences.Value column to NVARCHAR(MAX)");
                        }
                        else if (maxLength == null)
                        {
                            _loggingService.Log("Warning", "Could not determine UserPreferences.Value column size - table may not exist yet");
                        }
                        else
                        {
                            _loggingService.Log("Info", "UserPreferences.Value column is already NVARCHAR(MAX)");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Warning", "Could not alter UserPreferences.Value column", ex.ToString());
                // Don't throw - this is not critical for application startup
            }
        }

        /// <summary>
        /// Applies any pending migrations to the database
        /// </summary>
        public void ApplyMigrations()
        {
            try
            {
                var pendingMigrations = _dbContext.Database.GetPendingMigrations();
                if (pendingMigrations.Any())
                {
                    _loggingService.Log("Info", $"Applying {pendingMigrations.Count()} pending migrations");
                    _dbContext.Database.Migrate();
                    _loggingService.Log("Info", "Database migrations applied successfully");
                }
                else
                {
                    _loggingService.Log("Info", "No pending migrations to apply");
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Warning", "Could not apply migrations, using EnsureCreated instead", ex.ToString());
                // Fall back to EnsureCreated if migrations fail
                _dbContext.Database.EnsureCreated();
            }
            
            // After ensuring database exists, fix the UserPreferences column if needed
            FixUserPreferencesValueColumn();
        }

        /// <summary>
        /// Tests if the database connection is working
        /// </summary>
        public bool TestConnection()
        {
            try
            {
                return _dbContext.Database.CanConnect();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Database connection test failed", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Gets information about the database
        /// </summary>
        public string GetDatabaseInfo()
        {
            try
            {
                var connectionString = _dbContext.Database.GetConnectionString();
                var databaseName = _dbContext.Database.GetDbConnection().Database;
                return $"Database: {databaseName}";
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to get database info", ex.ToString());
                return "Unknown";
            }
        }
    }
}
