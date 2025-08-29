using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.IO;
using System.Windows;
using Dapper;
using Quantra.Models;
using Quantra.Services;
using System.Data;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Net.Http;
using Newtonsoft.Json;
using Microsoft.Extensions.Configuration;
using System.Reflection; // <-- Add this for reflection
using Quantra.CrossCutting.ErrorHandling;

namespace Quantra
{
    /// <summary>
    /// Monolithic database access layer for the Quantra algorithmic trading platform.
    /// 
    /// This class serves as the central data access point for all database operations in the application,
    /// providing a unified interface for SQLite database interactions. It handles everything from basic
    /// CRUD operations to complex trading data management, logging, settings persistence, and analytics.
    /// 
    /// The monolithic design was chosen to simplify dependency management and ensure consistent
    /// database connection handling across the entire application.
    /// </summary>
    /// <remarks>
    /// Key responsibilities:
    /// - Database initialization and schema management
    /// - Logging and error tracking
    /// - User settings and preferences storage
    /// - Trading data persistence (orders, history, rules)
    /// - Market data caching (quotes, stock data, analyst ratings)
    /// - API usage tracking and rate limiting
    /// - Configuration and layout persistence
    /// 
    /// This class uses SQLite as the underlying database with WAL (Write-Ahead Logging) mode
    /// for better concurrency and reduced locking issues.
    /// 
    /// Thread Safety: This class is designed to be thread-safe through proper connection management
    /// and transaction handling. Each method creates its own connection or uses appropriate locking.
    /// </remarks>
    /// <example>
    /// <code>
    /// // Basic logging
    /// DatabaseMonolith.Log("Info", "Application started", "Version 1.0");
    /// 
    /// // Error logging with context
    /// try {
    ///     // Some operation
    /// } catch (Exception ex) {
    ///     DatabaseMonolith.LogErrorWithContext(ex, "Operation failed");
    /// }
    /// 
    /// // Save user settings
    /// var settings = new UserSettings { EnableDarkMode = true };
    /// DatabaseMonolith.SaveUserSettings(settings);
    /// 
    /// // Trading operations
    /// var order = new OrderModel { Symbol = "AAPL", OrderType = "BUY", Quantity = 100 };
    /// DatabaseMonolith.AddOrderToHistory(order);
    /// </code>
    /// </example>
    public static class DatabaseMonolith
    {
        private static readonly string DbFilePath = "Quantra.db";
        public static readonly string ConnectionString = $"Data Source={DbFilePath};Version=3;Journal Mode=WAL;Busy Timeout=30000;";
        private static bool initialized = false;
        private static IConfiguration _configuration;
        
        // Store API keys for backward compatibility
        public static string AlphaVantageApiKey { get; internal set; }

        /// <summary>
        /// Gets a new SQLite database connection with automatic initialization.
        /// </summary>
        /// <returns>A new SQLiteConnection configured with the application's connection string</returns>
        /// <remarks>
        /// This method ensures the database is initialized before returning a connection.
        /// Each call returns a new connection instance - callers are responsible for proper disposal.
        /// The connection string includes WAL mode and optimized timeout settings for better concurrency.
        /// </remarks>
        /// <example>
        /// <code>
        /// using (var connection = DatabaseMonolith.GetConnection())
        /// {
        ///     connection.Open();
        ///     // Perform database operations
        /// }
        /// </code>
        /// </example>
        public static SQLiteConnection GetConnection()
        {
            if (!initialized)
                Initialize();
                
            return new SQLiteConnection(ConnectionString);
        }

        /// <summary>
        /// Logs a message to the database with optional details and automatic error alerting.
        /// </summary>
        /// <param name="level">Log level (Info, Warning, Error, Debug)</param>
        /// <param name="message">The main log message</param>
        /// <param name="details">Optional additional details. For Error level, file/method context is automatically added if not present</param>
        /// <remarks>
        /// This method provides centralized logging for the entire application. Error-level logs
        /// automatically generate alerts and include stack trace information for debugging.
        /// The method ensures database schema compatibility by automatically migrating column
        /// structures when needed (LogLevel -> Level, adding Details column).
        /// 
        /// Logging is resilient - if database logging fails, messages are written to console
        /// to ensure no log data is lost.
        /// </remarks>
        /// <example>
        /// <code>
        /// // Basic info logging
        /// DatabaseMonolith.Log("Info", "User logged in", "UserID: 123");
        /// 
        /// // Error logging (automatically adds context)
        /// DatabaseMonolith.Log("Error", "Database connection failed", "Timeout after 30s");
        /// 
        /// // Warning logging
        /// DatabaseMonolith.Log("Warning", "API rate limit approaching", "85% of daily quota used");
        /// </code>
        /// </example>
        public static void Log(string level, string message, string details = null)
        {
            try
            {
                // Add file/method info if not already present
                if (level == "Error" && (details == null || !details.Contains("File:")))
                {
                    var stack = new System.Diagnostics.StackTrace(1, true);
                    var frame = stack.GetFrame(0);
                    var method = frame?.GetMethod();
                    string file = method?.DeclaringType?.Name ?? "UnknownFile";
                    string methodName = method?.Name ?? "UnknownMethod";
                    details = $"File: {file}, Method: {methodName}, Details: {details}";
                }

                using var connection = GetConnection();
                connection.Open();
                EnsureLogsTableHasLevelColumn(connection); // Ensure Level column exists
                EnsureLogsTableHasDetailsColumn(connection); // Ensure Details column exists
                string sql = @"
                    INSERT INTO Logs (Level, Message, Details, Timestamp)
                    VALUES (@Level, @Message, @Details, @Timestamp)";

                using var command = connection.CreateCommand();
                command.CommandText = sql;
                command.Parameters.AddWithValue("@Level", level);
                command.Parameters.AddWithValue("@Message", message);
                command.Parameters.AddWithValue("@Details", (object)details ?? DBNull.Value);
                command.Parameters.AddWithValue("@Timestamp", DateTime.Now);
                command.ExecuteNonQuery();

                // If this is an error, emit to all AlertsControl instances
                if (level == "Error")
                {
                    var alert = new Quantra.Models.AlertModel
                    {
                        Name = message,
                        Condition = "Error",
                        AlertType = "Error",
                        IsActive = true,
                        Priority = 1,
                        CreatedDate = DateTime.Now,
                        Category = Quantra.Models.AlertCategory.Global,
                        Notes = details
                    };
                    Quantra.Controls.AlertsControl.EmitGlobalAlert(alert);
                }
            }
            catch (Exception ex)
            {
                // If logging fails, write to console at minimum
                Console.WriteLine($"Failed to log: {level} - {message}");
                Console.WriteLine($"Error: {ex.Message}");
            }
        }

        /// <summary>
        /// Sets the application configuration for database operations.
        /// </summary>
        /// <param name="configuration">IConfiguration instance containing app settings</param>
        /// <remarks>
        /// This method configures the database layer with application settings and handles
        /// migration from legacy configuration files (alphaVantageSettings.json) to the
        /// new configuration system. API keys and other sensitive settings are loaded
        /// and stored securely.
        /// 
        /// Should be called during application startup before other database operations.
        /// </remarks>
        /// <example>
        /// <code>
        /// var config = new ConfigurationBuilder()
        ///     .AddJsonFile("appsettings.json")
        ///     .Build();
        /// DatabaseMonolith.SetConfiguration(config);
        /// </code>
        /// </example>
        public static void SetConfiguration(IConfiguration configuration)
        {
            _configuration = configuration;
            
            // Copy any settings from alphaVantageSettings.json to main config if they exist
            if (File.Exists("alphaVantageSettings.json"))
            {
                try
                {
                    var alphaVantageSettings = JsonConvert.DeserializeObject<Dictionary<string, string>>(
                        File.ReadAllText("alphaVantageSettings.json"));
                    
                    if (alphaVantageSettings.TryGetValue("AlphaVantageApiKey", out var apiKey) && !string.IsNullOrWhiteSpace(apiKey))
                    {
                        // This is just temporary until we fully migrate to the new config system
                        // We'll store this in memory only for now
                        AlphaVantageApiKey = apiKey;
                    }
                }
                catch (Exception ex)
                {
                    Log("Error", "Error loading Alpha Vantage settings", ex.ToString());
                }
            }
        }

        static DatabaseMonolith()
        {
            Initialize();
        }

        private static void CreateDatabase()
        {
            SQLiteConnection.CreateFile(DbFilePath);

            using (var connection = GetConnection())
            {
                connection.Open();
                // DropAllTables is now available to be uncommented for development use
                // DropAllTables(connection);

                var createTableQuery = @"
                    CREATE TABLE UserAppSettings (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        TabName TEXT NOT NULL,
                        TabOrder INTEGER NOT NULL,
                        CardPositions TEXT,
                        ControlsConfig TEXT,
                        ToolsConfig TEXT,
                        GridRows INTEGER DEFAULT 4,
                        GridColumns INTEGER DEFAULT 4
                    );
                    CREATE TABLE UserCredentials (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Username TEXT NOT NULL,
                        Password TEXT NOT NULL,
                        Pin TEXT,
                        LastLoginDate DATETIME
                    );
                    CREATE TABLE Logs (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Timestamp DATETIME NOT NULL,
                        Level TEXT NOT NULL,
                        Message TEXT NOT NULL,
                        Exception TEXT
                    );
                    CREATE TABLE TabConfigs (
                        TabName TEXT NOT NULL PRIMARY KEY,
                        ToolsConfig TEXT
                    )";
                connection.Execute(createTableQuery);
                
                // Create Settings table separately with the EnsureSettingsTable method
                EnsureSettingsTable(connection);
                
                // Add default tab
                var defaultTabQuery = @"
                    INSERT INTO UserAppSettings (TabName, TabOrder, GridRows, GridColumns) 
                    VALUES ('Dashboard', 0, 4, 4);";
                connection.Execute(defaultTabQuery);
                
                Log("Info", "Database created successfully with default schema and data");
            }
        }

        /// <summary>
        /// Drops all user tables from the database - USE WITH EXTREME CAUTION.
        /// </summary>
        /// <param name="connection">Open SQLite connection to use for operations</param>
        /// <remarks>
        /// This method is intended for development use only. It will permanently delete
        /// all application data including logs, settings, trading history, and user configurations.
        /// 
        /// Only call this method during development, testing, or when performing a complete
        /// application reset with user consent.
        /// </remarks>
        /// <example>
        /// <code>
        /// // Development/testing only
        /// using (var connection = DatabaseMonolith.GetConnection())
        /// {
        ///     connection.Open();
        ///     DatabaseMonolith.DropAllTables(connection);
        /// }
        /// </code>
        /// </example>
        public static void DropAllTables(SQLiteConnection connection)
        {
            try
            {
                Log("Info", "Dropping all database tables");

                // Get a list of all tables in the database
                var tables = connection.Query<string>(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'");

                Log("Info", $"Found {tables.Count()} tables to drop");

                // Drop each table
                foreach (var table in tables)
                {
                    Log("Info", $"Dropping table: {table}");
                    connection.Execute($"DROP TABLE IF EXISTS {table}");
                }

                Log("Info", "All tables dropped successfully");
            }
            catch (Exception ex)
            {
                // Just log the error but don't show Message Box
                LogErrorWithContext(ex, "Error dropping tables");
                // Optionally, rethrow or handle in another way if needed
                throw new DatabaseException("Failed to drop database tables", ex);
            }
        }

        public static void DropAndRecreateAllTables()
        {
            try
            {
                // Store log messages temporarily since we can't log to database while tables are dropped
                List<(string Level, string Message, string Exception, DateTime Timestamp)> pendingLogs =
                    new List<(string, string, string, DateTime)>();

                pendingLogs.Add(("Info", "Starting database reset - dropping and recreating all tables", null, DateTime.Now));

                // Check if database file exists, if not, just create it
                if (!File.Exists(DbFilePath))
                {
                    pendingLogs.Add(("Info", "Database file does not exist. Creating new database.", null, DateTime.Now));

                    // We can create the database directly since we don't need to drop tables
                    SQLiteConnection.CreateFile(DbFilePath);
                    using (var connection = new SQLiteConnection($"Data Source={DbFilePath};Version=3;"))
                    {
                        connection.Open();
                        CreateDatabaseTables(connection);

                        // Now we can log all pending messages
                        foreach (var log in pendingLogs)
                        {
                            LogToDatabase(connection, log.Level, log.Message, log.Exception, log.Timestamp);
                        }

                        pendingLogs.Add(("Info", "Database created successfully with default schema and data", null, DateTime.Now));
                    }
                    return;
                }

                // For existing database, we need to drop and recreate
                using (var connection = new SQLiteConnection($"Data Source={DbFilePath};Version=3;"))
                {
                    connection.Open();

                    // Begin transaction to ensure all operations complete or none of them do
                    using (var transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            // Drop all existing tables
                            pendingLogs.Add(("Info", "Dropping all database tables", null, DateTime.Now));
                            var tables = connection.Query<string>(
                                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'");

                            pendingLogs.Add(("Info", $"Found {tables.Count()} tables to drop", null, DateTime.Now));

                            // Drop each table
                            foreach (var table in tables)
                            {
                                pendingLogs.Add(("Info", $"Dropping table: {table}", null, DateTime.Now));
                                connection.Execute($"DROP TABLE IF EXISTS {table}");
                            }

                            pendingLogs.Add(("Info", "All tables dropped successfully", null, DateTime.Now));

                            // Recreate tables with current schema
                            pendingLogs.Add(("Info", "Recreating tables with current schema", null, DateTime.Now));
                            CreateDatabaseTables(connection);

                            // Commit transaction after all tables are recreated
                            transaction.Commit();

                            // Now we can log all pending messages since we have recreated the Logs table
                            foreach (var log in pendingLogs)
                            {
                                LogToDatabase(connection, log.Level, log.Message, log.Exception, log.Timestamp);
                            }

                            pendingLogs.Add(("Info", "Database reset completed successfully", null, DateTime.Now));
                            // Log the final message
                            LogToDatabase(connection, "Info", "Database reset completed successfully", null, DateTime.Now);
                        }
                        catch (Exception ex)
                        {
                            // Rollback if any error occurs during recreation
                            transaction.Rollback();
                            LogErrorWithContext(ex, "Error during database recreation");
                            throw new DatabaseException("Error during database recreation", ex);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                // Since we can't guarantee the Logs table exists at this point, 
                // we'll just throw the exception to be handled by the caller
                LogErrorWithContext(ex, "Failed to reset database");
                throw new DatabaseException("Failed to reset database", ex);
            }
        }



        // Helper method to create all database tables
        private static void CreateDatabaseTables(SQLiteConnection connection)
        {
            try
            {
                // Create all the standard tables
                var createTablesQuery = @"
            CREATE TABLE UserAppSettings (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                TabName TEXT NOT NULL,
                TabOrder INTEGER NOT NULL,
                CardPositions TEXT,
                ControlsConfig TEXT,
                ToolsConfig TEXT,
                GridRows INTEGER DEFAULT 4,
                GridColumns INTEGER DEFAULT 4
            );
            CREATE TABLE UserCredentials (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                Username TEXT NOT NULL,
                Password TEXT NOT NULL,
                Pin TEXT,
                LastLoginDate DATETIME
            );
            CREATE TABLE Logs (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                Timestamp DATETIME NOT NULL,
                Level TEXT NOT NULL,
                Message TEXT NOT NULL,
                Exception TEXT
            );
            CREATE TABLE TabConfigs (
                TabName TEXT NOT NULL PRIMARY KEY,
                ToolsConfig TEXT
            );";

                connection.Execute(createTablesQuery);

                // Create Settings table separately
                EnsureSettingsTable(connection);

                // Add default tab
                var defaultTabQuery = @"
            INSERT INTO UserAppSettings (TabName, TabOrder, GridRows, GridColumns) 
            VALUES ('Dashboard', 0, 4, 4);";
                connection.Execute(defaultTabQuery);

                // Verify tables were created
                VerifyTablesExist(connection);

                Log("Info", "Database tables created successfully");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to create database tables");
                throw new DatabaseException("Failed to create database tables", ex);
            }
        }

        // New method to verify tables exist after creation
        private static void VerifyTablesExist(SQLiteConnection connection)
        {
            var requiredTables = new[] { "UserAppSettings", "UserCredentials", "Logs", "TabConfigs", "Settings", "UserPreferences" };
            foreach (var tableName in requiredTables)
            {
                var tableExists = connection.ExecuteScalar<int>(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=@TableName", 
                    new { TableName = tableName });
                
                if (tableExists == 0)
                {
                    throw new DatabaseException($"Failed to create required table: {tableName}");
                }
            }
        }

        // Helper method to log directly to the database with a specific connection
        private static void LogToDatabase(SQLiteConnection connection, string level, string message, string exception = null, DateTime? timestamp = null)
        {
            var insertQuery = @"
        INSERT INTO Logs (Timestamp, Level, Message, Exception)
        VALUES (@Timestamp, @Level, @Message, @Exception)";

            connection.Execute(insertQuery, new
            {
                Timestamp = timestamp ?? DateTime.Now,
                Level = level,
                Message = message,
                Exception = exception
            });
        }

        // Add this custom exception class at the bottom of the file or in a separate file
        public class DatabaseException : Exception
        {
            public DatabaseException(string message) : base(message)
            {
            }

            public DatabaseException(string message, Exception innerException) : base(message, innerException)
            {
            }
        }

        public static void EnsureDatabaseAndTables()
        {
            if (!File.Exists(DbFilePath))
            {
                CreateDatabase();
                return;
            }

            using (var connection = GetConnection())
            {
                connection.Open();

                var createTableQuery = @"
                    CREATE TABLE IF NOT EXISTS UserAppSettings (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        TabName TEXT NOT NULL,
                        TabOrder INTEGER NOT NULL,
                        CardPositions TEXT,
                        ControlsConfig TEXT,
                        ToolsConfig TEXT,
                        GridRows INTEGER DEFAULT 4,
                        GridColumns INTEGER DEFAULT 4
                    );
                    CREATE TABLE IF NOT EXISTS UserCredentials (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Username TEXT NOT NULL,
                        Password TEXT NOT NULL,
                        Pin TEXT,
                        LastLoginDate DATETIME
                    );
                    CREATE TABLE IF NOT EXISTS Logs (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Timestamp DATETIME NOT NULL,
                        Level TEXT NOT NULL,
                        Message TEXT NOT NULL,
                        Exception TEXT
                    );
                    CREATE TABLE IF NOT EXISTS TabConfigs (
                        TabName TEXT NOT NULL PRIMARY KEY,
                        ToolsConfig TEXT
                    );
                    CREATE TABLE IF NOT EXISTS UserPreferences (
                        Key TEXT PRIMARY KEY,
                        Value TEXT,
                        LastUpdated DATETIME NOT NULL
                    )";
                connection.Execute(createTableQuery);
                
                // Ensure Settings table exists
                EnsureSettingsTable(connection);
            }
        }

        public static void EnsureUserAppSettingsTable()
        {
            using (var connection = GetConnection())
            {
                connection.Open();

                var createTablesQuery = @"
                    CREATE TABLE IF NOT EXISTS UserAppSettings (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        TabName TEXT NOT NULL,
                        TabOrder INTEGER NOT NULL,
                        CardPositions TEXT,
                        ControlsConfig TEXT,
                        ToolsConfig TEXT,
                        GridRows INTEGER DEFAULT 4,
                        GridColumns INTEGER DEFAULT 4
                    );
                    
                    CREATE TABLE IF NOT EXISTS UserCredentials (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Username TEXT NOT NULL,
                        Password TEXT NOT NULL,
                        Pin TEXT,
                        LastLoginDate DATETIME
                    );
                    
                    CREATE TABLE IF NOT EXISTS Logs (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Timestamp DATETIME NOT NULL,
                        Level TEXT NOT NULL,
                        Message TEXT NOT NULL,
                        Exception TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS TabConfigs (
                        TabName TEXT NOT NULL PRIMARY KEY,
                        ToolsConfig TEXT
                    );";

                connection.Execute(createTablesQuery);
                
                // Ensure Settings table exists
                EnsureSettingsTable(connection);
                
                Log("Info", "Ensured all database tables exist");
            }
        }

        // Rest of your methods remain unchanged
        public static void SaveCardPositions(string cardPositions)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var updateQuery = "UPDATE UserAppSettings SET CardPositions = @CardPositions";
                connection.Execute(updateQuery, new { CardPositions = cardPositions });
            }
        }

        public static string LoadCardPositions()
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var selectQuery = "SELECT CardPositions FROM UserAppSettings LIMIT 1";
                return connection.QueryFirstOrDefault<string>(selectQuery);
            }
        }

        /// <summary>
        /// Stores user login credentials for remember me functionality.
        /// </summary>
        /// <param name="username">Trading account username</param>
        /// <param name="password">Trading account password (should be encrypted in production)</param>
        /// <param name="pin">Trading account PIN</param>
        /// <remarks>
        /// Saves trading account credentials for automatic login functionality.
        /// Security warning: Consider implementing proper encryption for stored passwords
        /// in production environments. This method is primarily for development convenience.
        /// 
        /// Records the login timestamp for account management purposes.
        /// </remarks>
        /// <example>
        /// <code>
        /// DatabaseMonolith.RememberAccount("trader123", "encryptedPassword", "1234");
        /// </code>
        /// </example>
        public static void RememberAccount(string username, string password, string pin)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var insertQuery = @"
                            INSERT INTO UserCredentials (Username, Password, Pin, LastLoginDate)
                            VALUES (@Username, @Password, @Pin, @LastLoginDate)";
                connection.Execute(insertQuery, new { Username = username, Password = password, Pin = pin, LastLoginDate = DateTime.Now });
            }
        }

        /// <summary>
        /// Retrieves all remembered trading account credentials.
        /// </summary>
        /// <returns>Dictionary mapping usernames to credential tuples (Username, Password, Pin)</returns>
        /// <remarks>
        /// Loads saved trading account credentials for automatic login functionality.
        /// Returns an empty dictionary if no accounts are remembered or if the UserCredentials
        /// table doesn't exist yet.
        /// 
        /// The method gracefully handles database schema issues by creating an empty result
        /// rather than throwing exceptions.
        /// </remarks>
        /// <example>
        /// <code>
        /// var accounts = DatabaseMonolith.GetRememberedAccounts();
        /// foreach (var account in accounts)
        /// {
        ///     string username = account.Key;
        ///     var (user, password, pin) = account.Value;
        ///     // Use credentials for auto-login
        /// }
        /// </code>
        /// </example>
        public static Dictionary<string, (string Username, string Password, string Pin)> GetRememberedAccounts()
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                
                // Check if table exists before querying
                var tableExists = connection.ExecuteScalar<int>(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='UserCredentials'");
                    
                if (tableExists == 0)
                {
                    // Table doesn't exist, return empty dictionary
                    Log("Warning", "UserCredentials table doesn't exist. Returning empty accounts dictionary.");
                    return new Dictionary<string, (string Username, string Password, string Pin)>();
                }
                
                var selectQuery = "SELECT Username, Password, Pin FROM UserCredentials";
                var accounts = connection.Query<(string Username, string Password, string Pin)>(selectQuery);
                var accountDict = new Dictionary<string, (string Username, string Password, string Pin)>();
                foreach (var account in accounts)
                {
                    accountDict[account.Username] = account;
                }
                return accountDict;
            }
        }

        public static void SetPinForCurrentUser(string pin)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var updateQuery = "UPDATE UserCredentials SET Pin = @Pin WHERE Username = @Username";
                connection.Execute(updateQuery, new { Pin = pin, Username = "current_user" }); // Replace "current_user" with the actual current user's username
            }
        }

        public static void SaveUserSettings(string pin, bool enableApiModalChecks)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var updateQuery = @"
                        INSERT INTO UserSettings (Pin, EnableApiModalChecks)
                        VALUES (@Pin, @EnableApiModalChecks)
                        ON CONFLICT(Id) DO UPDATE SET
                        Pin = excluded.Pin,
                        EnableApiModalChecks = excluded.EnableApiModalChecks";
                connection.Execute(updateQuery, new { Pin = pin, EnableApiModalChecks = enableApiModalChecks });
            }
        }

        /// <summary>
        /// Saves UI control configuration for a specific tab.
        /// </summary>
        /// <param name="tabName">Name of the tab</param>
        /// <param name="controlsConfig">Serialized control configuration string</param>
        /// <remarks>
        /// Stores the layout and configuration of UI controls for persistence across application sessions.
        /// If the tab doesn't exist, it creates a new entry with default grid settings.
        /// Control configurations are stored as serialized strings (typically semicolon-separated).
        /// </remarks>
        /// <example>
        /// <code>
        /// // Save control layout for a tab
        /// string config = "PredictionControl,0,0,2,2;ChartControl,0,2,1,2";
        /// DatabaseMonolith.SaveControlsConfig("Trading Dashboard", config);
        /// </code>
        /// </example>
        public static void SaveControlsConfig(string tabName, string controlsConfig)
        {
            using (var connection = GetConnection())
            {
                connection.Open();

                // First check if the tab exists
                var tabExists = connection.QuerySingleOrDefault<int>(
                    "SELECT COUNT(*) FROM UserAppSettings WHERE TabName = @TabName",
                    new { TabName = tabName }) > 0;

                if (tabExists)
                {
                    // Update existing tab
                    var updateQuery = "UPDATE UserAppSettings SET ControlsConfig = @ControlsConfig WHERE TabName = @TabName";
                    connection.Execute(updateQuery, new { ControlsConfig = controlsConfig, TabName = tabName });
                }
                else
                {
                    // Insert new tab with default values
                    var insertQuery = "INSERT INTO UserAppSettings (TabName, TabOrder, ControlsConfig, GridRows, GridColumns) VALUES (@TabName, 0, @ControlsConfig, 4, 4)";
                    connection.Execute(insertQuery, new { TabName = tabName, ControlsConfig = controlsConfig });
                }
            }
        }

        /// <summary>
        /// Loads UI control configuration for a specific tab.
        /// </summary>
        /// <param name="tabName">Name of the tab to load configuration for</param>
        /// <returns>Serialized control configuration string, or null if tab not found</returns>
        /// <remarks>
        /// Retrieves previously saved control layout and configuration for the specified tab.
        /// Returns null if no configuration exists for the tab.
        /// </remarks>
        /// <example>
        /// <code>
        /// string config = DatabaseMonolith.LoadControlsConfig("Trading Dashboard");
        /// if (config != null)
        /// {
        ///     // Parse and apply the control configuration
        ///     string[] controls = config.Split(';');
        /// }
        /// </code>
        /// </example>
        public static string LoadControlsConfig(string tabName)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var selectQuery = "SELECT ControlsConfig FROM UserAppSettings WHERE TabName = @TabName";
                return connection.QueryFirstOrDefault<string>(selectQuery, new { TabName = tabName });
            }
        }

        public static void SaveToolsConfig(string tabName, string toolsConfig)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var query = "INSERT INTO TabConfigs (TabName, ToolsConfig) VALUES (@TabName, @ToolsConfig) " +
                            "ON CONFLICT(TabName) DO UPDATE SET ToolsConfig = @ToolsConfig";
                connection.Execute(query, new { TabName = tabName, ToolsConfig = toolsConfig });
            }
        }

        public static string LoadToolsConfig(string tabName)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var query = "SELECT ToolsConfig FROM TabConfigs WHERE TabName = @TabName";
                return connection.QuerySingleOrDefault<string>(query, new { TabName = tabName }) ?? string.Empty;
            }
        }

        public static (int Rows, int Columns) LoadGridConfig(string tabName)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var selectQuery = "SELECT GridRows, GridColumns FROM UserAppSettings WHERE TabName = @TabName";
                var result = connection.QueryFirstOrDefault<(int Rows, int Columns)>(selectQuery, new { TabName = tabName });

                // If no configuration exists yet or values are invalid, return defaults
                if (result.Rows <= 0 || result.Columns <= 0)
                {
                    return (4, 4);
                }
                return result;
            }
        }

        public static void SaveGridConfig(string tabName, int rows, int columns)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var updateQuery = "UPDATE UserAppSettings SET GridRows = @Rows, GridColumns = @Columns WHERE TabName = @TabName";
                connection.Execute(updateQuery, new { Rows = rows, Columns = columns, TabName = tabName });
            }
        }

        public static Models.DataGridSettings LoadDataGridConfig(string tabName, string controlName)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                var selectQuery = "SELECT DataGridConfig FROM UserAppSettings WHERE TabName = @TabName";
                var configJson = connection.QueryFirstOrDefault<string>(selectQuery, new { TabName = tabName });

                if (string.IsNullOrEmpty(configJson))
                {
                    return new Models.DataGridSettings();
                }

                try
                {
                    // Parse the JSON to get all DataGrid configs for this tab
                    var allConfigs = JsonConvert.DeserializeObject<Dictionary<string, Models.DataGridSettings>>(configJson);
                    if (allConfigs != null && allConfigs.ContainsKey(controlName))
                    {
                        return allConfigs[controlName];
                    }
                }
                catch (Exception ex)
                {
                    Log("Warning", $"Failed to parse DataGridConfig for tab '{tabName}', control '{controlName}'", ex.ToString());
                }

                return new Models.DataGridSettings();
            }
        }

        public static void SaveDataGridConfig(string tabName, string controlName, Models.DataGridSettings settings)
        {
            using (var connection = GetConnection())
            {
                connection.Open();
                
                // First, get existing config
                var selectQuery = "SELECT DataGridConfig FROM UserAppSettings WHERE TabName = @TabName";
                var configJson = connection.QueryFirstOrDefault<string>(selectQuery, new { TabName = tabName });

                Dictionary<string, Models.DataGridSettings> allConfigs;
                if (string.IsNullOrEmpty(configJson))
                {
                    allConfigs = new Dictionary<string, Models.DataGridSettings>();
                }
                else
                {
                    try
                    {
                        allConfigs = JsonConvert.DeserializeObject<Dictionary<string, Models.DataGridSettings>>(configJson) 
                                   ?? new Dictionary<string, Models.DataGridSettings>();
                    }
                    catch (Exception ex)
                    {
                        Log("Warning", $"Failed to parse existing DataGridConfig for tab '{tabName}'", ex.ToString());
                        allConfigs = new Dictionary<string, Models.DataGridSettings>();
                    }
                }

                // Update the specific control's settings
                allConfigs[controlName] = settings;

                // Serialize back to JSON
                var updatedConfigJson = JsonConvert.SerializeObject(allConfigs);

                // Update or insert the record
                var updateQuery = "UPDATE UserAppSettings SET DataGridConfig = @DataGridConfig WHERE TabName = @TabName";
                var rowsAffected = connection.Execute(updateQuery, new { DataGridConfig = updatedConfigJson, TabName = tabName });

                if (rowsAffected == 0)
                {
                    // Don't create tab entries for StockExplorer fallback names (auto-generated GUIDs)
                    // These are temporary names used for DataGrid settings persistence only
                    if (!tabName.StartsWith("StockExplorer_"))
                    {
                        // Insert new record if tab doesn't exist and it's not an auto-generated name
                        var insertQuery = "INSERT INTO UserAppSettings (TabName, TabOrder, DataGridConfig, GridRows, GridColumns) VALUES (@TabName, 0, @DataGridConfig, 4, 4)";
                        connection.Execute(insertQuery, new { TabName = tabName, DataGridConfig = updatedConfigJson });
                    }
                    // For StockExplorer fallback names, just log that settings were not saved to avoid tab creation
                    else
                    {
                        Log("Info", $"Skipped creating tab entry for auto-generated StockExplorer name: {tabName}");
                    }
                }
            }
        }

        public static void AddCustomControl(string tabName, string controlDefinition)
        {
            try
            {
                var currentControls = LoadControlsConfig(tabName) ?? string.Empty;

                // Fix: Ensure we're using semicolons as separators instead of newlines
                // Trim any trailing semicolons to avoid empty entries
                currentControls = currentControls.Trim().TrimEnd(';');

                // Append the new control with proper semicolon separator
                var updatedControls = string.IsNullOrEmpty(currentControls)
                    ? controlDefinition
                    : currentControls + ";" + controlDefinition;

                SaveControlsConfig(tabName, updatedControls);

                Log("Info", $"Added control to tab '{tabName}': {controlDefinition}");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to add control to tab '{tabName}'");
            }
        }

        public static void UpdateControlPosition(string tabName, int controlIndex, int row, int column, int rowSpan, int columnSpan)
        {
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();

                    // Get existing controls configuration
                    string controlsConfig = LoadControlsConfig(tabName) ?? string.Empty;

                    // Parse controls configuration into list of controls
                    var controls = controlsConfig.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries).ToList();

                    // Make sure index is valid
                    if (controlIndex >= 0 && controlIndex < controls.Count)
                    {
                        // Parse the control definition
                        string[] parts = controls[controlIndex].Split(',');

                        // Extract the control type (first part)
                        string controlType = parts[0];

                        // Create updated control definition with spans
                        string updatedControl = $"{controlType},{row},{column},{rowSpan},{columnSpan}";

                        // Replace the control at the specified index
                        controls[controlIndex] = updatedControl;

                        // Save the updated controls configuration
                        string updatedConfig = string.Join(";", controls);
                        SaveControlsConfig(tabName, updatedConfig);

                        Log("Info", $"Updated control position in tab '{tabName}': {updatedControl}");
                    }
                    else
                    {
                        Log("Error", $"Control index {controlIndex} is out of range for tab '{tabName}'");
                    }
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to update control position for tab '{tabName}'");
            }
        }

        public static void AddCustomControlWithSpans(string tabName, string controlType, int row, int column, int rowSpan, int columnSpan)
        {
            try
            {
                var controlDefinition = $"{controlType},{row},{column},{rowSpan},{columnSpan}";
                var currentControls = LoadControlsConfig(tabName) ?? string.Empty;

                // Fix: Ensure we're using semicolons as separators instead of newlines
                // Trim any trailing semicolons to avoid empty entries
                currentControls = currentControls.Trim().TrimEnd(';');

                // Append the new control with proper semicolon separator
                var updatedControls = string.IsNullOrEmpty(currentControls)
                    ? controlDefinition
                    : currentControls + ";" + controlDefinition;

                SaveControlsConfig(tabName, updatedControls);

                Log("Info", $"Added control to tab '{tabName}' with spans: {controlDefinition}");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to add control with spans to tab '{tabName}'");
            }
        }

        public static void RemoveControl(string tabName, int controlIndex)
        {
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();

                    // Get existing controls configuration
                    string controlsConfig = LoadControlsConfig(tabName) ?? string.Empty;

                    // Parse controls configuration into list of controls
                    var controls = controlsConfig.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries).ToList();

                    // Make sure index is valid
                    if (controlIndex >= 0 && controlIndex < controls.Count)
                    {
                        // Remove the control at the specified index
                        controls.RemoveAt(controlIndex);

                        // Save the updated controls configuration
                        string updatedConfig = string.Join(";", controls);
                        SaveControlsConfig(tabName, updatedConfig);

                        Log("Info", $"Removed control at index {controlIndex} from tab '{tabName}'");
                    }
                    else
                    {
                        Log("Error", $"Control index {controlIndex} is out of range for tab '{tabName}'");
                    }
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to remove control from tab '{tabName}'");
            }
        }

        public static UserSettings GetUserSettings()
        {
            try
            {
                // Get the default settings profile and convert it to UserSettings
                var profile = SettingsService.GetDefaultSettingsProfile();
                if (profile != null)
                {
                    return profile.ToUserSettings();
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to get default settings profile");
            }
            
            // Fallback to default settings
            return new UserSettings();
        }

        private static void EnsureSettingsTable(SQLiteConnection connection)
        {
            // Check if Settings table exists
            var tableCheckQuery = "SELECT name FROM sqlite_master WHERE type='table' AND name='Settings'";
            using (var command = new SQLiteCommand(tableCheckQuery, connection))
            {
                if (command.ExecuteScalar() == null)
                {
                    // Create Settings table if it doesn't exist
                    var createTableQuery = @"
                        CREATE TABLE Settings (
                            ID INTEGER PRIMARY KEY,
                            EnableApiModalChecks INTEGER,
                            ApiTimeoutSeconds INTEGER DEFAULT 30,
                            CacheDurationMinutes INTEGER DEFAULT 15,
                            EnableHistoricalDataCache INTEGER DEFAULT 1,
                            EnableDarkMode INTEGER DEFAULT 1,
                            ChartUpdateIntervalSeconds INTEGER DEFAULT 2,
                            EnablePriceAlerts INTEGER DEFAULT 1,
                            EnableTradeNotifications INTEGER DEFAULT 1,
                            EnablePaperTrading INTEGER DEFAULT 1,
                            RiskLevel TEXT DEFAULT 'Low',
                            DefaultGridRows INTEGER DEFAULT 4,
                            DefaultGridColumns INTEGER DEFAULT 4,
                            GridBorderColor TEXT DEFAULT '#FF00FFFF',
                            AlertEmail TEXT DEFAULT 'tylortrub@gmail.com'
                        )";
                    
                    using (var createCommand = new SQLiteCommand(createTableQuery, connection))
                    {
                        createCommand.ExecuteNonQuery();
                    }
                    
                    // Insert default settings
                    var insertDefaultQuery = @"
                        INSERT INTO Settings (
                            ID, EnableApiModalChecks, ApiTimeoutSeconds, CacheDurationMinutes, 
                            EnableHistoricalDataCache, EnableDarkMode, ChartUpdateIntervalSeconds,
                            EnablePriceAlerts, EnableTradeNotifications, EnablePaperTrading, RiskLevel,
                            DefaultGridRows, DefaultGridColumns, GridBorderColor, AlertEmail
                        ) VALUES (
                            1, 1, 30, 15, 1, 1, 2, 1, 1, 1, 'Low', 4, 4, '#FF00FFFF', 'tylortrub@gmail.com'
                        )";
                    
                    using (var insertCommand = new SQLiteCommand(insertDefaultQuery, connection))
                    {
                        insertCommand.ExecuteNonQuery();
                    }
                    
                    Log("Info", "Created Settings table with default values");
                }
                else
                {
                    // Table exists, check if we need to add the AlertEmail column
                    var columnCheckQuery = "PRAGMA table_info(Settings)";
                    bool hasAlertEmailColumn = false;
                    
                    using (var checkCommand = new SQLiteCommand(columnCheckQuery, connection))
                    using (var reader = checkCommand.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            if (reader["name"].ToString() == "AlertEmail")
                            {
                                hasAlertEmailColumn = true;
                                break;
                            }
                        }
                    }
                    
                    // Add AlertEmail column if it doesn't exist
                    if (!hasAlertEmailColumn)
                    {
                        var addColumnQuery = "ALTER TABLE Settings ADD COLUMN AlertEmail TEXT DEFAULT 'tylortrub@gmail.com'";
                        using (var alterCommand = new SQLiteCommand(addColumnQuery, connection))
                        {
                            alterCommand.ExecuteNonQuery();
                            Log("Info", "Added AlertEmail column to Settings table");
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Saves user application settings and preferences.
        /// </summary>
        /// <param name="settings">UserSettings object containing all user preferences</param>
        /// <remarks>
        /// Persists user settings to the database using the settings profile system.
        /// Creates a default profile if none exists and updates all configurable user preferences
        /// including UI settings, API timeouts, trading preferences, and visual configurations.
        /// 
        /// This method integrates with the SettingsService to maintain profile-based settings.
        /// </remarks>
        /// <example>
        /// <code>
        /// var settings = new UserSettings
        /// {
        ///     EnableDarkMode = true,
        ///     ApiTimeoutSeconds = 30,
        ///     EnablePaperTrading = true,
        ///     DefaultGridRows = 4,
        ///     DefaultGridColumns = 6
        /// };
        /// DatabaseMonolith.SaveUserSettings(settings);
        /// </code>
        /// </example>
        public static void SaveUserSettings(UserSettings settings)
        {
            try
            {
                // Get the current default profile
                var profile = SettingsService.GetDefaultSettingsProfile();
                
                // If no profile exists, create one
                if (profile == null)
                {
                    SettingsService.EnsureSettingsProfiles();
                    profile = SettingsService.GetDefaultSettingsProfile();
                    if (profile == null)
                    {
                        Log("Error", "Could not create or retrieve default settings profile");
                        return;
                    }
                }
                
                // Update the profile with new settings
                profile.EnableApiModalChecks = settings.EnableApiModalChecks;
                profile.ApiTimeoutSeconds = settings.ApiTimeoutSeconds;
                profile.CacheDurationMinutes = settings.CacheDurationMinutes;
                profile.EnableHistoricalDataCache = settings.EnableHistoricalDataCache;
                profile.EnableDarkMode = settings.EnableDarkMode;
                profile.ChartUpdateIntervalSeconds = settings.ChartUpdateIntervalSeconds;
                profile.DefaultGridRows = settings.DefaultGridRows;
                profile.DefaultGridColumns = settings.DefaultGridColumns;
                profile.GridBorderColor = settings.GridBorderColor;
                profile.EnablePriceAlerts = settings.EnablePriceAlerts;
                profile.EnableTradeNotifications = settings.EnableTradeNotifications;
                profile.EnablePaperTrading = settings.EnablePaperTrading;
                profile.RiskLevel = settings.RiskLevel;
                
                // Save the updated profile
                SettingsService.UpdateSettingsProfile(profile);
                
                Log("Info", "Settings saved successfully to profile: " + profile.Name);
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Error saving settings to profile");
                throw;
            }
        }

        public static void UpdateSettingsTable()
        {
            // This method is called during application startup to migrate old settings to the new profile system
            EnsureSettingsProfiles();
        }

        /// <summary>
        /// Adds a completed order to the trading history.
        /// </summary>
        /// <param name="order">OrderModel containing order details</param>
        /// <remarks>
        /// Persists trading order information for historical tracking and analysis.
        /// Stores complete order details including symbol, type, quantities, prices,
        /// stop-loss/take-profit levels, and execution metadata.
        /// 
        /// Handles both paper trading and live trading orders with appropriate flagging.
        /// Automatically creates the OrderHistory table if it doesn't exist.
        /// </remarks>
        /// <example>
        /// <code>
        /// var order = new OrderModel
        /// {
        ///     Symbol = "AAPL",
        ///     OrderType = "BUY",
        ///     Quantity = 100,
        ///     Price = 150.50,
        ///     StopLoss = 145.00,
        ///     TakeProfit = 160.00,
        ///     IsPaperTrade = false,
        ///     Status = "FILLED",
        ///     Timestamp = DateTime.Now
        /// };
        /// DatabaseMonolith.AddOrderToHistory(order);
        /// </code>
        /// </example>
        public static void AddOrderToHistory(OrderModel order)
        {
            if (order == null)
            {
                Log("Error", "Cannot add null order to history");
                return;
            }

            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();

                    using (var command = new SQLiteCommand(connection))
                    {
                        command.CommandText = @"
                            INSERT INTO OrderHistory (
                                Symbol, OrderType, Quantity, Price, StopLoss, TakeProfit, 
                                IsPaperTrade, Status, PredictionSource, Timestamp
                            )
                            VALUES (
                                @Symbol, @OrderType, @Quantity, @Price, @StopLoss, @TakeProfit, 
                                @IsPaperTrade, @Status, @PredictionSource, @Timestamp
                            );";

                        command.Parameters.AddWithValue("@Symbol", order.Symbol);
                        command.Parameters.AddWithValue("@OrderType", order.OrderType);
                        command.Parameters.AddWithValue("@Quantity", order.Quantity);
                        command.Parameters.AddWithValue("@Price", order.Price);
                        command.Parameters.AddWithValue("@StopLoss", order.StopLoss);
                        command.Parameters.AddWithValue("@TakeProfit", order.TakeProfit);
                        command.Parameters.AddWithValue("@IsPaperTrade", order.IsPaperTrade ? 1 : 0);
                        command.Parameters.AddWithValue("@Status", order.Status);
                        command.Parameters.AddWithValue("@PredictionSource", order.PredictionSource ?? string.Empty);
                        command.Parameters.AddWithValue("@Timestamp", order.Timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff"));

                        command.ExecuteNonQuery();
                    }
                }

                Log("Info", $"Order added to history: {order.Symbol} {order.OrderType} {order.Quantity} @ {order.Price:C2}");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to add order to history: {order.Symbol}");
            }
        }

        private static void EnsureSettingsProfiles()
        {
            // Just delegate to the SettingsService
            SettingsService.EnsureSettingsProfiles();
        }

        public static string GetSetting(string key, string defaultValue = null)
        {
            // Alias for GetUserPreference to maintain compatibility
            return GetUserPreference(key, defaultValue);
        }

        public static string GetUserPreference(string key, string defaultValue = null)
        {
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();

                    // First ensure the table exists
                    EnsureUserPreferencesTable(connection);
                    
                    // Query for the preference with the given key
                    var selectQuery = "SELECT Value FROM UserPreferences WHERE Key = @Key";
                    using (var command = new SQLiteCommand(selectQuery, connection))
                    {
                        command.Parameters.AddWithValue("@Key", key);
                        var result = command.ExecuteScalar();
                        
                        if (result != null)
                        {
                            string storedValue = result.ToString();
                            // Check if the value is compressed and decompress if needed
                            if (Utilities.CompressionHelper.IsCompressed(storedValue))
                            {
                                return Utilities.CompressionHelper.DecompressString(storedValue);
                            }
                            return storedValue;
                        }
                    }
                }
                
                // Return default if not found
                return defaultValue;
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to retrieve user preference '{key}'");
                return defaultValue;
            }
        }

        public static void SaveSetting(string key, string value)
        {
            // Alias for SaveUserPreference to maintain compatibility
            SaveUserPreference(key, value);
        }

        public static void SaveUserPreference(string key, string value)
        {
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    
                    // First ensure the table exists
                    EnsureUserPreferencesTable(connection);
                    
                    // Compress the value if it's longer than a minimum threshold (100 chars)
                    string storedValue = value;
                    if (!string.IsNullOrEmpty(value) && value.Length > 100)
                    {
                        storedValue = Utilities.CompressionHelper.CompressString(value);
                    }
                    
                    // Insert or update the preference
                    var upsertQuery = @"
                        INSERT INTO UserPreferences (Key, Value, LastUpdated)
                        VALUES (@Key, @Value, @LastUpdated)
                        ON CONFLICT(Key) DO UPDATE SET
                        Value = excluded.Value,
                        LastUpdated = excluded.LastUpdated";
                        
                    using (var command = new SQLiteCommand(upsertQuery, connection))
                    {
                        command.Parameters.AddWithValue("@Key", key);
                        command.Parameters.AddWithValue("@Value", storedValue);
                        command.Parameters.AddWithValue("@LastUpdated", DateTime.Now);
                        command.ExecuteNonQuery();
                    }
                    
                    Log("Info", $"User preference '{key}' saved successfully");
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to save user preference '{key}'");
            }
        }

        private static void EnsureUserPreferencesTable(SQLiteConnection connection)
        {
            // Check if UserPreferences table exists
            var tableCheckQuery = "SELECT name FROM sqlite_master WHERE type='table' AND name='UserPreferences'";
            using (var command = new SQLiteCommand(tableCheckQuery, connection))
            {
                if (command.ExecuteScalar() == null)
                {
                    // Create UserPreferences table if it doesn't exist
                    var createTableQuery = @"
                        CREATE TABLE UserPreferences (
                            Key TEXT PRIMARY KEY,
                            Value TEXT,
                            LastUpdated DATETIME NOT NULL
                        )";
                    
                    using (var createCommand = new SQLiteCommand(createTableQuery, connection))
                    {
                        createCommand.ExecuteNonQuery();
                    }
                    
                    Log("Info", "Created UserPreferences table");
                }
            }
        }

        // Add this method to the DatabaseMonolith class to support retrieving control type by index

        /// <summary>
        /// Gets the control type for a specific control in a tab by its index
        /// </summary>
        /// <param name="tabName">The name of the tab containing the control</param>
        /// <param name="controlIndex">The index of the control in the tab</param>
        /// <returns>The control type as a string, or null if not found</returns>
        public static string GetControlTypeByIndex(string tabName, int controlIndex)
        {
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    
                    // Load the controls config for the tab
                    var configQuery = "SELECT ControlsConfig FROM UserAppSettings WHERE TabName = @TabName";
                    var controlsConfig = connection.QueryFirstOrDefault<string>(configQuery, new { TabName = tabName });
                    
                    if (string.IsNullOrEmpty(controlsConfig))
                        return null;
                    
                    // Split controls by semicolon
                    var controlEntries = controlsConfig
                        .Replace("\r\n", ";")
                        .Replace("\n", ";")
                        .Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
                    
                    // If index is out of range, return null
                    if (controlIndex < 0 || controlIndex >= controlEntries.Length)
                        return null;
                    
                    // Get the control configuration at the specified index
                    var controlConfig = controlEntries[controlIndex];
                    
                    // Parse the control type (first part before comma)
                    var parts = controlConfig.Split(',');
                    if (parts.Length > 0)
                    {
                        return parts[0].Trim();
                    }
                    
                    return null;
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Error getting control type by index");
                return null;
            }
        }

        /// <summary>
        /// Loads controls for a specific tab and returns them as a list of control models
        /// </summary>
        /// <param name="tabName">The name of the tab to load controls for</param>
        /// <returns>List of ControlModel objects for the tab</returns>
        public static List<ControlModel> LoadControlsForTab(string tabName)
        {
            var controlModels = new List<ControlModel>();
            
            try
            {
                // Get the control configuration string from the database
                string controlsConfig = LoadControlsConfig(tabName);
                
                if (string.IsNullOrEmpty(controlsConfig))
                    return controlModels;
                
                // Split into individual control definitions
                var controlDefinitions = controlsConfig.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
                
                // Parse each control definition
                foreach (var definition in controlDefinitions)
                {
                    var parts = definition.Split(',');
                    if (parts.Length >= 3) // At minimum, we need type, row, column
                    {
                        string type = parts[0].Trim();
                        
                        // Try to parse row and column values
                        if (int.TryParse(parts[1], out int row) && 
                            int.TryParse(parts[2], out int column))
                        {
                            int rowSpan = 1;  // Default span
                            int colSpan = 1;  // Default span
                            
                            // Try to parse optional rowSpan and colSpan if provided
                            if (parts.Length > 3 && int.TryParse(parts[3], out int parsedRowSpan))
                                rowSpan = parsedRowSpan;
                            
                            if (parts.Length > 4 && int.TryParse(parts[4], out int parsedColSpan))
                                colSpan = parsedColSpan;
                            
                            controlModels.Add(new ControlModel(type, row, column, rowSpan, colSpan));
                        }
                    }
                }
                
                Log("Info", $"Loaded {controlModels.Count} controls for tab '{tabName}'");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to load controls for tab '{tabName}'");
            }
            
            return controlModels;
        }

        public static async Task<List<double>> GetHistoricalIndicatorData(string symbol, string indicatorType)
        {
            try
            {
                // Get real historical price data from database cache first
                var historicalData = await GetCachedHistoricalPrices(symbol, "daily");
                
                if (historicalData.Count < 10)
                {
                    Log("Warning", $"Insufficient historical data for {symbol} ({historicalData.Count} points), returning empty list");
                    return new List<double>();
                }

                List<double> result = new List<double>();

                // Calculate real indicator values based on historical price data
                switch (indicatorType.ToUpperInvariant())
                {
                    case "RSI":
                        var closingPrices = historicalData.Select(h => h.Close).ToList();
                        var rsiValues = CalculateRSI(closingPrices, Math.Min(14, closingPrices.Count - 1));
                        result = rsiValues.Where(v => !double.IsNaN(v)).ToList();
                        break;

                    case "MACD":
                        var prices = historicalData.Select(h => h.Close).ToList();
                        var (macdLine, signalLine, histogram) = CalculateMACD(prices, 12, 26, 9);
                        
                        // Return both MACD line and signal line values
                        result.AddRange(macdLine.Where(v => !double.IsNaN(v)));
                        result.AddRange(signalLine.Where(v => !double.IsNaN(v)));
                        break;

                    case "VOLUME":
                        result = historicalData.Select(h => (double)h.Volume).ToList();
                        break;

                    case "ADX":
                        var highs = historicalData.Select(h => h.High).ToList();
                        var lows = historicalData.Select(h => h.Low).ToList();
                        var closes = historicalData.Select(h => h.Close).ToList();
                        var adxValues = CalculateADX(highs, lows, closes, Math.Min(14, closes.Count / 2));
                        result = adxValues.Where(v => !double.IsNaN(v)).ToList();
                        break;

                    case "ROC":
                        var rocPrices = historicalData.Select(h => h.Close).ToList();
                        var rocValues = CalculateROC(rocPrices, Math.Min(10, rocPrices.Count / 2));
                        result = rocValues.Where(v => !double.IsNaN(v)).ToList();
                        break;

                    case "BB_WIDTH":
                        // Bollinger Bands Width calculation
                        var bbPrices = historicalData.Select(h => h.Close).ToList();
                        var (upper, middle, lower) = CalculateBollingerBands(bbPrices, Math.Min(20, bbPrices.Count / 2), 2.0);
                        
                        for (int i = 0; i < upper.Count && i < lower.Count; i++)
                        {
                            if (!double.IsNaN(upper[i]) && !double.IsNaN(lower[i]))
                            {
                                double width = (upper[i] - lower[i]) / middle[i] * 100; // Width as percentage
                                result.Add(width);
                            }
                        }
                        break;

                    default:
                        Log("Warning", $"Unknown indicator type: {indicatorType}. Returning empty list");
                        return new List<double>();
                }

                Log("Info", $"Calculated {result.Count} real {indicatorType} values for {symbol} from {historicalData.Count} historical data points");
                return result;
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Error calculating real historical data for {indicatorType} on {symbol}");
                return new List<double>(); // Return empty list on error
            }
        }

        /// <summary>
        /// Gets cached historical price data from database for indicator calculations
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="interval">Data interval</param>
        /// <returns>List of cached historical prices</returns>
        private static async Task<List<HistoricalPrice>> GetCachedHistoricalPrices(string symbol, string interval)
        {
            // Convert interval to timeRange format expected by database
            string timeRange = interval switch
            {
                "1min" => "1day",
                "5min" => "5day", 
                "15min" => "1week",
                "30min" => "2week",
                "1hour" => "1month",
                "daily" => "3month",
                _ => "1month"
            };

            try
            {
                (StockData stockData, DateTime? timestamp) = await GetStockDataWithTimestamp(symbol, timeRange);
                
                if (stockData?.Dates != null && stockData.Dates.Count > 0)
                {
                    var historicalPrices = new List<HistoricalPrice>();
                    
                    for (int i = 0; i < stockData.Dates.Count && i < stockData.Prices.Count; i++)
                    {
                        var price = stockData.Prices[i];
                        var volume = stockData.Volumes != null && i < stockData.Volumes.Count ? (long)stockData.Volumes[i] : 1000;
                        
                        historicalPrices.Add(new HistoricalPrice
                        {
                            Date = stockData.Dates[i],
                            Open = price,
                            High = price * 1.01, // Approximate high as 1% above close
                            Low = price * 0.99,  // Approximate low as 1% below close
                            Close = price,
                            Volume = volume,
                            AdjClose = price
                        });
                    }
                    
                    // Sort by date to ensure chronological order
                    historicalPrices = historicalPrices.OrderBy(h => h.Date).ToList();
                    
                    Log("Info", $"Retrieved {historicalPrices.Count} cached historical prices for {symbol}");
                    return historicalPrices;
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Error retrieving cached historical data for {symbol}");
            }

            return new List<HistoricalPrice>();
        }

        #region Technical Indicator Calculation Methods

        /// <summary>
        /// Calculate RSI (Relative Strength Index) from price data
        /// </summary>
        private static List<double> CalculateRSI(List<double> prices, int period)
        {
            var result = new List<double>();
            
            if (prices.Count <= period)
            {
                for (int i = 0; i < prices.Count; i++)
                {
                    result.Add(double.NaN);
                }
                return result;
            }
            
            // Calculate price changes
            var priceChanges = new List<double>();
            for (int i = 1; i < prices.Count; i++)
            {
                priceChanges.Add(prices[i] - prices[i - 1]);
            }
            
            // Calculate gains and losses
            var gains = new List<double>();
            var losses = new List<double>();
            
            for (int i = 0; i < priceChanges.Count; i++)
            {
                double change = priceChanges[i];
                gains.Add(change > 0 ? change : 0);
                losses.Add(change < 0 ? Math.Abs(change) : 0);
                
                // Add NaN for initial values
                if (i < period - 1)
                {
                    result.Add(double.NaN);
                }
                else
                {
                    // Calculate average gains and average losses for this period
                    double avgGain = gains.Skip(i - period + 1).Take(period).Average();
                    double avgLoss = losses.Skip(i - period + 1).Take(period).Average();
                    
                    // Calculate RS and RSI
                    double rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                    double rsi = 100 - (100 / (1 + rs));
                    result.Add(rsi);
                }
            }
            
            return result;
        }

        /// <summary>
        /// Calculate MACD (Moving Average Convergence Divergence)
        /// </summary>
        private static (List<double> MacdLine, List<double> SignalLine, List<double> Histogram) CalculateMACD(List<double> prices, int fastPeriod, int slowPeriod, int signalPeriod)
        {
            // Calculate EMAs
            var fastEMA = CalculateEMA(prices, fastPeriod);
            var slowEMA = CalculateEMA(prices, slowPeriod);
            
            // Calculate MACD line
            var macdLine = new List<double>();
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < slowPeriod - 1)
                {
                    macdLine.Add(double.NaN);
                }
                else
                {
                    macdLine.Add(fastEMA[i] - slowEMA[i]);
                }
            }
            
            // Calculate signal line (EMA of MACD line)
            var signalLine = CalculateEMA(macdLine, signalPeriod);
            
            // Calculate histogram (MACD - Signal)
            var histogram = new List<double>();
            for (int i = 0; i < macdLine.Count; i++)
            {
                if (i < slowPeriod + signalPeriod - 2)
                {
                    histogram.Add(double.NaN);
                }
                else
                {
                    histogram.Add(macdLine[i] - signalLine[i]);
                }
            }
            
            return (macdLine, signalLine, histogram);
        }

        /// <summary>
        /// Calculate EMA (Exponential Moving Average)
        /// </summary>
        private static List<double> CalculateEMA(List<double> prices, int period)
        {
            var result = new List<double>();
            
            // First EMA value is SMA
            var sma = CalculateSMA(prices, period);
            
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period - 1)
                {
                    result.Add(double.NaN);
                    continue;
                }
                
                if (i == period - 1)
                {
                    result.Add(sma[i]);
                    continue;
                }
                
                // Calculate EMA: EMA = Price * k + EMA(previous) * (1-k)
                // where k = 2/(period+1)
                double multiplier = 2.0 / (period + 1);
                double ema = (prices[i] * multiplier) + (result[i - 1] * (1 - multiplier));
                result.Add(ema);
            }
            
            return result;
        }

        /// <summary>
        /// Calculate SMA (Simple Moving Average)
        /// </summary>
        private static List<double> CalculateSMA(List<double> prices, int period)
        {
            var result = new List<double>();
            
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period - 1)
                {
                    result.Add(double.NaN);
                    continue;
                }
                
                // Calculate SMA for this window
                var sum = 0.0;
                for (int j = i - period + 1; j <= i; j++)
                {
                    sum += prices[j];
                }
                result.Add(sum / period);
            }
            
            return result;
        }

        /// <summary>
        /// Calculate ADX (Average Directional Index)
        /// </summary>
        private static List<double> CalculateADX(List<double> highPrices, List<double> lowPrices, List<double> closePrices, int period = 14)
        {
            var result = new List<double>();
            
            int length = Math.Min(Math.Min(highPrices.Count, lowPrices.Count), closePrices.Count);
            
            if (length < period + 1)
            {
                for (int i = 0; i < length; i++)
                {
                    result.Add(double.NaN);
                }
                return result;
            }
            
            // Calculate True Range (TR) and Directional Movement (+DM, -DM)
            var trueRanges = new List<double>();
            var plusDMs = new List<double>();
            var minusDMs = new List<double>();
            
            for (int i = 1; i < length; i++)
            {
                // True Range
                double tr1 = highPrices[i] - lowPrices[i];
                double tr2 = Math.Abs(highPrices[i] - closePrices[i - 1]);
                double tr3 = Math.Abs(lowPrices[i] - closePrices[i - 1]);
                double tr = Math.Max(tr1, Math.Max(tr2, tr3));
                trueRanges.Add(tr);
                
                // Directional Movement
                double highDiff = highPrices[i] - highPrices[i - 1];
                double lowDiff = lowPrices[i - 1] - lowPrices[i];
                
                double plusDM = (highDiff > lowDiff && highDiff > 0) ? highDiff : 0;
                double minusDM = (lowDiff > highDiff && lowDiff > 0) ? lowDiff : 0;
                
                plusDMs.Add(plusDM);
                minusDMs.Add(minusDM);
            }
            
            // Calculate smoothed versions (using EMA)
            var smoothedTRs = CalculateEMA(trueRanges, period);
            var smoothedPlusDMs = CalculateEMA(plusDMs, period);
            var smoothedMinusDMs = CalculateEMA(minusDMs, period);
            
            // Calculate +DI and -DI
            var plusDIs = new List<double>();
            var minusDIs = new List<double>();
            
            for (int i = 0; i < smoothedTRs.Count; i++)
            {
                double plusDI = smoothedTRs[i] == 0 ? 0 : (smoothedPlusDMs[i] / smoothedTRs[i]) * 100;
                double minusDI = smoothedTRs[i] == 0 ? 0 : (smoothedMinusDMs[i] / smoothedTRs[i]) * 100;
                
                plusDIs.Add(plusDI);
                minusDIs.Add(minusDI);
            }
            
            // Calculate DX
            var dxValues = new List<double>();
            for (int i = 0; i < plusDIs.Count; i++)
            {
                double diSum = plusDIs[i] + minusDIs[i];
                double dx = diSum == 0 ? 0 : (Math.Abs(plusDIs[i] - minusDIs[i]) / diSum) * 100;
                dxValues.Add(dx);
            }
            
            // Calculate ADX (EMA of DX)
            var adxValues = CalculateEMA(dxValues, period);
            
            // Pad with NaN for initial periods
            for (int i = 0; i < period; i++)
            {
                result.Add(double.NaN);
            }
            
            result.AddRange(adxValues);
            
            return result;
        }

        /// <summary>
        /// Calculate ROC (Rate of Change)
        /// </summary>
        private static List<double> CalculateROC(List<double> prices, int period = 10)
        {
            var result = new List<double>();
            
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period)
                {
                    result.Add(double.NaN);
                }
                else
                {
                    double current = prices[i];
                    double previous = prices[i - period];
                    
                    if (previous == 0)
                    {
                        result.Add(0);
                    }
                    else
                    {
                        double roc = ((current - previous) / previous) * 100;
                        result.Add(roc);
                    }
                }
            }
            
            return result;
        }

        /// <summary>
        /// Calculate Bollinger Bands
        /// </summary>
        private static (List<double> Upper, List<double> Middle, List<double> Lower) CalculateBollingerBands(List<double> prices, int period, double stdDevMultiplier)
        {
            var result = (Upper: new List<double>(), Middle: new List<double>(), Lower: new List<double>());
            
            // Calculate Simple Moving Average (SMA)
            var sma = CalculateSMA(prices, period);
            result.Middle = sma;
            
            // Calculate standard deviation for each window
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period - 1)
                {
                    result.Upper.Add(double.NaN);
                    result.Lower.Add(double.NaN);
                    continue;
                }
                
                // Get window of prices for calculating std dev
                var window = prices.Skip(i - period + 1).Take(period).ToList();
                var mean = sma[i];
                var stdDev = Math.Sqrt(window.Average(v => Math.Pow(v - mean, 2)));
                
                // Calculate upper and lower bands
                result.Upper.Add(mean + stdDevMultiplier * stdDev);
                result.Lower.Add(mean - stdDevMultiplier * stdDev);
            }
            
            return result;
        }

        #endregion

        /// <summary>
        /// Initializes the database, creating the file and tables if they don't exist.
        /// </summary>
        /// <remarks>
        /// This method is automatically called when needed but can be called explicitly during
        /// application startup. It creates the SQLite database file, configures performance
        /// settings (WAL mode, timeouts), and creates all required tables with proper schema.
        /// 
        /// The initialization is thread-safe and idempotent - multiple calls are safe.
        /// </remarks>
        /// <example>
        /// <code>
        /// // Explicit initialization at startup
        /// DatabaseMonolith.Initialize();
        /// </code>
        /// </example>
        public static void Initialize()
        {
            if (!initialized)
            {
                try
                {
                    // Create database file if it doesn't exist
                    if (!File.Exists(DbFilePath))
                    {
                        SQLiteConnection.CreateFile(DbFilePath);
                    }

                    using (var connection = new SQLiteConnection(ConnectionString))
                    {
                        connection.Open();
                        
                        // Configure SQLite for better concurrency and reduced locking
                        using (var command = new SQLiteCommand("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=30000;", connection))
                        {
                            command.ExecuteNonQuery();
                        }

                        // Create logging table
                        using (var command = new SQLiteCommand(
                            @"CREATE TABLE IF NOT EXISTS Logs (
                            Id INTEGER PRIMARY KEY AUTOINCREMENT,
                            Timestamp TEXT NOT NULL,
                            Level TEXT NOT NULL,
                            Message TEXT NOT NULL,
                            Details TEXT
                            )", connection))
                        {
                            command.ExecuteNonQuery();
                        }
                        EnsureLogsTableHasLevelColumn(connection); // Ensure Level column exists
                        EnsureLogsTableHasDetailsColumn(connection); // Ensure Details column exists

                        // Create stock symbols table
                        using (var command = new SQLiteCommand(
                            @"CREATE TABLE IF NOT EXISTS StockSymbols (
                            Symbol TEXT PRIMARY KEY,
                            Name TEXT,
                            Sector TEXT,
                            Industry TEXT,
                            LastUpdated DATETIME
                            )", connection))
                        {
                            command.ExecuteNonQuery();
                        }

                        // Create predictions table
                        using (var command = new SQLiteCommand(
                            @"CREATE TABLE IF NOT EXISTS StockPredictions (
                            Id INTEGER PRIMARY KEY AUTOINCREMENT,
                            Symbol TEXT NOT NULL,
                            PredictedAction TEXT NOT NULL,
                            Confidence REAL NOT NULL,
                            CurrentPrice REAL NOT NULL,
                            TargetPrice REAL NOT NULL,
                            PotentialReturn REAL NOT NULL,
                            CreatedDate DATETIME NOT NULL,
                            FOREIGN KEY(Symbol) REFERENCES StockSymbols(Symbol)
                            )", connection))
                        {
                            command.ExecuteNonQuery();
                        }

                        // Create indicators table
                        using (var command = new SQLiteCommand(
                            @"CREATE TABLE IF NOT EXISTS PredictionIndicators (
                            PredictionId INTEGER NOT NULL,
                            IndicatorName TEXT NOT NULL,
                            IndicatorValue REAL NOT NULL,
                            PRIMARY KEY(PredictionId, IndicatorName),
                            FOREIGN KEY(PredictionId) REFERENCES StockPredictions(Id)
                            )", connection))
                        {
                            command.ExecuteNonQuery();
                        }

                        // Create Settings table
                        EnsureSettingsTable(connection);

                        // Create UserAppSettings and other tables
                        EnsureUserAppSettingsTables(connection);
                    }

                    initialized = true;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error initializing database: {ex.Message}");
                    throw;
                }
            }
        }

        private static void EnsureUserAppSettingsTables(SQLiteConnection connection)
        {
            var createTableQuery = @"
                CREATE TABLE IF NOT EXISTS UserAppSettings (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    TabName TEXT NOT NULL,
                    TabOrder INTEGER NOT NULL,
                    CardPositions TEXT,
                    ControlsConfig TEXT,
                    ToolsConfig TEXT,
                    GridRows INTEGER DEFAULT 4,
                    GridColumns INTEGER DEFAULT 4,
                    DataGridConfig TEXT
                );
                CREATE TABLE IF NOT EXISTS UserCredentials (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Username TEXT NOT NULL,
                    Password TEXT NOT NULL,
                    Pin TEXT,
                    LastLoginDate DATETIME
                );
                CREATE TABLE IF NOT EXISTS TabConfigs (
                    TabName TEXT NOT NULL PRIMARY KEY,
                    ToolsConfig TEXT
                )";
            
            using (var command = new SQLiteCommand(createTableQuery, connection))
            {
                command.ExecuteNonQuery();
            }

            // Add DataGridConfig column to existing tables if it doesn't exist
            try
            {
                var checkColumnQuery = "PRAGMA table_info(UserAppSettings)";
                var columns = connection.Query(checkColumnQuery);
                var hasDataGridConfig = false;
                
                foreach (var column in columns)
                {
                    var columnDict = column as IDictionary<string, object>;
                    if (columnDict != null && columnDict.ContainsKey("name") && 
                        columnDict["name"]?.ToString() == "DataGridConfig")
                    {
                        hasDataGridConfig = true;
                        break;
                    }
                }

                if (!hasDataGridConfig)
                {
                    var addColumnQuery = "ALTER TABLE UserAppSettings ADD COLUMN DataGridConfig TEXT";
                    using (var command = new SQLiteCommand(addColumnQuery, connection))
                    {
                        command.ExecuteNonQuery();
                    }
                }
            }
            catch (Exception ex)
            {
                // Log error but don't fail initialization
                Log("Warning", "Failed to add DataGridConfig column", ex.ToString());
            }
        }

        // todo why is there a custom ExecuteNonQuery wrapper... why not use dapper?
        public static void ExecuteNonQuery(string sql, params SQLiteParameter[] parameters)
        {
            ResilienceHelper.Retry(() =>
            {
                try
                {
                    using (var connection = new SQLiteConnection(ConnectionString))
                    {
                        connection.Open();
                        using (var command = new SQLiteCommand(sql, connection))
                        {
                            if (parameters != null && parameters.Length > 0)
                            {
                                command.Parameters.AddRange(parameters);
                            }
                            command.ExecuteNonQuery();
                        }
                    }
                }
                catch (Exception ex)
                {
                    LogErrorWithContext(ex, $"Error executing SQL: {sql}");
                    throw;
                }
            }, RetryOptions.ForCriticalOperation());
        }

        public static DataTable ExecuteQuery(string sql, params SQLiteParameter[] parameters)
        {
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    using (var command = new SQLiteCommand(sql, connection))
                    {
                        if (parameters != null && parameters.Length > 0)
                        {
                            command.Parameters.AddRange(parameters);
                        }

                        using (var adapter = new SQLiteDataAdapter(command))
                        {
                            var dataTable = new DataTable();
                            adapter.Fill(dataTable);
                            return dataTable;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Error executing query: {sql}");
                throw;
            }
        }

        public static T ExecuteScalar<T>(string sql, params SQLiteParameter[] parameters)
        {
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    using (var command = new SQLiteCommand(sql, connection))
                    {
                        if (parameters != null && parameters.Length > 0)
                        {
                            command.Parameters.AddRange(parameters);
                        }
                        object result = command.ExecuteScalar();
                        return (T)Convert.ChangeType(result, typeof(T));
                    }
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Error executing scalar query: {sql}");
                throw;
            }
        }

        public static void ResetDatabase()
        {
            try
            {
                // Close all connections before deleting
                GC.Collect();
                GC.WaitForPendingFinalizers();

                // Delete database file
                if (File.Exists(DbFilePath))
                {
                    File.Delete(DbFilePath);
                }

                // Reinitialize
                initialized = false;
                Initialize();
                Log("Info", "Database has been reset successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error resetting database: {ex.Message}");
                throw;
            }
        }

        public static List<TransactionModel> LoadTransactions(string symbol)
        {
            List<TransactionModel> results = new List<TransactionModel>();
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    string query = @"
                        SELECT Date, Type, Symbol, Price, Shares, Cost, PnL 
                        FROM Transactions 
                        WHERE Symbol = @Symbol
                        ORDER BY Date DESC";

                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", symbol);
                        
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                results.Add(new TransactionModel
                                {
                                    Symbol = reader["Symbol"].ToString(),
                                    TransactionType = reader["Type"].ToString(),
                                    ExecutionPrice = Convert.ToDouble(reader["Price"]),
                                    Quantity = Convert.ToInt32(reader["Shares"]),
                                    TotalValue = Convert.ToDouble(reader["Cost"]),
                                    RealizedPnL = Convert.ToDouble(reader["PnL"]),
                                    ExecutionTime = Convert.ToDateTime(reader["Date"])
                                });
                            }
                        }
                    }

                    Log("Info", $"Loaded {results.Count} transactions for symbol {symbol}");
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to load transactions for symbol {symbol}");
            }
            return results;
        }

        #region Stock Symbol Caching Methods

        /// <summary>
        /// Caches a list of stock symbols in the database.
        /// If symbols already exist, their information will be updated.
        /// </summary>
        /// <param name="symbols">List of stock symbols to cache</param>
        public static void CacheStockSymbols(List<StockSymbol> symbols)
        {
            if (symbols == null || !symbols.Any())
            {
                Log("Warning", "Attempted to cache empty symbol list");
                return;
            }

            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    using (var transaction = connection.BeginTransaction())
                    {
                        var insertOrUpdateQuery = @"
                            INSERT INTO StockSymbols (Symbol, Name, Sector, Industry, LastUpdated)
                            VALUES (@Symbol, @Name, @Sector, @Industry, @LastUpdated)
                            ON CONFLICT(Symbol) DO UPDATE SET
                                Name = @Name,
                                Sector = @Sector,
                                Industry = @Industry,
                                LastUpdated = @LastUpdated";

                        using (var command = new SQLiteCommand(insertOrUpdateQuery, connection))
                        {
                            foreach (var symbol in symbols)
                            {
                                command.Parameters.Clear();
                                command.Parameters.AddWithValue("@Symbol", symbol.Symbol);
                                command.Parameters.AddWithValue("@Name", symbol.Name ?? string.Empty);
                                command.Parameters.AddWithValue("@Sector", symbol.Sector ?? string.Empty);
                                command.Parameters.AddWithValue("@Industry", symbol.Industry ?? string.Empty);
                                command.Parameters.AddWithValue("@LastUpdated", DateTime.Now);
                                command.ExecuteNonQuery();
                            }
                        }

                        transaction.Commit();
                    }
                    Log("Info", $"Successfully cached {symbols.Count} stock symbols");
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to cache stock symbols");
                throw;
            }
        }

        /// <summary>
        /// Retrieves all cached stock symbols from the database.
        /// </summary>
        /// <returns>ObservableCollection of StockSymbol objects</returns>
        public static ObservableCollection<StockSymbol> GetAllStockSymbols()
        {
            var symbols = new ObservableCollection<StockSymbol>();
            
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    string query = "SELECT Symbol, Name, Sector, Industry, LastUpdated FROM StockSymbols ORDER BY Symbol";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            symbols.Add(new StockSymbol
                            {
                                Symbol = reader["Symbol"].ToString(),
                                Name = reader["Name"].ToString(),
                                Sector = reader["Sector"].ToString(),
                                Industry = reader["Industry"].ToString(),
                                LastUpdated = reader["LastUpdated"] != DBNull.Value ? 
                                    Convert.ToDateTime(reader["LastUpdated"]) : DateTime.MinValue
                            });
                        }
                    }
                    Log("Info", $"Retrieved {symbols.Count} cached stock symbols");
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to retrieve stock symbols");
            }

            return symbols;
        }

        /// <summary>
        /// Retrieves a specific stock symbol from the cache.
        /// </summary>
        /// <param name="symbol">The stock symbol to retrieve</param>
        /// <returns>StockSymbol object or null if not found</returns>
        public static StockSymbol GetStockSymbol(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                Log("Warning", "Attempted to retrieve empty symbol");
                return null;
            }

            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    string query = @"
                        SELECT Symbol, Name, Sector, Industry, LastUpdated 
                        FROM StockSymbols 
                        WHERE Symbol = @Symbol";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", symbol.Trim().ToUpper());
                        
                        using (var reader = command.ExecuteReader())
                        {
                            if (reader.Read())
                            {
                                return new StockSymbol
                                {
                                    Symbol = reader["Symbol"].ToString(),
                                    Name = reader["Name"].ToString(),
                                    Sector = reader["Sector"].ToString(),
                                    Industry = reader["Industry"].ToString(),
                                    LastUpdated = reader["LastUpdated"] != DBNull.Value ? 
                                        Convert.ToDateTime(reader["LastUpdated"]) : DateTime.MinValue
                                };
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to retrieve stock symbol {symbol}");
            }

            return null;
        }

        /// <summary>
        /// Checks if the symbol cache is still valid based on the LastUpdated timestamp.
        /// </summary>
        /// <param name="maxAgeDays">Maximum age of the cache in days (default: 7)</param>
        /// <returns>True if cache is valid, false otherwise</returns>
        public static bool IsSymbolCacheValid(int maxAgeDays = 7)
        {
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    
                    // First check if we have any symbols cached at all
                    string countQuery = "SELECT COUNT(*) FROM StockSymbols";
                    using (var command = new SQLiteCommand(countQuery, connection))
                    {
                        var count = Convert.ToInt32(command.ExecuteScalar());
                        if (count == 0)
                        {
                            return false; // No symbols in cache
                        }
                    }
                    
                    // Check if the oldest entry is still valid
                    string ageQuery = "SELECT MIN(LastUpdated) FROM StockSymbols";
                    using (var command = new SQLiteCommand(ageQuery, connection))
                    {
                        var oldestUpdate = Convert.ToDateTime(command.ExecuteScalar());
                        var cacheAge = (DateTime.Now - oldestUpdate).TotalDays;
                        return cacheAge <= maxAgeDays;
                    }
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to validate symbol cache");
                return false;
            }
        }

        /// <summary>
        /// Forces a refresh of the symbol cache by updating the LastUpdated timestamp.
        /// </summary>
        /// <returns>Number of symbols refreshed</returns>
        public static int RefreshSymbolCache()
        {
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    string updateQuery = "UPDATE StockSymbols SET LastUpdated = @Now";
                    
                    using (var command = new SQLiteCommand(updateQuery, connection))
                    {
                        command.Parameters.AddWithValue("@Now", DateTime.Now);
                        var count = command.ExecuteNonQuery();
                        Log("Info", $"Refreshed timestamps for {count} stock symbols");
                        return count;
                    }
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to refresh symbol cache");
                return 0;
            }
        }

        /// <summary>
        /// Searches for stock symbols by name or ticker.
        /// </summary>
        /// <param name="searchTerm">Search term to match against symbol or name</param>
        /// <returns>List of matching StockSymbol objects</returns>
        public static List<StockSymbol> SearchSymbols(string searchTerm)
        {
            var results = new List<StockSymbol>();
            
            if (string.IsNullOrWhiteSpace(searchTerm))
            {
                return results;
            }

            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    string query = @"
                        SELECT Symbol, Name, Sector, Industry, LastUpdated 
                        FROM StockSymbols 
                        WHERE Symbol LIKE @SearchTerm OR Name LIKE @SearchTerm
                        ORDER BY Symbol
                        LIMIT 100"; // Limiting results for performance
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@SearchTerm", $"%{searchTerm}%");
                        
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                results.Add(new StockSymbol
                                {
                                    Symbol = reader["Symbol"].ToString(),
                                    Name = reader["Name"].ToString(),
                                    Sector = reader["Sector"].ToString(),
                                    Industry = reader["Industry"].ToString(),
                                    LastUpdated = reader["LastUpdated"] != DBNull.Value ? 
                                        Convert.ToDateTime(reader["LastUpdated"]) : DateTime.MinValue
                                });
                            }
                        }
                    }
                    Log("Info", $"Found {results.Count} matching stock symbols for search term '{searchTerm}'");
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to search for stock symbols with term '{searchTerm}'");
            }

            return results;
        }

        #endregion

        /// <summary>
        /// Gets the latest stock quote data for a symbol using the financialmodelingprep API.
        /// </summary>
        /// <param name="symbol">The stock symbol to fetch.</param>
        /// <returns>The latest QuoteData object, or null if not found.</returns>
        public static async Task<QuoteData> GetLatestQuoteData(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return null;

            try
            {
                // Use a static instance or create a new one as needed
                var tradingBot = new WebullTradingBot();
                return await tradingBot.FetchQuoteData(symbol);
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to fetch latest quote data for {symbol}");
                return null;
            }
        }

        /// <summary>
        /// Gets the latest stock quote data for multiple symbols.
        /// </summary>
        /// <param name="symbols">A list of stock symbols.</param>
        /// <returns>A list of QuoteData objects.</returns>
        public static async Task<List<QuoteData>> GetLatestQuoteData(IEnumerable<string> symbols)
        {
            var results = new List<QuoteData>();
            if (symbols == null)
                return results;

            var tradingBot = new WebullTradingBot();
            foreach (var symbol in symbols)
            {
                try
                {
                    var data = await tradingBot.FetchQuoteData(symbol);
                    if (data != null)
                        results.Add(data);
                }
                catch (Exception ex)
                {
                    LogErrorWithContext(ex, $"Failed to fetch latest quote data for {symbol}");
                }
            }
            return results;
        }

        /// <summary>
        /// Gets the latest stock quote data and its timestamp for a symbol.
        /// Returns (QuoteData, DateTime?) where DateTime is the LastUpdated time if available.
        /// </summary>
        public static async Task<(QuoteData, DateTime?)> GetLatestQuoteDataWithTimestamp(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return (null, null);

            try
            {
                // Try to get from cache (StockSymbols table)
                QuoteData cached = null;
                DateTime? lastUpdated = null;
                using (var connection = GetConnection())
                {
                    connection.Open();
                    var query = @"
                        SELECT Symbol, Name, '' as Sector, '' as Industry, LastUpdated
                        FROM StockSymbols
                        WHERE Symbol = @Symbol";
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", symbol.Trim().ToUpper());
                        using (var reader = command.ExecuteReader())
                        {
                            if (reader.Read())
                            {
                                cached = new QuoteData
                                {
                                    Symbol = reader["Symbol"].ToString(),
                                    Name = reader["Name"].ToString(),
                                    // Price, Change, etc. not available in StockSymbols table
                                    LastAccessed = DateTime.Now
                                };
                                if (reader["LastUpdated"] != DBNull.Value)
                                    lastUpdated = Convert.ToDateTime(reader["LastUpdated"]);
                            }
                        }
                    }
                }

                // Try to get latest from API
                var latest = await GetLatestQuoteData(symbol);
                if (latest != null)
                {
                    return (latest, latest.LastUpdated != default ? latest.LastUpdated : lastUpdated);
                }
                // Fallback to cached (if available)
                return (cached, lastUpdated);
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to get latest quote data with timestamp for {symbol}");
                return (null, null);
            }
        }

        /// <summary>
        /// Gets the latest stock data and its timestamp for a symbol and timeRange.
        /// Returns (StockData, DateTime?) where DateTime is the most recent date in the data if available.
        /// </summary>
        public static async Task<(StockData, DateTime?)> GetStockDataWithTimestamp(string symbol, string timeRange)
        {
            if (string.IsNullOrWhiteSpace(symbol) || string.IsNullOrWhiteSpace(timeRange))
                return (null, null);

            try
            {
                // Try to get from cache (if implemented elsewhere, e.g. StockDataCacheService)
                // For now, always fetch from API
                var tradingBot = new WebullTradingBot();
                var stockData = await tradingBot.FetchChartData(symbol, timeRange);

                DateTime? lastDate = null;
                if (stockData?.Dates != null && stockData.Dates.Count > 0)
                    lastDate = stockData.Dates.Last();

                return (stockData, lastDate);
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to get stock data with timestamp for {symbol} {timeRange}");
                return (null, null);
            }
        }

        /// <summary>
        /// Saves a real-time quote (QuoteData) to the StockSymbols table.
        /// Updates fields if the symbol already exists.
        /// </summary>
        public static async Task SaveQuoteData(QuoteData quote)
        {
            if (quote == null || string.IsNullOrWhiteSpace(quote.Symbol))
                return;

            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    var query = @"
                        INSERT INTO StockSymbols (Symbol, Name, LastUpdated)
                        VALUES (@Symbol, @Name, @LastUpdated)
                        ON CONFLICT(Symbol) DO UPDATE SET
                            Name = @Name,
                            LastUpdated = @LastUpdated;
                    ";
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", quote.Symbol);
                        command.Parameters.AddWithValue("@Name", quote.Name ?? string.Empty);
                        command.Parameters.AddWithValue("@LastUpdated", quote.LastUpdated == default ? DateTime.Now : quote.LastUpdated);
                        command.ExecuteNonQuery();
                    }
                }
                Log("Info", $"Saved quote data for {quote.Symbol}");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to save quote data for {quote?.Symbol}");
            }
            await Task.CompletedTask;
        }

        /// <summary>
        /// Saves historical/time-series stock data (StockData) to the StockDataCache table.
        /// Data is serialized as JSON and compressed.
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="timeRange">Time range (e.g., '1day', '1week')</param>
        /// <param name="interval">Interval (e.g., '5min', '1day')</param>
        /// <param name="data">StockData object</param>
        public static void SaveStockData(string symbol, string timeRange, string interval, StockData data)
        {
            if (string.IsNullOrWhiteSpace(symbol) || data == null)
                return;

            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    // Serialize to JSON and then compress
                    var json = JsonConvert.SerializeObject(data);
                    var compressedJson = Utilities.CompressionHelper.CompressString(json);
                    
                    var query = @"
                        INSERT INTO StockDataCache (Symbol, TimeRange, Interval, Data, CacheTime)
                        VALUES (@Symbol, @TimeRange, @Interval, @Data, @CacheTime)
                        ON CONFLICT(Symbol, TimeRange, Interval) DO UPDATE SET
                            Data = @Data,
                            CacheTime = @CacheTime;
                    ";
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", symbol.Trim().ToUpper());
                        command.Parameters.AddWithValue("@TimeRange", timeRange ?? "");
                        command.Parameters.AddWithValue("@Interval", interval ?? "");
                        command.Parameters.AddWithValue("@Data", compressedJson);
                        command.Parameters.AddWithValue("@CacheTime", DateTime.Now);
                        command.ExecuteNonQuery();
                    }
                }
                Log("Info", $"Saved compressed stock data for {symbol} [{timeRange}, {interval}]");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to save stock data for {symbol} [{timeRange}, {interval}]");
            }
        }

        public static void EnsureAlphaVantageApiUsageTable()
        {
            using (var conn = GetConnection())
            {
                conn.Open();
                var cmd = conn.CreateCommand();
                cmd.CommandText = @"
                CREATE TABLE IF NOT EXISTS AlphaVantageApiUsage (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    TimestampUtc DATETIME NOT NULL,
                    Endpoint TEXT,
                    Parameters TEXT
                )";
                cmd.ExecuteNonQuery();
            }
        }

        /// <summary>
        /// Logs Alpha Vantage API usage for rate limiting and monitoring.
        /// </summary>
        /// <param name="endpoint">API endpoint that was called</param>
        /// <param name="parameters">Parameters passed to the API call</param>
        /// <remarks>
        /// Tracks API usage to prevent rate limit violations and provide usage analytics.
        /// Alpha Vantage APIs have strict rate limits (typically 75 calls per minute for free tier).
        /// This data is used by the rate limiting system to queue and throttle API requests.
        /// 
        /// Automatically creates the AlphaVantageApiUsage table if needed.
        /// </remarks>
        /// <example>
        /// <code>
        /// // Log an API call
        /// DatabaseMonolith.LogAlphaVantageApiUsage(
        ///     "TIME_SERIES_DAILY", 
        ///     "symbol=AAPL&outputsize=compact");
        /// </code>
        /// </example>
        public static void LogAlphaVantageApiUsage(string endpoint, string parameters)
        {
            try
            {
                EnsureAlphaVantageApiUsageTable();
                using (var conn = GetConnection())
                {
                    conn.Open();
                    var cmd = conn.CreateCommand();
                    cmd.CommandText = "INSERT INTO AlphaVantageApiUsage (TimestampUtc, Endpoint, Parameters) VALUES (@ts, @ep, @par)";
                    cmd.Parameters.AddWithValue("@ts", DateTime.UtcNow);
                    cmd.Parameters.AddWithValue("@ep", endpoint ?? "");
                    cmd.Parameters.AddWithValue("@par", parameters ?? "");
                    cmd.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to log Alpha Vantage API usage");
            }
        }

        /// <summary>
        /// Gets the current API usage count for rate limiting enforcement.
        /// </summary>
        /// <param name="utcNow">Current UTC time to check usage against</param>
        /// <returns>Number of API calls made in the last 60 seconds (sliding window)</returns>
        /// <remarks>
        /// Used by the rate limiting system to enforce Alpha Vantage API limits.
        /// Counts calls within a 60-second sliding window ending at the current time.
        /// The free tier typically allows 75 calls per minute, premium tiers have higher limits.
        /// </remarks>
        /// <example>
        /// <code>
        /// int currentUsage = DatabaseMonolith.GetAlphaVantageApiUsageCount(DateTime.UtcNow);
        /// if (currentUsage >= 75) 
        /// {
        ///     // Wait before making next API call
        ///     await Task.Delay(60000);
        /// }
        /// </code>
        /// </example>
        public static int GetAlphaVantageApiUsageCount(DateTime utcNow)
        {
            EnsureAlphaVantageApiUsageTable();
            using (var conn = GetConnection())
            {
                conn.Open();
                var cmd = conn.CreateCommand();
                // Count calls in the last 60 seconds (true sliding window)
                var windowStart = utcNow.AddSeconds(-60);
                cmd.CommandText = "SELECT COUNT(*) FROM AlphaVantageApiUsage WHERE TimestampUtc >= @start AND TimestampUtc <= @end";
                cmd.Parameters.AddWithValue("@start", windowStart);
                cmd.Parameters.AddWithValue("@end", utcNow);
                return Convert.ToInt32(cmd.ExecuteScalar());
            }
        }

        /// <summary>
        /// Deletes all error logs older than 7 days from the Logs table.
        /// </summary>
        public static void DeleteOldErrors()
        {
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    EnsureLogsTableHasLevelColumn(connection); // Ensure Level column exists
                    var deleteQuery = @"DELETE FROM Logs WHERE Level = 'Error' AND Timestamp < @ThresholdDate";
                    using (var command = new SQLiteCommand(deleteQuery, connection))
                    {
                        command.Parameters.AddWithValue("@ThresholdDate", DateTime.Now.AddDays(-7));
                        command.ExecuteNonQuery();
                    }
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to delete old error logs");
            }
        }

        /// <summary>
        /// Loads the user settings from the database or settings profile.
        /// </summary>
        /// <returns>UserSettings object with current settings.</returns>
        public static UserSettings LoadUserSettings()
        {
            // For now, just use GetUserSettings (profile-based)
            return GetUserSettings();
        }

        private static readonly string StockCacheKey = "StockCache";
        private static readonly string VolatileStocksCacheKey = "VolatileStocks";
        private static readonly string AnalystRatingHistoryKey = "AnalystRatingsHistory";
        private static readonly string ConsensusHistoryKey = "ConsensusHistory";

        public static void CacheSymbols(List<string> symbols)
        {
            if (symbols == null || !symbols.Any())
                return;

            try
            {
                // Store in UserPreferences table with timestamp
                string symbolsJson = JsonConvert.SerializeObject(symbols);
                SaveUserPreference(StockCacheKey, symbolsJson);
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to cache symbols");
            }
        }

        public static List<string> GetCachedSymbols()
        {
            try
            {
                var cachedJson = GetUserPreference(StockCacheKey);
                if (!string.IsNullOrEmpty(cachedJson))
                {
                    return JsonConvert.DeserializeObject<List<string>>(cachedJson) ?? new List<string>();
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to get cached symbols");
            }
            return new List<string>();
        }

        /// <summary>
        /// Saves analyst ratings data for a specific stock symbol.
        /// </summary>
        /// <param name="symbol">Stock symbol (e.g., "AAPL")</param>
        /// <param name="ratings">List of analyst ratings to save</param>
        /// <remarks>
        /// Persists analyst rating data including rating changes, price targets, and analyst information.
        /// Supports upsert operations - existing ratings for the same analyst and date are updated.
        /// Creates the AnalystRatings table and indexes if they don't exist.
        /// 
        /// This data is used for sentiment analysis and decision support in trading algorithms.
        /// </remarks>
        /// <example>
        /// <code>
        /// var ratings = new List&lt;AnalystRating&gt;
        /// {
        ///     new AnalystRating
        ///     {
        ///         AnalystName = "Goldman Sachs",
        ///         Rating = "BUY",
        ///         PriceTarget = 180.00,
        ///         RatingDate = DateTime.Now,
        ///         ChangeType = RatingChangeType.Upgrade
        ///     }
        /// };
        /// DatabaseMonolith.SaveAnalystRatings("AAPL", ratings);
        /// </code>
        /// </example>
        public static void SaveAnalystRatings(string symbol, List<AnalystRating> ratings)
        {
            if (string.IsNullOrWhiteSpace(symbol) || ratings == null || !ratings.Any())
                return;

            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    
                    // Ensure the table exists
                    EnsureAnalystRatingsTable(connection);
                    
                    // Insert each rating
                    foreach (var rating in ratings)
                    {
                        var query = @"
                            INSERT INTO AnalystRatings (
                                Symbol, AnalystName, Rating, PreviousRating, 
                                PriceTarget, PreviousPriceTarget, RatingDate, ChangeType
                            )
                            VALUES (
                                @Symbol, @AnalystName, @Rating, @PreviousRating,
                                @PriceTarget, @PreviousPriceTarget, @RatingDate, @ChangeType
                            )
                            ON CONFLICT(Symbol, AnalystName, RatingDate) DO UPDATE SET
                                Rating = excluded.Rating,
                                PreviousRating = excluded.PreviousRating,
                                PriceTarget = excluded.PriceTarget,
                                PreviousPriceTarget = excluded.PreviousPriceTarget,
                                ChangeType = excluded.ChangeType
                        ";
                        
                        using (var command = new SQLiteCommand(query, connection))
                        {
                            command.Parameters.AddWithValue("@Symbol", symbol);
                            command.Parameters.AddWithValue("@AnalystName", rating.AnalystName);
                            command.Parameters.AddWithValue("@Rating", rating.Rating);
                            command.Parameters.AddWithValue("@PreviousRating", rating.PreviousRating ?? (object)DBNull.Value);
                            command.Parameters.AddWithValue("@PriceTarget", rating.PriceTarget);
                            command.Parameters.AddWithValue("@PreviousPriceTarget", rating.PreviousPriceTarget);
                            command.Parameters.AddWithValue("@RatingDate", rating.RatingDate);
                            command.Parameters.AddWithValue("@ChangeType", rating.ChangeType.ToString());
                            command.ExecuteNonQuery();
                        }
                    }
                    
                    Log("Info", $"Saved {ratings.Count} analyst ratings for {symbol}");
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to save analyst ratings for {symbol}");
            }
        }
        
        /// <summary>
        /// Ensures the AnalystRatings table exists in the database
        /// </summary>
        private static void EnsureAnalystRatingsTable(SQLiteConnection connection)
        {
            var createTableQuery = @"
                CREATE TABLE IF NOT EXISTS AnalystRatings (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Symbol TEXT NOT NULL,
                    AnalystName TEXT NOT NULL,
                    Rating TEXT NOT NULL,
                    PreviousRating TEXT,
                    PriceTarget REAL NOT NULL,
                    PreviousPriceTarget REAL,
                    RatingDate TEXT NOT NULL,
                    ChangeType TEXT NOT NULL,
                    UNIQUE(Symbol, AnalystName, RatingDate)
                )
            ";
            
            using (var command = new SQLiteCommand(createTableQuery, connection))
            {
                command.ExecuteNonQuery();
            }
            
            // Create an index for faster queries
            var createIndexQuery = @"
                CREATE INDEX IF NOT EXISTS IDX_AnalystRatings_Symbol_Date 
                ON AnalystRatings(Symbol, RatingDate)
            ";
            
            using (var command = new SQLiteCommand(createIndexQuery, connection))
            {
                command.ExecuteNonQuery();
            }
        }
        
        /// <summary>
        /// Retrieves analyst ratings for a specific symbol since a given date.
        /// </summary>
        /// <param name="symbol">Stock symbol to retrieve ratings for</param>
        /// <param name="since">Minimum date for ratings to include</param>
        /// <returns>List of analyst ratings ordered by date (most recent first)</returns>
        /// <remarks>
        /// Loads historical analyst rating data for analysis and decision making.
        /// Results are ordered by rating date in descending order (newest first).
        /// Creates the table if it doesn't exist to prevent runtime errors.
        /// </remarks>
        /// <example>
        /// <code>
        /// var ratings = DatabaseMonolith.GetAnalystRatings("AAPL", DateTime.Now.AddDays(-30));
        /// foreach (var rating in ratings)
        /// {
        ///     Console.WriteLine($"{rating.AnalystName}: {rating.Rating} - ${rating.PriceTarget}");
        /// }
        /// </code>
        /// </example>
        public static List<AnalystRating> GetAnalystRatings(string symbol, DateTime since)
        {
            var ratings = new List<AnalystRating>();
            
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    
                    // Ensure the table exists
                    EnsureAnalystRatingsTable(connection);
                    
                    var query = @"
                        SELECT 
                            Id, Symbol, AnalystName, Rating, PreviousRating,
                            PriceTarget, PreviousPriceTarget, RatingDate, ChangeType
                        FROM AnalystRatings
                        WHERE Symbol = @Symbol AND datetime(RatingDate) >= datetime(@Since)
                        ORDER BY datetime(RatingDate) DESC
                    ";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", symbol);
                        command.Parameters.AddWithValue("@Since", since.ToString("yyyy-MM-dd HH:mm:ss"));
                        
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                var rating = new AnalystRating
                                {
                                    Id = Convert.ToInt32(reader["Id"]),
                                    Symbol = reader["Symbol"].ToString(),
                                    AnalystName = reader["AnalystName"].ToString(),
                                    Rating = reader["Rating"].ToString(),
                                    PreviousRating = reader["PreviousRating"] == DBNull.Value ? null : reader["PreviousRating"].ToString(),
                                    PriceTarget = Convert.ToDouble(reader["PriceTarget"]),
                                    PreviousPriceTarget = reader["PreviousPriceTarget"] == DBNull.Value ? 0 : Convert.ToDouble(reader["PreviousPriceTarget"]),
                                    RatingDate = DateTime.Parse(reader["RatingDate"].ToString()),
                                    ChangeType = (RatingChangeType)Enum.Parse(typeof(RatingChangeType), reader["ChangeType"].ToString())
                                };
                                
                                ratings.Add(rating);
                            }
                        }
                    }
                }
                
                Log("Info", $"Retrieved {ratings.Count} analyst ratings for {symbol} since {since.ToString("yyyy-MM-dd")}");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to retrieve analyst ratings for {symbol}");
            }
            
            return ratings;
        }
        
        /// <summary>
        /// Saves analyst consensus history to the database
        /// </summary>
        public static void SaveConsensusHistory(AnalystRatingAggregate consensus) 
        {
            if (consensus == null || string.IsNullOrWhiteSpace(consensus.Symbol))
                return;
                
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    
                    // Ensure the table exists
                    EnsureConsensusHistoryTable(connection);
                    
                    var query = @"
                        INSERT INTO ConsensusHistory (
                            Symbol, ConsensusRating, ConsensusScore, BuyCount, HoldCount, 
                            SellCount, UpgradeCount, DowngradeCount, AveragePriceTarget,
                            ConsensusTrend, RatingsStrengthIndex, SnapshotDate
                        )
                        VALUES (
                            @Symbol, @ConsensusRating, @ConsensusScore, @BuyCount, @HoldCount,
                            @SellCount, @UpgradeCount, @DowngradeCount, @AveragePriceTarget,
                            @ConsensusTrend, @RatingsStrengthIndex, @SnapshotDate
                        )
                    ";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", consensus.Symbol);
                        command.Parameters.AddWithValue("@ConsensusRating", consensus.ConsensusRating);
                        command.Parameters.AddWithValue("@ConsensusScore", consensus.ConsensusScore);
                        command.Parameters.AddWithValue("@BuyCount", consensus.BuyCount);
                        command.Parameters.AddWithValue("@HoldCount", consensus.HoldCount);
                        command.Parameters.AddWithValue("@SellCount", consensus.SellCount);
                        command.Parameters.AddWithValue("@UpgradeCount", consensus.UpgradeCount);
                        command.Parameters.AddWithValue("@DowngradeCount", consensus.DowngradeCount);
                        command.Parameters.AddWithValue("@AveragePriceTarget", consensus.AveragePriceTarget);
                        command.Parameters.AddWithValue("@ConsensusTrend", consensus.ConsensusTrend);
                        command.Parameters.AddWithValue("@RatingsStrengthIndex", consensus.RatingsStrengthIndex);
                        command.Parameters.AddWithValue("@SnapshotDate", DateTime.Now);
                        command.ExecuteNonQuery();
                    }
                    
                    Log("Info", $"Saved consensus history for {consensus.Symbol}");
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to save consensus history for {consensus.Symbol}");
            }
        }
        
        /// <summary>
        /// Ensures the ConsensusHistory table exists in the database
        /// </summary>
        private static void EnsureConsensusHistoryTable(SQLiteConnection connection)
        {
            var createTableQuery = @"
                CREATE TABLE IF NOT EXISTS ConsensusHistory (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Symbol TEXT NOT NULL,
                    ConsensusRating TEXT NOT NULL,
                    ConsensusScore REAL NOT NULL,
                    BuyCount INTEGER NOT NULL,
                    HoldCount INTEGER NOT NULL,
                    SellCount INTEGER NOT NULL,
                    UpgradeCount INTEGER NOT NULL,
                    DowngradeCount INTEGER NOT NULL,
                    AveragePriceTarget REAL NOT NULL,
                    ConsensusTrend TEXT NOT NULL,
                    RatingsStrengthIndex REAL NOT NULL,
                    SnapshotDate TEXT NOT NULL
                )
            ";
            
            using (var command = new SQLiteCommand(createTableQuery, connection))
            {
                command.ExecuteNonQuery();
            }
            
            // Create an index for faster queries
            var createIndexQuery = @"
                CREATE INDEX IF NOT EXISTS IDX_ConsensusHistory_Symbol_Date 
                ON ConsensusHistory(Symbol, SnapshotDate)
            ";
            
            using (var command = new SQLiteCommand(createIndexQuery, connection))
            {
                command.ExecuteNonQuery();
            }
        }
        
        /// <summary>
        /// Gets historical consensus data for trend analysis
        /// </summary>
        public static List<AnalystRatingAggregate> GetConsensusHistory(string symbol, int daysBack = 90)
        {
            var result = new List<AnalystRatingAggregate>();
            
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    
                    // Ensure the table exists
                    EnsureConsensusHistoryTable(connection);
                    
                    var query = @"
                        SELECT 
                            Id, Symbol, ConsensusRating, ConsensusScore, BuyCount, 
                            HoldCount, SellCount, UpgradeCount, DowngradeCount, 
                            AveragePriceTarget, ConsensusTrend, RatingsStrengthIndex, SnapshotDate
                        FROM ConsensusHistory
                        WHERE Symbol = @Symbol 
                        AND datetime(SnapshotDate) >= datetime(@Since)
                        ORDER BY datetime(SnapshotDate) DESC
                    ";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Symbol", symbol);
                        command.Parameters.AddWithValue("@Since", DateTime.Now.AddDays(-daysBack).ToString("yyyy-MM-dd HH:mm:ss"));
                        
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                var consensus = new AnalystRatingAggregate
                                {
                                    Id = Convert.ToInt32(reader["Id"]),
                                    Symbol = reader["Symbol"].ToString(),
                                    ConsensusRating = reader["ConsensusRating"].ToString(),
                                    ConsensusScore = Convert.ToDouble(reader["ConsensusScore"]),
                                    BuyCount = Convert.ToInt32(reader["BuyCount"]),
                                    HoldCount = Convert.ToInt32(reader["HoldCount"]),
                                    SellCount = Convert.ToInt32(reader["SellCount"]),
                                    UpgradeCount = Convert.ToInt32(reader["UpgradeCount"]),
                                    DowngradeCount = Convert.ToInt32(reader["DowngradeCount"]),
                                    AveragePriceTarget = Convert.ToDouble(reader["AveragePriceTarget"]),
                                    ConsensusTrend = reader["ConsensusTrend"].ToString(),
                                    RatingsStrengthIndex = Convert.ToDouble(reader["RatingsStrengthIndex"]),
                                    LastUpdated = DateTime.Parse(reader["SnapshotDate"].ToString())
                                };
                                
                                result.Add(consensus);
                            }
                        }
                    }
                }
                
                Log("Info", $"Retrieved {result.Count} consensus history records for {symbol}");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to retrieve consensus history for {symbol}");
            }
            
            return result;
        }

        public static void CacheVolatileStocks(List<string> volatileStocks)
        {
            if (volatileStocks == null || !volatileStocks.Any())
                return;

            try
            {
                // Store in UserPreferences table with timestamp
                string stocksJson = JsonConvert.SerializeObject(volatileStocks);
                SaveUserPreference(VolatileStocksCacheKey, stocksJson);
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to cache volatile stocks");
            }
        }

        private static readonly string _connectionString = "Data Source=(localdb)\\MSSQLLocalDB;Initial Catalog=Quantra;Integrated Security=True";

        /// <summary>
        /// Retrieves active trading rules, optionally filtered by symbol.
        /// </summary>
        /// <param name="symbol">Optional stock symbol to filter rules (null for all active rules)</param>
        /// <returns>List of active trading rules matching the criteria</returns>
        /// <remarks>
        /// Loads trading rules for the automated trading engine to evaluate.
        /// Only returns rules marked as active (IsActive = true).
        /// Rules are ordered by name for consistent processing.
        /// 
        /// The automated trading system calls this method to get rules to evaluate
        /// against current market conditions.
        /// </remarks>
        /// <example>
        /// <code>
        /// // Get all active rules
        /// var allRules = DatabaseMonolith.GetTradingRules();
        /// 
        /// // Get rules for specific symbol
        /// var appleRules = DatabaseMonolith.GetTradingRules("AAPL");
        /// 
        /// foreach (var rule in appleRules)
        /// {
        ///     // Evaluate rule conditions against current market data
        /// }
        /// </code>
        /// </example>
        public static List<TradingRule> GetTradingRules(string symbol = null)
        {
            try
            {
                using var connection = GetConnection();
                connection.Open();
                string sql = @"
            SELECT * FROM TradingRules 
            WHERE (@Symbol IS NULL OR Symbol = @Symbol)
            AND IsActive = 1 
            ORDER BY Name";

                using var command = connection.CreateCommand();
                command.CommandText = sql;
                command.Parameters.AddWithValue("@Symbol", (object)symbol ?? DBNull.Value);

                var rules = new List<TradingRule>();
                using var reader = command.ExecuteReader();
                while (reader.Read())
                {
                    // Read Conditions as string, then parse to List<string>
                    string conditionsRaw = reader["Conditions"]?.ToString() ?? string.Empty;
                    List<string> conditionsList;
                    try
                    {
                        // Try JSON first
                        conditionsList = Newtonsoft.Json.JsonConvert.DeserializeObject<List<string>>(conditionsRaw);
                        if (conditionsList == null)
                        {
                            // fallback to comma split
                            conditionsList = conditionsRaw.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Select(s => s.Trim()).ToList();
                        }
                    }
                    catch
                    {
                        // fallback to comma split
                        conditionsList = conditionsRaw.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Select(s => s.Trim()).ToList();
                    }

                    rules.Add(new TradingRule
                    {
                        Id = reader.GetInt32(reader.GetOrdinal("Id")),
                        Name = reader.GetString(reader.GetOrdinal("Name")),
                        Symbol = reader.GetString(reader.GetOrdinal("Symbol")),
                        OrderType = reader.GetString(reader.GetOrdinal("OrderType")),
                        IsActive = reader.GetBoolean(reader.GetOrdinal("IsActive")),
                        Conditions = conditionsList,
                        CreatedDate = reader.GetDateTime(reader.GetOrdinal("CreatedDate")),
                        LastModified = reader.GetDateTime(reader.GetOrdinal("LastModified"))
                    });
                }
                return rules;
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to get trading rules");
                throw;
            }
        }

        /// <summary>
        /// Saves trading rules to the database for automated trading execution.
        /// </summary>
        /// <param name="rule">TradingRule object containing rule definition</param>
        /// <remarks>
        /// Persists trading rule configurations for the automated trading system.
        /// Supports both insert and update operations based on rule ID.
        /// Trading rules define conditions and actions for automated trade execution.
        /// 
        /// Rules include conditions (technical indicators, price levels), order types,
        /// symbols, and activation status. The automated trading engine uses these
        /// rules to make trading decisions.
        /// </remarks>
        /// <example>
        /// <code>
        /// var rule = new TradingRule
        /// {
        ///     Name = "AAPL RSI Oversold",
        ///     Symbol = "AAPL",
        ///     OrderType = "BUY",
        ///     IsActive = true,
        ///     Conditions = new List&lt;string&gt; { "RSI < 30", "Volume > AvgVolume * 1.5" }
        /// };
        /// DatabaseMonolith.SaveTradingRule(rule);
        /// </code>
        /// </example>
        public static void SaveTradingRule(TradingRule rule)
        {
            try
            {
                using var connection = GetConnection();
                connection.Open();
                string sql;
            
                if (rule.Id == 0)
                {
                    sql = @"
                INSERT INTO TradingRules (Name, Symbol, OrderType, IsActive, Conditions, CreatedDate, LastModified)
                VALUES (@Name, @Symbol, @OrderType, @IsActive, @Conditions, @CreatedDate, @LastModified);
                SELECT SCOPE_IDENTITY();";
                }
                else
                {
                    sql = @"
                UPDATE TradingRules 
                SET Name = @Name,
                    Symbol = @Symbol,
                    OrderType = @OrderType,
                    IsActive = @IsActive,
                    Conditions = @Conditions,
                    LastModified = @LastModified
                WHERE Id = @Id";
                }

                using var command = connection.CreateCommand();
                command.CommandText = sql;
                command.Parameters.AddWithValue("@Name", rule.Name);
                command.Parameters.AddWithValue("@Symbol", rule.Symbol);
                command.Parameters.AddWithValue("@OrderType", rule.OrderType);
                command.Parameters.AddWithValue("@IsActive", rule.IsActive);
                command.Parameters.AddWithValue("@Conditions", rule.Conditions);
                command.Parameters.AddWithValue("@LastModified", DateTime.Now);

                if (rule.Id == 0)
                {
                    command.Parameters.AddWithValue("@CreatedDate", DateTime.Now);
                    rule.Id = Convert.ToInt32(command.ExecuteScalar());
                }
                else
                {
                    command.Parameters.AddWithValue("@Id", rule.Id);
                    command.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to save trading rule");
                throw;
            }
        }

        public static void DeleteRule(int ruleId)
        {
            try
            {
                using var connection = GetConnection();
                connection.Open();
                string sql = "DELETE FROM TradingRules WHERE Id = @Id";
                using var command = connection.CreateCommand();
                command.CommandText = sql;
                command.Parameters.AddWithValue("@Id", ruleId);
                command.ExecuteNonQuery();
                Log("Info", $"Deleted trading rule with Id {ruleId}");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to delete trading rule with Id {ruleId}");
                throw;
            }
        }

        public static List<string> GetCachedVolatileStocks()
        {
            try
            {
                var cachedJson = GetUserPreference(VolatileStocksCacheKey);
                if (!string.IsNullOrEmpty(cachedJson))
                {
                    return JsonConvert.DeserializeObject<List<string>>(cachedJson) ?? new List<string>();
                }
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, "Failed to get cached volatile stocks");
            }
            return new List<string>();
        }

        /// <summary>
        /// Saves a record of a completed trade for analysis and tracking.
        /// </summary>
        /// <param name="trade">TradeRecord containing execution details</param>
        /// <remarks>
        /// Persists detailed trade execution records for performance analysis and compliance.
        /// Creates the TradeRecords table if it doesn't exist.
        /// 
        /// Trade records include execution details, prices, targets, confidence levels,
        /// and notes for comprehensive trade tracking and analysis.
        /// </remarks>
        /// <example>
        /// <code>
        /// var trade = new TradeRecord
        /// {
        ///     Symbol = "TSLA",
        ///     Action = "BUY",
        ///     Price = 250.00,
        ///     TargetPrice = 275.00,
        ///     Confidence = 0.85,
        ///     ExecutionTime = DateTime.Now,
        ///     Status = "EXECUTED",
        ///     Notes = "Technical breakout signal"
        /// };
        /// DatabaseMonolith.SaveTradeRecord(trade);
        /// </code>
        /// </example>
        public static void SaveTradeRecord(TradeRecord trade)
        {
            if (trade == null)
            {
                Log("Error", "Cannot save null TradeRecord");
                return;
            }

            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();

                    // Ensure the TradeRecords table exists
                    var createTableQuery = @"
                        CREATE TABLE IF NOT EXISTS TradeRecords (
                            Id INTEGER PRIMARY KEY AUTOINCREMENT,
                            Symbol TEXT NOT NULL,
                            Action TEXT NOT NULL,
                            Price REAL NOT NULL,
                            TargetPrice REAL NOT NULL,
                            Confidence REAL DEFAULT 0,
                            ExecutionTime DATETIME NOT NULL,
                            Status TEXT NOT NULL,
                            Notes TEXT
                        )";
                    using (var createCmd = new SQLiteCommand(createTableQuery, connection))
                    {
                        createCmd.ExecuteNonQuery();
                    }

                    // Insert the trade record
                    var insertQuery = @"
                        INSERT INTO TradeRecords (Symbol, Action, Price, TargetPrice, Confidence, ExecutionTime, Status, Notes)
                        VALUES (@Symbol, @Action, @Price, @TargetPrice, @Confidence, @ExecutionTime, @Status, @Notes)
                    ";
                    using (var insertCmd = new SQLiteCommand(insertQuery, connection))
                    {
                        insertCmd.Parameters.AddWithValue("@Symbol", trade.Symbol);
                        insertCmd.Parameters.AddWithValue("@Action", trade.Action);
                        insertCmd.Parameters.AddWithValue("@Price", trade.Price);
                        insertCmd.Parameters.AddWithValue("@TargetPrice", trade.TargetPrice);
                        insertCmd.Parameters.AddWithValue("@Confidence", trade.Confidence);
                        insertCmd.Parameters.AddWithValue("@ExecutionTime", trade.ExecutionTime);
                        insertCmd.Parameters.AddWithValue("@Status", (object)trade.Status ?? DBNull.Value);
                        insertCmd.Parameters.AddWithValue("@Notes", (object)trade.Notes ?? DBNull.Value);
                        insertCmd.ExecuteNonQuery();
                    }
                }
                Log("Info", $"TradeRecord saved: {trade.Symbol} {trade.Action} @ {trade.Price:C}");
            }
            catch (Exception ex)
            {
                LogErrorWithContext(ex, $"Failed to save TradeRecord for {trade?.Symbol}");
            }
        }

        // Ensures the Logs table has a Details column (migrates if needed)
        private static void EnsureLogsTableHasDetailsColumn(SQLiteConnection connection)
        {
            // Check if Details column exists
            var columnCheckQuery = "PRAGMA table_info(Logs)";
            bool hasDetailsColumn = false;
            using (var checkCommand = new SQLiteCommand(columnCheckQuery, connection))
            using (var reader = checkCommand.ExecuteReader())
            {
                while (reader.Read())
                {
                    if (reader["name"].ToString() == "Details")
                    {
                        hasDetailsColumn = true;
                        break;
                    }
                }
            }
            if (!hasDetailsColumn)
            {
                // Add Details column if it doesn't exist
                var addColumnQuery = "ALTER TABLE Logs ADD COLUMN Details TEXT";
                using (var alterCommand = new SQLiteCommand(addColumnQuery, connection))
                {
                    alterCommand.ExecuteNonQuery();
                    Log("Info", "Added Details column to Logs table");
                }
            }
        }

        // Ensures the Logs table has a Level column (migrates from LogLevel if needed)
        private static void EnsureLogsTableHasLevelColumn(SQLiteConnection connection)
        {
            // Check what columns exist
            var columnCheckQuery = "PRAGMA table_info(Logs)";
            bool hasLevelColumn = false;
            bool hasLogLevelColumn = false;
            
            using (var checkCommand = new SQLiteCommand(columnCheckQuery, connection))
            using (var reader = checkCommand.ExecuteReader())
            {
                while (reader.Read())
                {
                    string columnName = reader["name"].ToString();
                    if (columnName == "Level")
                    {
                        hasLevelColumn = true;
                    }
                    else if (columnName == "LogLevel")
                    {
                        hasLogLevelColumn = true;
                    }
                }
            }
            
            // If we have LogLevel but not Level, we need to rename the column
            if (hasLogLevelColumn && !hasLevelColumn)
            {
                // Check if Details column exists in the original table
                bool hasDetailsColumn = false;
                using (var detailsCheckCommand = new SQLiteCommand(columnCheckQuery, connection))
                using (var reader = detailsCheckCommand.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        if (reader["name"].ToString() == "Details")
                        {
                            hasDetailsColumn = true;
                            break;
                        }
                    }
                }
                
                // SQLite doesn't support RENAME COLUMN directly, so we need to recreate the table
                // First backup the data
                string backupQuery;
                if (hasDetailsColumn)
                {
                    backupQuery = "CREATE TEMPORARY TABLE Logs_backup AS SELECT Id, Timestamp, LogLevel as Level, Message, Details FROM Logs";
                }
                else
                {
                    backupQuery = "CREATE TEMPORARY TABLE Logs_backup AS SELECT Id, Timestamp, LogLevel as Level, Message, NULL as Details FROM Logs";
                }
                
                using (var backupCommand = new SQLiteCommand(backupQuery, connection))
                {
                    backupCommand.ExecuteNonQuery();
                }
                
                // Drop the original table
                using (var dropCommand = new SQLiteCommand("DROP TABLE Logs", connection))
                {
                    dropCommand.ExecuteNonQuery();
                }
                
                // Recreate the table with correct column names
                var createQuery = @"CREATE TABLE Logs (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Timestamp TEXT NOT NULL,
                    Level TEXT NOT NULL,
                    Message TEXT NOT NULL,
                    Details TEXT
                )";
                
                using (var createCommand = new SQLiteCommand(createQuery, connection))
                {
                    createCommand.ExecuteNonQuery();
                }
                
                // Restore the data
                var restoreQuery = "INSERT INTO Logs SELECT * FROM Logs_backup";
                using (var restoreCommand = new SQLiteCommand(restoreQuery, connection))
                {
                    restoreCommand.ExecuteNonQuery();
                }
                
                // Drop the backup table
                using (var cleanupCommand = new SQLiteCommand("DROP TABLE Logs_backup", connection))
                {
                    cleanupCommand.ExecuteNonQuery();
                }
                
                Console.WriteLine("Migrated LogLevel column to Level in Logs table");
            }
            else if (!hasLevelColumn && !hasLogLevelColumn)
            {
                // Neither column exists, add Level column
                var addColumnQuery = "ALTER TABLE Logs ADD COLUMN Level TEXT NOT NULL DEFAULT 'Info'";
                using (var alterCommand = new SQLiteCommand(addColumnQuery, connection))
                {
                    alterCommand.ExecuteNonQuery();
                    Console.WriteLine("Added Level column to Logs table");
                }
            }
        }

        /// <summary>
        /// Deletes a stock symbol entirely from the database, including both StockSymbols and StockDataCache tables.
        /// </summary>
        /// <param name="symbol">The stock symbol to delete</param>
        /// <returns>True if the symbol was deleted successfully, false otherwise</returns>
        public static bool DeleteStockSymbol(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                Log("Warning", "DeleteStockSymbol called with null or empty symbol");
                return false;
            }

            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    using (var transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            // Delete from StockDataCache table first (due to potential foreign key constraints)
                            var deleteCacheQuery = "DELETE FROM StockDataCache WHERE Symbol = @Symbol";
                            using (var cacheCommand = new SQLiteCommand(deleteCacheQuery, connection))
                            {
                                cacheCommand.Parameters.AddWithValue("@Symbol", symbol);
                                var cacheDeleted = cacheCommand.ExecuteNonQuery();
                                Log("Info", $"Deleted {cacheDeleted} cache entries for symbol {symbol}");
                            }

                            // Delete from StockSymbols table
                            var deleteSymbolQuery = "DELETE FROM StockSymbols WHERE Symbol = @Symbol";
                            using (var symbolCommand = new SQLiteCommand(deleteSymbolQuery, connection))
                            {
                                symbolCommand.Parameters.AddWithValue("@Symbol", symbol);
                                var symbolDeleted = symbolCommand.ExecuteNonQuery();
                                
                                if (symbolDeleted > 0)
                                {
                                    transaction.Commit();
                                    Log("Info", $"Successfully deleted stock symbol {symbol} from database");
                                    return true;
                                }
                                else
                                {
                                    transaction.Rollback();
                                    Log("Warning", $"Stock symbol {symbol} was not found in database");
                                    return false;
                                }
                            }
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
                LogErrorWithContext(ex, $"Failed to delete stock symbol {symbol}");
                return false;
            }
        }

        #region Fundamental Data Caching

        /// <summary>
        /// Ensures the fundamental data cache table exists in the database
        /// </summary>
        private static void EnsureFundamentalCacheTableExists()
        {
            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    var command = new SQLiteCommand(
                        @"CREATE TABLE IF NOT EXISTS FundamentalDataCache (
                            Symbol TEXT NOT NULL,
                            DataType TEXT NOT NULL,
                            Value REAL,
                            CacheTime DATETIME NOT NULL,
                            PRIMARY KEY (Symbol, DataType)
                        )", connection);
                    command.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                Log("Error", "Failed to create FundamentalDataCache table", ex.ToString());
            }
        }

        /// <summary>
        /// Caches a fundamental data value for a stock symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="dataType">Type of fundamental data (e.g., "PE_RATIO")</param>
        /// <param name="value">The value to cache</param>
        public static void CacheFundamentalData(string symbol, string dataType, double? value)
        {
            if (string.IsNullOrWhiteSpace(symbol) || string.IsNullOrWhiteSpace(dataType))
                return;

            EnsureFundamentalCacheTableExists();

            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    var command = new SQLiteCommand(
                        @"INSERT OR REPLACE INTO FundamentalDataCache 
                          (Symbol, DataType, Value, CacheTime) 
                          VALUES (@Symbol, @DataType, @Value, @CacheTime)", connection);
                    
                    command.Parameters.AddWithValue("@Symbol", symbol.ToUpper());
                    command.Parameters.AddWithValue("@DataType", dataType);
                    command.Parameters.AddWithValue("@Value", value.HasValue ? (object)value.Value : DBNull.Value);
                    command.Parameters.AddWithValue("@CacheTime", DateTime.Now);
                    
                    command.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                Log("Error", $"Failed to cache fundamental data for {symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Gets cached fundamental data for a stock symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="dataType">Type of fundamental data (e.g., "PE_RATIO")</param>
        /// <param name="maxAgeHours">Maximum age of cached data in hours (default: 24)</param>
        /// <returns>Cached value or null if not found or expired</returns>
        public static double? GetCachedFundamentalData(string symbol, string dataType, int maxAgeHours = 24)
        {
            if (string.IsNullOrWhiteSpace(symbol) || string.IsNullOrWhiteSpace(dataType))
                return null;

            EnsureFundamentalCacheTableExists();

            try
            {
                using (var connection = GetConnection())
                {
                    connection.Open();
                    var command = new SQLiteCommand(
                        @"SELECT Value FROM FundamentalDataCache 
                          WHERE Symbol = @Symbol AND DataType = @DataType 
                          AND CacheTime > @ExpiryTime", connection);
                    
                    command.Parameters.AddWithValue("@Symbol", symbol.ToUpper());
                    command.Parameters.AddWithValue("@DataType", dataType);
                    command.Parameters.AddWithValue("@ExpiryTime", DateTime.Now.AddHours(-maxAgeHours));
                    
                    var result = command.ExecuteScalar();
                    if (result != null && result != DBNull.Value)
                    {
                        return Convert.ToDouble(result);
                    }
                }
            }
            catch (Exception ex)
            {
                Log("Error", $"Failed to retrieve cached fundamental data for {symbol}", ex.ToString());
            }

            return null;
        }

        #endregion

        /// <summary>
        /// Logs an error with automatic file and method context using reflection/stack trace.
        /// </summary>
        /// <param name="ex">The exception to log</param>
        /// <param name="message">A short message describing the error</param>
        /// <param name="details">Optional additional details</param>
        public static void LogErrorWithContext(Exception ex, string message = null, string details = null)
        {
            var stack = new System.Diagnostics.StackTrace(1, true);
            var frame = stack.GetFrame(0);
            var method = frame?.GetMethod();
            string file = method?.DeclaringType?.Name ?? "UnknownFile";
            string methodName = method?.Name ?? "UnknownMethod";
            string contextDetails = $"File: {file}, Method: {methodName}, Exception: {ex}";
            if (!string.IsNullOrEmpty(details))
            {
                contextDetails += $", Details: {details}";
            }
            Log("Error", message ?? ex.Message, contextDetails);
        }
    }
}
