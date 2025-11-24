using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using Quantra.DAL.Services;
using Quantra.Utilities;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using System.Linq;

namespace Quantra
{
    /// <summary>
    /// Database access layer for the Quantra algorithmic trading platform.
    /// 
    /// MIGRATION NOTICE: This class is now a thin facade over proper services using Entity Framework Core.
    /// New code should use services directly via dependency injection (e.g., UserSettingsService, 
    /// OrderHistoryService, TradeRecordService, etc.).
    /// 
    /// This facade pattern maintains backward compatibility while the application migrates to
    /// proper repository pattern with EF Core and dependency injection.
    /// </summary>
    [Obsolete("Use services directly via dependency injection instead of DatabaseMonolith")]
    public class DatabaseMonolith
    {
        private bool initialized = false;
        private SettingsService _settingsService;
        private QuantraDbContext _dbContext;
        private UserSettingsService _userSettingsService;
        private LoggingService _loggingService;
        private static IConfiguration _configuration;
        private static string AlphaVantageApiKey;

        public DatabaseMonolith(SettingsService settingsService, LoggingService loggingService)
        {
            _settingsService = settingsService;

            // Initialize DbContext
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            _dbContext = new QuantraDbContext(optionsBuilder.Options);
            _loggingService = loggingService;

            // Initialize UserSettingsService
            _userSettingsService = new UserSettingsService(_dbContext, loggingService);

            Initialize();
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
        public void SetConfiguration(IConfiguration configuration)
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
                    _loggingService.LogError("Error loading Alpha Vantage settings", ex);
                }
            }
        }

        /// <summary>
        /// Logs a message to the database with optional details and automatic error alerting.
        /// </summary>
        [Obsolete("Use _loggingService.Log instead")]
        public void Log(string level, string message, string details = null)
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

                // Use EF Core to log
                var logEntry = new LogEntry
                {
                    Level = level,
                    Message = message,
                    Details = details,
                    Timestamp = DateTime.Now
                };

                _dbContext.Logs.Add(logEntry);
                _dbContext.SaveChanges();

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
                    Quantra.Utilities.Alerting.EmitGlobalAlert(alert);
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
        /// Initializes the database, creating tables if they don't exist.
        /// </summary>
        public void Initialize()
        {
            if (!initialized)
            {
                try
                {
                    // Use EF Core to initialize
                    _dbContext.Initialize();
                    initialized = true;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error initializing database: {ex.Message}");
                    throw;
                }
            }
        }

        /// <summary>
        /// Adds an order to the order history table using Entity Framework Core
        /// </summary>
        /// <param name="order">The order to add to history</param>
        [Obsolete("Use OrderHistoryService.AddOrderToHistory instead")]
        public void AddOrderToHistory(OrderModel order)
        {
            if (order == null)
            {
                Log("Error", "Cannot add null order to history");
                return;
            }

            try
            {
                // Map OrderModel to OrderHistoryEntity
                var entity = new OrderHistoryEntity
                {
                    Symbol = order.Symbol,
                    OrderType = order.OrderType,
                    Quantity = order.Quantity,
                    Price = order.Price,
                    StopLoss = order.StopLoss > 0 ? order.StopLoss : null,
                    TakeProfit = order.TakeProfit > 0 ? order.TakeProfit : null,
                    IsPaperTrade = order.IsPaperTrade,
                    Status = order.Status,
                    PredictionSource = order.PredictionSource ?? string.Empty,
                    Timestamp = order.Timestamp
                };

                _dbContext.OrderHistory.Add(entity);
                _dbContext.SaveChanges();

                Log("Info", $"Order added to history: {order.Symbol} {order.OrderType} {order.Quantity} @ {order.Price:C2}");
            }
            catch (Exception ex)
            {
                Log("Error", $"Failed to add order to history: {order.Symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Saves a trade record to the database using Entity Framework Core
        /// </summary>
        /// <param name="trade">The trade record to save</param>
        [Obsolete("Use TradeRecordService.SaveTradeRecord instead")]
        public void SaveTradeRecord(TradeRecord trade)
        {
            if (trade == null)
            {
                Log("Error", "Cannot save null TradeRecord");
                return;
            }

            try
            {
                // Map TradeRecord to TradeRecordEntity
                var entity = new TradeRecordEntity
                {
                    Symbol = trade.Symbol?.ToUpper(),
                    Action = trade.Action,
                    Price = trade.Price,
                    TargetPrice = trade.TargetPrice,
                    Confidence = trade.Confidence,
                    ExecutionTime = trade.ExecutionTime,
                    Status = trade.Status ?? "Executed",
                    Notes = trade.Notes
                };

                _dbContext.TradeRecords.Add(entity);
                _dbContext.SaveChanges();

                // Update the model with the generated ID
                trade.Id = entity.Id;

                Log("Info", $"TradeRecord saved: {trade.Symbol} {trade.Action} @ {trade.Price:C}");
            }
            catch (Exception ex)
            {
                Log("Error", $"Failed to save TradeRecord for {trade?.Symbol}", ex.ToString());
            }
        }

        #region Static Facade Methods (Backward Compatibility)

        /// <summary>
        /// Ensures all required database tables exist using Entity Framework Core
        /// </summary>
        [Obsolete("Use DatabaseInitializationService.EnsureUserAppSettingsTable via dependency injection")]
        public void EnsureUserAppSettingsTable()
        {
            try
            {
                // Create a temporary DbContext and service for backward compatibility
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new DatabaseInitializationService(dbContext, loggingService);
                service.EnsureUserAppSettingsTable();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to ensure UserAppSettings table", ex.ToString());
            }
        }

        /// <summary>
        /// Saves user settings to database
        /// </summary>
        [Obsolete("Use UserSettingsService.SaveUserSettings via dependency injection")]
        public static void SaveUserSettings(UserSettings settings)
        {
            try
            {
                // Create a temporary DbContext and service for backward compatibility
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new UserSettingsService(dbContext, loggingService);
                service.SaveUserSettings(settings);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to save user settings - {ex.Message}");
            }
        }

        /// <summary>
        /// Saves a setting by key-value pair to the database
        /// </summary>
        /// <param name="key">Setting key</param>
        /// <param name="value">Setting value</param>
        [Obsolete("Use UserSettingsService.SaveUserPreference via dependency injection")]
        public static void SaveSetting(string key, string value)
        {
            try
            {
                // Create a temporary DbContext and service for backward compatibility
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new UserSettingsService(dbContext, loggingService);
                service.SaveUserPreference(key, value);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to save setting: {key} - {ex.Message}");
            }
        }

        /// <summary>
        /// Gets user preference by key
        /// </summary>
        [Obsolete("Use UserSettingsService.GetUserPreference via dependency injection")]
        public static string GetUserPreference(String key, string defaultValue = null)
        {
            try
            {
                // Create a temporary DbContext and service for backward compatibility
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new UserSettingsService(dbContext, loggingService);
                return service.GetUserPreference(key, defaultValue);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to get user preference: {key} - {ex.Message}");
                return defaultValue;
            }
        }

        /// <summary>
        /// Saves user preference
        /// </summary>
        [Obsolete("Use UserSettingsService.SaveUserPreference via dependency injection")]
        public static void SaveUserPreference(string key, string value)
        {
            try
            {
                // Create a temporary DbContext and service for backward compatibility
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new UserSettingsService(dbContext, loggingService);
                service.SaveUserPreference(key, value);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to save user preference: {key} - {ex.Message}");
            }
        }

        /// <summary>
        /// Gets remembered accounts
        /// </summary>
        [Obsolete("Use UserSettingsService.GetRememberedAccounts via dependency injection")]
        public static Dictionary<string, (string Username, string Password, String Pin)> GetRememberedAccounts()
        {
            try
            {
                // Create a temporary DbContext and service for backward compatibility
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>
        ();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new UserSettingsService(dbContext, loggingService);
                return service.GetRememberedAccounts();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to get remembered accounts - {ex.Message}");
                return new Dictionary<string, (string, string, string)>();
            }
        }

        /// <summary>
        /// Remembers an account
        /// </summary>
        [Obsolete("Use UserSettingsService.RememberAccount via dependency injection")]
        public static void RememberAccount(string username, string password, string pin)
        {
            try
            {
                // Create a temporary DbContext and service for backward compatibility
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new UserSettingsService(dbContext, loggingService);
                service.RememberAccount(username, password, pin);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to remember account: {username} - {ex.Message}");
            }
        }

        #endregion

        /// <summary>
        /// Loads grid configuration (rows and columns) for a specific tab
        /// </summary>
        /// <param name="tabName">The name of the tab</param>
        /// <returns>A tuple containing the number of rows and columns</returns>
        [Obsolete("Use a dedicated TabConfigurationService via dependency injection")]
        public static (int Rows, int Columns) LoadGridConfig(string tabName)
        {
            try
            {
                // Create a temporary DbContext for backward compatibility
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);

                // Query the UserAppSettings table for the tab
                var tabConfig = dbContext.UserAppSettings
                     .AsNoTracking()
                      .FirstOrDefault(t => t.TabName == tabName);

                if (tabConfig != null)
                {
                    // Return the grid dimensions, ensuring minimum of 4x4
                    int rows = Math.Max(4, tabConfig.GridRows);
                    int columns = Math.Max(4, tabConfig.GridColumns);
                    return (rows, columns);
                }

                // If no config found, return default 4x4
                Console.WriteLine($"Warning: No grid configuration found for tab '{tabName}', using default 4x4");
                return (4, 4);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to load grid configuration for tab '{tabName}' - {ex.Message}");
                // Return default on error
                return (4, 4);
            }
        }

        /// <summary>
        /// Loads controls configuration for a specific tab
        /// </summary>
        /// <param name="tabName">The name of the tab</param>
        /// <returns>The controls configuration string, or empty string if not found</returns>
        [Obsolete("Use a dedicated TabConfigurationService via dependency injection")]
        public static string LoadControlsConfig(string tabName)
        {
            try
            {
                // Create a temporary DbContext and service for backward compatibility
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new TabConfigurationService(dbContext, loggingService);
                return service.LoadControlsConfig(tabName);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to load controls configuration for tab '{tabName}' - {ex.Message}");
                return string.Empty;
            }
        }

        /// <summary>
        /// Saves controls configuration for a specific tab
        /// </summary>
        /// <param name="tabName">The name of the tab</param>
        /// <param name="controlsConfig">The controls configuration string</param>
        [Obsolete("Use TabConfigurationService.SaveControlsConfig via dependency injection")]
        public static void SaveControlsConfig(string tabName, string controlsConfig)
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new TabConfigurationService(dbContext, loggingService);
                service.SaveControlsConfig(tabName, controlsConfig);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to save controls configuration for tab '{tabName}' - {ex.Message}");
            }
        }

        /// <summary>
        /// Adds a custom control with span configuration to a tab
        /// </summary>
        [Obsolete("Use TabConfigurationService.AddCustomControlWithSpans via dependency injection")]
        public static void AddCustomControlWithSpans(string tabName, string controlType, int row, int column, int rowSpan, int columnSpan)
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new TabConfigurationService(dbContext, loggingService);
                service.AddCustomControlWithSpans(tabName, controlType, row, column, rowSpan, columnSpan);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to add control with spans to tab '{tabName}' - {ex.Message}");
            }
        }

        /// <summary>
        /// Updates the position of a control in a tab
        /// </summary>
        [Obsolete("Use TabConfigurationService.UpdateControlPosition via dependency injection")]
        public static void UpdateControlPosition(string tabName, int controlIndex, int row, int column, int rowSpan, int columnSpan)
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new TabConfigurationService(dbContext, loggingService);
                service.UpdateControlPosition(tabName, controlIndex, row, column, rowSpan, columnSpan);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to update control position in tab '{tabName}' - {ex.Message}");
            }
        }

        /// <summary>
        /// Removes a control from a tab
        /// </summary>
        [Obsolete("Use TabConfigurationService.RemoveControl via dependency injection")]
        public static void RemoveControl(string tabName, int controlIndex)
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var loggingService = new LoggingService();
                var service = new TabConfigurationService(dbContext, loggingService);
                service.RemoveControl(tabName, controlIndex);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to remove control from tab '{tabName}' - {ex.Message}");
            }
        }

        /// <summary>
        /// Loads card positions (stub for now, not fully implemented)
        /// </summary>
        [Obsolete("This feature is not yet implemented")]
        public static string LoadCardPositions()
        {
            // This is a stub for future implementation
            return string.Empty;
        }

        /// <summary>
        /// Saves card positions (stub for now, not fully implemented)
        /// </summary>
        [Obsolete("This feature is not yet implemented")]
        public static void SaveCardPositions(string cardPositions)
        {
            // This is a stub for future implementation
        }

        /// <summary>
        /// Deletes a trading rule by ID
        /// </summary>
        /// <param name="ruleId">The ID of the rule to delete</param>
        [Obsolete("Use TradingRuleService.DeleteTradingRuleAsync via dependency injection")]
        public static void DeleteRule(int ruleId)
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                using var dbContext = new QuantraDbContext(optionsBuilder.Options);
                var service = new TradingRuleService(dbContext);

                // Call async method synchronously for backward compatibility
                bool deleted = service.DeleteTradingRuleAsync(ruleId).GetAwaiter().GetResult();

                if (deleted)
                {
                    Console.WriteLine($"Info: Deleted trading rule with Id {ruleId}");
                }
                else
                {
                    Console.WriteLine($"Warning: Trading rule with Id {ruleId} not found");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to delete trading rule with Id {ruleId} - {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Loads controls for a specific tab as a list of ControlModel objects
        /// </summary>
        /// <param name="tabName">The name of the tab</param>
        /// <returns>List of ControlModel objects</returns>
        [Obsolete("Use a dedicated TabConfigurationService via dependency injection")]
        public static List<ControlModel> LoadControlsForTab(string tabName)
        {
            try
            {
                // Load the controls configuration string
                var controlsConfig = LoadControlsConfig(tabName);

                // Return empty list if no configuration found
                if (string.IsNullOrWhiteSpace(controlsConfig))
                {
                    return new List<ControlModel>();
                }

                // Parse the controls configuration string
                // Format: "ControlType,Row,Column,RowSpan,ColSpan;ControlType,Row,Column,RowSpan,ColSpan;..."
                var controlsList = new List<ControlModel>();

                // Split by semicolons or newlines
                var controlEntries = controlsConfig
               .Replace("\r\n", ";")
              .Replace("\n", ";")
                  .Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);

                foreach (var entry in controlEntries)
                {
                    try
                    {
                        var parts = entry.Split(',');

                        if (parts.Length >= 3)
                        {
                            string type = parts[0].Trim();
                            int row = int.Parse(parts[1]);
                            int column = int.Parse(parts[2]);
                            int rowSpan = parts.Length >= 4 ? int.Parse(parts[3]) : 1;
                            int colSpan = parts.Length >= 5 ? int.Parse(parts[4]) : 1;

                            controlsList.Add(new ControlModel
                            {
                                Type = type,
                                Row = row,
                                Column = column,
                                RowSpan = rowSpan,
                                ColSpan = colSpan
                            });
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Warning: Failed to parse control entry: {entry} - {ex.Message}");
                    }
                }

                return controlsList;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: Failed to load controls for tab '{tabName}' - {ex.Message}");
                return new List<ControlModel>();
            }
        }
    }
}
