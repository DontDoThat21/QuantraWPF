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
  private IConfiguration _configuration;
   private SettingsService _settingsService;
        private QuantraDbContext _dbContext;
        private UserSettingsService _userSettingsService;

   // Store API keys for backward compatibility
        public string AlphaVantageApiKey { get; internal set; }

   public DatabaseMonolith(SettingsService settingsService)
 {
    _settingsService = settingsService;
 
      // Initialize DbContext
        var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
       optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
        _dbContext = new QuantraDbContext(optionsBuilder.Options);
     
        // Initialize UserSettingsService
  _userSettingsService = new UserSettingsService(_dbContext);
       
   Initialize();
        }

      /// <summary>
        /// Logs a message to the database with optional details and automatic error alerting.
   /// </summary>
  [Obsolete("Use LoggingService.Log instead")]
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
    // Store in memory for backward compatibility
     AlphaVantageApiKey = apiKey;
}
        }
      catch (Exception ex)
 {
  Log("Error", "Error loading Alpha Vantage settings", ex.ToString());
}
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
        /// Gets user settings from database. Returns default settings if none exist.
   /// </summary>
      [Obsolete("Use UserSettingsService.GetUserSettings via dependency injection")]
  public static UserSettings GetUserSettings()
        {
  try
            {
    // Create a temporary DbContext and service for backward compatibility
     var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
        optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
      using var dbContext = new QuantraDbContext(optionsBuilder.Options);
  var service = new UserSettingsService(dbContext);
 return service.GetUserSettings();
 }
          catch (Exception ex)
 {
     LoggingService.Log("Error", "Failed to get user settings, returning defaults", ex.ToString());
   return new UserSettings();
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
        var service = new UserSettingsService(dbContext);
      service.SaveUserSettings(settings);
   }
        catch (Exception ex)
       {
    LoggingService.Log("Error", "Failed to save user settings", ex.ToString());
   }
    }

        /// <summary>
        /// Saves user settings to database with pin and API modal checks
      /// </summary>
      /// <param name="pin">User pin</param>
        /// <param name="enableApiModalChecks">Enable API modal checks</param>
        [Obsolete("Use UserSettingsService.SaveUserSettings via dependency injection")]
public static void SaveUserSettings(string pin, bool enableApiModalChecks)
  {
         try
         {
     var settings = GetUserSettings();
                settings.EnableApiModalChecks = enableApiModalChecks;
            SaveUserSettings(settings);
            }
    catch (Exception ex)
 {
    LoggingService.Log("Error", "Failed to save user settings with pin", ex.ToString());
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
                var service = new UserSettingsService(dbContext);
         service.SaveUserPreference(key, value);
       }
         catch (Exception ex)
         {
    LoggingService.Log("Error", $"Failed to save setting: {key}", ex.ToString());
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
    var service = new UserSettingsService(dbContext);
        return service.GetUserPreference(key, defaultValue);
}
       catch (Exception ex)
    {
    LoggingService.Log("Error", $"Failed to get user preference: {key}", ex.ToString());
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
         var service = new UserSettingsService(dbContext);
   service.SaveUserPreference(key, value);
}
            catch (Exception ex)
{
  LoggingService.Log("Error", $"Failed to save user preference: {key}", ex.ToString());
      }
 }

        /// <summary>
        /// Gets remembered accounts
        /// </summary>
    [Obsolete("Use UserSettingsService.GetRememberedAccounts via dependency injection")]
        public static Dictionary<string, (string Username, string Password, string Pin)> GetRememberedAccounts()
        {
      try
   {
 // Create a temporary DbContext and service for backward compatibility
        var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
   optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
  using var dbContext = new QuantraDbContext(optionsBuilder.Options);
 var service = new UserSettingsService(dbContext);
       return service.GetRememberedAccounts();
    }
     catch (Exception ex)
    {
      LoggingService.Log("Error", "Failed to get remembered accounts", ex.ToString());
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
var service = new UserSettingsService(dbContext);
      service.RememberAccount(username, password, pin);
    }
 catch (Exception ex)
        {
     LoggingService.Log("Error", $"Failed to remember account: {username}", ex.ToString());
  }
 }

        #endregion
  }
}
