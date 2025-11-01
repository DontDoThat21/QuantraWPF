using Dapper;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Data;
//using System.Data.SQLite;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows;
using Quantra.DAL.Services;
using Quantra.Utilities;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Data.Repositories;

namespace Quantra
{
    /// <summary>
    /// Database access layer for the Quantra algorithmic trading platform.
    /// 
    /// MIGRATION NOTICE: This class now uses Entity Framework Core (QuantraDbContext) internally
    /// while maintaining backward compatibility with existing code. New code should use
    /// QuantraDbContext directly via dependency injection.
    /// 
    /// This facade pattern allows gradual migration from the monolithic approach to proper
    /// repository pattern with EF Core.
    /// </summary>
    public class DatabaseMonolith
    {
        private static readonly string DbFilePath = "Quantra.db";
  //public readonly string ConnectionString = $"Data Source={DbFilePath};Version=3;Journal Mode=WAL;Busy Timeout=30000;";
     private bool initialized = false;
        private IConfiguration _configuration;
     private SettingsService _settingsService;
    private QuantraDbContext _dbContext;

        // Store API keys for backward compatibility
  public string AlphaVantageApiKey { get; internal set; }

     public DatabaseMonolith(SettingsService settingsService)
        {
          _settingsService = settingsService;
          Initialize();
        }

        /// <summary>
        /// Logs a message to the database with optional details and automatic error alerting.
        /// </summary>
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

                // Use EF Core instead of raw SQLite
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
            Alerting.EmitGlobalAlert(alert);
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
                    Log("Error", "Error loading Alpha Vantage settings", ex.ToString());
                }
            }
        }

        /// <summary>
        /// Initializes the database, creating the file and tables if they don't exist.
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
        /// Gets user settings from database. Returns default settings if none exist.
        /// TODO: Implement proper persistence using UserPreferences or a dedicated settings table
        /// </summary>
        public static UserSettings GetUserSettings()
        {
            // For now, return default UserSettings to allow compilation
            // TODO: Load from database using _dbContext or UserPreferences table
            return new UserSettings();
        }

        /// <summary>
        /// Saves user settings to database
        /// TODO: Implement proper persistence using UserPreferences or a dedicated settings table
        /// </summary>
        public static void SaveUserSettings(UserSettings settings)
        {
            // TODO: Persist to database using _dbContext or UserPreferences table
            LoggingService.Log("Info", "SaveUserSettings called - persistence not yet implemented");
        }

        /// <summary>
        /// Gets user preference by key
        /// </summary>
        public static string GetUserPreference(string key, string defaultValue = null)
        {
            // TODO: Implement using _dbContext.UserPreferences
            return defaultValue;
        }

        /// <summary>
        /// Saves user preference
        /// </summary>
        public static void SaveUserPreference(string key, string value)
        {
            // TODO: Implement using _dbContext.UserPreferences
            LoggingService.Log("Info", $"SaveUserPreference called for key: {key}");
        }

        /// <summary>
        /// Gets remembered accounts
        /// </summary>
        public static Dictionary<string, (string Username, string Password, string Pin)> GetRememberedAccounts()
        {
            // TODO: Implement using _dbContext.UserCredentials
            return new Dictionary<string, (string, string, string)>();
        }

        /// <summary>
        /// Remembers an account
        /// </summary>
        public static void RememberAccount(string username, string password, string pin)
        {
            // TODO: Implement using _dbContext.UserCredentials
            LoggingService.Log("Info", $"RememberAccount called for username: {username}");
        }

        // ...existing code...
    }
}
