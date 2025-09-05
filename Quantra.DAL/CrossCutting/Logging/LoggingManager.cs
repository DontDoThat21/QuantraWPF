using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Configuration;
using Quantra;
using Quantra.CrossCutting;
using Serilog;
using Serilog.Core;
using Serilog.Events;
using Serilog.Formatting.Compact;
using Serilog.Formatting.Json;
using Serilog.Sinks.SystemConsole.Themes;

namespace Quantra.CrossCutting.Logging
{
    /// <summary>
    /// Manages all logging functionality through a centralized system.
    /// </summary>
    public class LoggingManager : ILoggingManager
    {
        private static readonly Lazy<LoggingManager> _instance = new Lazy<LoggingManager>(() => new LoggingManager());
        private readonly ConcurrentDictionary<string, ILogger> _loggers = new ConcurrentDictionary<string, ILogger>();
        private LoggingConfiguration _configuration;
        private Serilog.ILogger _rootLogger;
        private LoggingLevelSwitch _levelSwitch;
        private string _correlationId;
        private string _tradingSessionId;
        private bool _initialized;
        
        /// <summary>
        /// Gets the singleton instance of the LoggingManager.
        /// </summary>
        public static LoggingManager Instance => _instance.Value;

        /// <inheritdoc />
        public string ModuleName => "Logging";

        /// <summary>
        /// Private constructor to enforce singleton pattern.
        /// </summary>
        private LoggingManager()
        {
            _configuration = new LoggingConfiguration();
            _levelSwitch = new LoggingLevelSwitch(GetLogEventLevel(_configuration.MinimumLevel));
        }

        /// <inheritdoc />
        public void Initialize(string configurationSection = "Logging")
        {
            if (_initialized)
            {
                return;
            }

            // Load configuration if available
            if (!string.IsNullOrEmpty(configurationSection))
            {
                LoadConfiguration(configurationSection);
            }

            // Configure Serilog
            var loggerConfiguration = new LoggerConfiguration()
                .MinimumLevel.ControlledBy(_levelSwitch)
                .Enrich.FromLogContext()
                .Enrich.WithThreadId()
                .Enrich.WithMachineName();

            // Apply any source context overrides
            foreach (var kvp in _configuration.SourceLevelOverrides)
            {
                loggerConfiguration.MinimumLevel.Override(kvp.Key, GetLogEventLevel(kvp.Value));
            }

            // Add console logging if enabled
            if (_configuration.UseConsoleLogging)
            {
                if (_configuration.UseJsonFormatting)
                {
                    loggerConfiguration.WriteTo.Console(new CompactJsonFormatter());
                }
                else
                {
                    loggerConfiguration.WriteTo.Console(
                        theme: AnsiConsoleTheme.Code,
                        outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] [{SourceContext}] {Message:lj}{NewLine}{Exception}");
                }
            }

            // Add file logging if enabled
            if (_configuration.UseFileLogging)
            {
                // Ensure the directory exists
                string directory = Path.GetDirectoryName(_configuration.LogFilePath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                if (_configuration.UseJsonFormatting)
                {
                    loggerConfiguration.WriteTo.File(
                        new CompactJsonFormatter(),
                        _configuration.LogFilePath,
                        rollingInterval: RollingInterval.Day,
                        retainedFileCountLimit: _configuration.RetainedFileCountLimit,
                        fileSizeLimitBytes: _configuration.FileSizeLimitBytes);
                }
                else
                {
                    loggerConfiguration.WriteTo.File(
                        _configuration.LogFilePath,
                        rollingInterval: RollingInterval.Day,
                        retainedFileCountLimit: _configuration.RetainedFileCountLimit,
                        fileSizeLimitBytes: _configuration.FileSizeLimitBytes,
                        outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] [{SourceContext}] ({CorrelationId}) {Message:lj}{NewLine}{Exception}");
                }
            }

            // Use SQLite database logging if enabled
            if (_configuration.UseDatabaseLogging)
            {
                // We'll implement database logging through our adapter to existing system
                // This is handled in our custom database sink
                loggerConfiguration.WriteTo.Sink(new DatabaseLogSink());
            }

            // Configure asynchronous logging
            if (_configuration.EnableAsyncLogging)
            {
                // Configure background worker with buffer
                loggerConfiguration.WriteTo.Async(c => { }, _configuration.AsyncBufferSize);
            }

            // Create the root logger
            _rootLogger = loggerConfiguration.CreateLogger();

            // Initialize global static logger for compatibility
            Serilog.Log.Logger = _rootLogger;

            // Set the initialized flag
            _initialized = true;

            // Log the initialization
            var logger = GetLogger(GetType());
            logger.Information("Logging system initialized. JSON formatting: {JsonFormatting}, Async: {AsyncLogging}", 
                _configuration.UseJsonFormatting, _configuration.EnableAsyncLogging);
        }

        /// <inheritdoc />
        public ILogger GetLogger<T>()
        {
            return GetLogger(typeof(T));
        }

        /// <inheritdoc />
        public ILogger GetLogger(Type contextType)
        {
            if (contextType == null)
            {
                throw new ArgumentNullException(nameof(contextType));
            }

            return GetLogger(contextType.FullName);
        }

        /// <inheritdoc />
        public ILogger GetLogger(string context)
        {
            if (!_initialized)
            {
                Initialize();
            }

            if (string.IsNullOrEmpty(context))
            {
                context = "Unknown";
            }

            return _loggers.GetOrAdd(context, key =>
            {
                var contextualLogger = _rootLogger.ForContext("SourceContext", key);
                
                // Add global contexts if they exist
                if (!string.IsNullOrEmpty(_correlationId))
                {
                    contextualLogger = contextualLogger.ForContext("CorrelationId", _correlationId);
                }
                
                if (!string.IsNullOrEmpty(_tradingSessionId))
                {
                    contextualLogger = contextualLogger.ForContext("TradingSessionId", _tradingSessionId);
                }
                
                return new SerilogLogger(contextualLogger);
            });
        }

        /// <inheritdoc />
        public void SetCurrentTradingSession(string sessionId)
        {
            _tradingSessionId = sessionId;
            _loggers.Clear(); // Force regeneration of loggers with new session ID
        }

        /// <inheritdoc />
        public void SetCorrelationId(string correlationId)
        {
            _correlationId = correlationId;
            _loggers.Clear(); // Force regeneration of loggers with new correlation ID
        }

        /// <inheritdoc />
        public string GetCurrentCorrelationId()
        {
            return _correlationId ?? (_correlationId = Guid.NewGuid().ToString("N"));
        }

        /// <inheritdoc />
        public void SetAsyncLogging(bool enabled)
        {
            // This change requires reinitialization
            _configuration.EnableAsyncLogging = enabled;
            _initialized = false;
            _loggers.Clear();
            Initialize();
        }

        /// <inheritdoc />
        public void Flush()
        {
            Serilog.Log.CloseAndFlush();
        }

        /// <summary>
        /// Updates the minimum log level dynamically.
        /// </summary>
        public void SetMinimumLevel(LogLevel level)
        {
            _configuration.MinimumLevel = level;
            _levelSwitch.MinimumLevel = GetLogEventLevel(level);
        }
        
        /// <summary>
        /// Loads configuration from application settings.
        /// </summary>
        private void LoadConfiguration(string sectionName)
        {
            try
            {
                // Try to load from appsettings.json
                var config = new ConfigurationBuilder()
                    .SetBasePath(Directory.GetCurrentDirectory())
                    .AddJsonFile("appsettings.json", optional: true)
                    .AddJsonFile($"appsettings.{Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT")}.json", optional: true)
                    .Build();

                var section = config.GetSection(sectionName);
                if (section.Exists())
                {
                    // Map the configuration section to our LoggingConfiguration class
                    _configuration = new LoggingConfiguration
                    {
                        MinimumLevel = Enum.TryParse<LogLevel>(section["MinimumLevel"], true, out var level) ? level : LogLevel.Information,
                        UseConsoleLogging = bool.TryParse(section["UseConsoleLogging"], out var useCon) ? useCon : true,
                        UseFileLogging = bool.TryParse(section["UseFileLogging"], out var useFile) ? useFile : true,
                        UseDatabaseLogging = bool.TryParse(section["UseDatabaseLogging"], out var useDb) ? useDb : true,
                        LogFilePath = section["LogFilePath"] ?? "logs/quantra-.log",
                        UseJsonFormatting = bool.TryParse(section["UseJsonFormatting"], out var useJson) ? useJson : true,
                        EnableAsyncLogging = bool.TryParse(section["EnableAsyncLogging"], out var asyncLog) ? asyncLog : true,
                        RetainedFileCountLimit = int.TryParse(section["RetainedFileCountLimit"], out var retainCount) ? retainCount : 31,
                        FileSizeLimitBytes = long.TryParse(section["FileSizeLimitBytes"], out var sizeLimit) ? sizeLimit : 1073741824,
                        RedactSensitiveData = bool.TryParse(section["RedactSensitiveData"], out var redact) ? redact : true
                    };

                    // Load override configurations
                    var overridesSection = section.GetSection("SourceLevelOverrides");
                    if (overridesSection.Exists())
                    {
                        foreach (var child in overridesSection.GetChildren())
                        {
                            if (Enum.TryParse<LogLevel>(child.Value, true, out var overrideLevel))
                            {
                                _configuration.SourceLevelOverrides[child.Key] = overrideLevel;
                            }
                        }
                    }

                    // Load email notifications
                    var emailSection = section.GetSection("CriticalErrorEmailRecipients");
                    if (emailSection.Exists())
                    {
                        _configuration.CriticalErrorEmailRecipients = new List<string>();
                        foreach (var email in emailSection.GetChildren())
                        {
                            _configuration.CriticalErrorEmailRecipients.Add(email.Value);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                // Log this to console since the logger isn't fully initialized yet
                Console.Error.WriteLine($"Error loading logging configuration: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Converts our LogLevel enum to Serilog's LogEventLevel.
        /// </summary>
        private LogEventLevel GetLogEventLevel(LogLevel level)
        {
            return level switch
            {
                LogLevel.Verbose => LogEventLevel.Verbose,
                LogLevel.Debug => LogEventLevel.Debug,
                LogLevel.Information => LogEventLevel.Information,
                LogLevel.Warning => LogEventLevel.Warning,
                LogLevel.Error => LogEventLevel.Error,
                LogLevel.Fatal => LogEventLevel.Fatal,
                _ => LogEventLevel.Information
            };
        }

        /// <summary>
        /// Custom sink to write logs to the database using the existing DatabaseMonolith.
        /// </summary>
        private class DatabaseLogSink : Serilog.Core.ILogEventSink
        {
            public void Emit(LogEvent logEvent)
            {
                try
                {
                    // Map Serilog log level to our existing level strings
                    string level = logEvent.Level switch
                    {
                        LogEventLevel.Verbose => "Trace",
                        LogEventLevel.Debug => "Debug",
                        LogEventLevel.Information => "Info",
                        LogEventLevel.Warning => "Warning",
                        LogEventLevel.Error => "Error",
                        LogEventLevel.Fatal => "Critical",
                        _ => "Info"
                    };

                    // Extract message and format it
                    string message = logEvent.RenderMessage();

                    // Extract exception details if any
                    string details = null;
                    if (logEvent.Exception != null)
                    {
                        details = logEvent.Exception.ToString();
                    }
                    
                    // Add contextual information to details if available
                    var contextualDetails = new List<string>();
                    
                    if (logEvent.Properties.TryGetValue("SourceContext", out var sourceContext))
                    {
                        contextualDetails.Add($"Source: {sourceContext}");
                    }
                    
                    if (logEvent.Properties.TryGetValue("CorrelationId", out var correlationId))
                    {
                        contextualDetails.Add($"CorrelationId: {correlationId}");
                    }
                    
                    if (logEvent.Properties.TryGetValue("TradingSessionId", out var sessionId))
                    {
                        contextualDetails.Add($"SessionId: {sessionId}");
                    }
                    
                    // Add any other properties as JSON
                    var otherProps = new Dictionary<string, object>();
                    foreach (var prop in logEvent.Properties)
                    {
                        if (prop.Key != "SourceContext" && 
                            prop.Key != "CorrelationId" && 
                            prop.Key != "TradingSessionId")
                        {
                            // Try to extract the raw value from the property value
                            try
                            {
                                var value = prop.Value.ToString().Trim('"');
                                otherProps[prop.Key] = value;
                            }
                            catch
                            {
                                otherProps[prop.Key] = prop.Value.ToString();
                            }
                        }
                    }

                    if (otherProps.Count > 0)
                    {
                        try
                        {
                            var propsJson = System.Text.Json.JsonSerializer.Serialize(otherProps);
                            contextualDetails.Add($"Properties: {propsJson}");
                        }
                        catch
                        {
                            // Ignore JSON serialization errors for properties
                        }
                    }
                    
                    // Combine details
                    if (contextualDetails.Count > 0)
                    {
                        string contextsString = string.Join(", ", contextualDetails);
                        details = string.IsNullOrEmpty(details) 
                            ? contextsString 
                            : $"{contextsString}\n{details}";
                    }

                    // Use the existing logging infrastructure
                    DatabaseMonolith.Log(level, message, details);
                }
                catch (Exception ex)
                {
                    // Last resort fallback to console
                    Console.Error.WriteLine($"Failed to write log to database: {ex.Message}");
                }
            }
        }
    }
}