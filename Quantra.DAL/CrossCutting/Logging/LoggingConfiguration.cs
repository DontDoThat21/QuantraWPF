using System;
using System.Collections.Generic;

namespace Quantra.CrossCutting.Logging
{
    /// <summary>
    /// Configuration options for the logging system.
    /// </summary>
    public class LoggingConfiguration
    {
        /// <summary>
        /// Minimum level for log events to be captured.
        /// </summary>
        public LogLevel MinimumLevel { get; set; } = LogLevel.Information;
        
        /// <summary>
        /// Path for log files.
        /// </summary>
        public string LogFilePath { get; set; } = "logs/quantra-.log";
        
        /// <summary>
        /// Number of days to retain log files.
        /// </summary>
        public int RetainedFileCountLimit { get; set; } = 31;
        
        /// <summary>
        /// Maximum size of individual log files.
        /// </summary>
        public long FileSizeLimitBytes { get; set; } = 1073741824; // 1GB
        
        /// <summary>
        /// Whether to use console logging.
        /// </summary>
        public bool UseConsoleLogging { get; set; } = true;
        
        /// <summary>
        /// Whether to use file logging.
        /// </summary>
        public bool UseFileLogging { get; set; } = true;
        
        /// <summary>
        /// Whether to use database logging.
        /// </summary>
        public bool UseDatabaseLogging { get; set; } = true;
        
        /// <summary>
        /// Whether to use JSON formatting for logs.
        /// </summary>
        public bool UseJsonFormatting { get; set; } = true;
        
        /// <summary>
        /// Whether to enable asynchronous logging.
        /// </summary>
        public bool EnableAsyncLogging { get; set; } = true;
        
        /// <summary>
        /// Buffer size for asynchronous logging.
        /// </summary>
        public int AsyncBufferSize { get; set; } = 10000;
        
        /// <summary>
        /// Overrides for specific source contexts.
        /// </summary>
        public Dictionary<string, LogLevel> SourceLevelOverrides { get; set; } = new Dictionary<string, LogLevel>();
        
        /// <summary>
        /// Email recipients for critical error notifications.
        /// </summary>
        public List<string> CriticalErrorEmailRecipients { get; set; } = new List<string>();
        
        /// <summary>
        /// SMS recipients for critical error notifications.
        /// </summary>
        public List<string> CriticalErrorSmsRecipients { get; set; } = new List<string>();
        
        /// <summary>
        /// Whether to redact sensitive information in logs.
        /// </summary>
        public bool RedactSensitiveData { get; set; } = true;
        
        /// <summary>
        /// List of property names containing sensitive data to be redacted.
        /// </summary>
        public List<string> SensitiveProperties { get; set; } = new List<string> 
        { 
            "Password", "ApiKey", "Token", "Secret", "Credential", "SSN", "CreditCard" 
        };
        
        /// <summary>
        /// Max response size to log (to prevent massive responses from filling logs).
        /// </summary>
        public int MaxResponseSizeToLog { get; set; } = 4096; // 4KB
        
        /// <summary>
        /// Default correlation ID header name.
        /// </summary>
        public string CorrelationIdHeaderName { get; set; } = "X-Correlation-ID";
    }
    
    /// <summary>
    /// Log level severity.
    /// </summary>
    public enum LogLevel
    {
        Verbose,
        Debug,
        Information,
        Warning,
        Error,
        Fatal
    }
}