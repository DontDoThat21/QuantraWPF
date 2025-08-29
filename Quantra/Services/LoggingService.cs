using System;
using System.Data.SQLite;
using System.Runtime.CompilerServices;
using Quantra.CrossCutting;
using Quantra.CrossCutting.Logging;
using Quantra.CrossCutting.Security;

namespace Quantra.Services
{
    /// <summary>
    /// Service for centralized logging, providing backward compatibility with existing code.
    /// </summary>
    public static class LoggingService
    {
        static LoggingService()
        {
            // Make sure cross-cutting concerns are initialized
            CrossCuttingRegistry.Initialize();
        }
        
        /// <summary>
        /// Logs a message with the specified level.
        /// </summary>
        public static void Log(string level, string message, string details = null)
        {
            // For backward compatibility, map to both old and new logging systems
            DatabaseMonolith.Log(level, message, details);
            
            // Map to the new logging system
            ILogger logger = Quantra.CrossCutting.Logging.Log.ForContext("Legacy");
            string sanitizedMessage = Security.SanitizeLogMessage(message);
            string sanitizedDetails = Security.SanitizeLogMessage(details);
            
            switch (level?.ToLowerInvariant())
            {
                case "trace":
                case "verbose":
                    logger.Verbose(sanitizedMessage);
                    break;
                case "debug":
                    logger.Debug(sanitizedMessage);
                    break;
                case "info":
                case "information":
                    logger.Information(sanitizedMessage);
                    break;
                case "warning":
                case "warn":
                    logger.Warning(sanitizedMessage);
                    break;
                case "error":
                    logger.Error(sanitizedMessage);
                    break;
                case "critical":
                case "fatal":
                    logger.Fatal(sanitizedMessage);
                    break;
                default:
                    logger.Information(sanitizedMessage);
                    break;
            }
            
            // Log details if provided
            if (!string.IsNullOrEmpty(sanitizedDetails))
            {
                logger.Debug("Details: {Details}", sanitizedDetails);
            }
        }

        /// <summary>
        /// Logs an error with automatic file and method context using reflection/stack trace.
        /// </summary>
        public static void LogErrorWithContext(Exception ex, string message = null, string details = null, 
            [CallerMemberName] string memberName = "", 
            [CallerFilePath] string sourceFilePath = "", 
            [CallerLineNumber] int sourceLineNumber = 0)
        {
            // For backward compatibility
            DatabaseMonolith.LogErrorWithContext(ex, message, details);
            
            // Use the new logging system with better context handling
            string errorMessage = message ?? ex.Message;
            string sanitizedMessage = Security.SanitizeLogMessage(errorMessage);
            string sanitizedDetails = Security.SanitizeLogMessage(details);
            
            // Extract class name from file path
            string className = System.IO.Path.GetFileNameWithoutExtension(sourceFilePath);
            Type type = Type.GetType(className) ?? typeof(LoggingService);
            var logger = Quantra.CrossCutting.Logging.Log.ForType(type)
                          .ForContext("MemberName", memberName)
                          .ForContext("FilePath", sourceFilePath)
                          .ForContext("LineNumber", sourceLineNumber);
            
            if (!string.IsNullOrEmpty(sanitizedDetails))
            {
                logger = logger.ForContext("Details", sanitizedDetails);
            }
            
            logger.Error(ex, sanitizedMessage);
        }
        
        /// <summary>
        /// Logs an informational message.
        /// </summary>
        public static void LogInformation(string message, object[] args = null)
        {
            if (args != null && args.Length > 0)
            {
                Quantra.CrossCutting.Logging.Log.Here().Information(message, args);
            }
            else
            {
                LoggingService.Log("Info", message);
            }
        }
        
        /// <summary>
        /// Logs a warning message.
        /// </summary>
        public static void LogWarning(string message, object[] args = null)
        {
            if (args != null && args.Length > 0)
            {
                Quantra.CrossCutting.Logging.Log.Here().Warning(message, args);
            }
            else
            {
                LoggingService.Log("Warning", message);
            }
        }
        
        /// <summary>
        /// Logs a debug message.
        /// </summary>
        public static void LogDebug(string message, object[] args = null)
        {
            if (args != null && args.Length > 0)
            {
                Quantra.CrossCutting.Logging.Log.Here().Debug(message, args);
            }
            else
            {
                LoggingService.Log("Debug", message);
            }
        }
        
        /// <summary>
        /// Logs an error message with optional exception details.
        /// </summary>
        public static void LogError(string message, Exception ex = null)
        {
            if (ex != null)
            {
                Quantra.CrossCutting.Logging.Log.Here().Error(ex, message);
            }
            else
            {
                LoggingService.Log("Error", message);
            }
        }
    }
}
