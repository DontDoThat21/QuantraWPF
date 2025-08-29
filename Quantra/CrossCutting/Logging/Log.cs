using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Quantra.CrossCutting.Logging
{
    /// <summary>
    /// Static facade for logging operations.
    /// </summary>
    public static class Log
    {
        private static readonly ILoggingManager _manager = LoggingManager.Instance;

        /// <summary>
        /// Initializes the logging system.
        /// </summary>
        public static void Initialize(string configSectionName = "Logging")
        {
            _manager.Initialize(configSectionName);
        }

        /// <summary>
        /// Gets a logger for the specified type.
        /// </summary>
        public static ILogger ForType<T>()
        {
            return _manager.GetLogger<T>();
        }

        /// <summary>
        /// Gets a logger for the specified type.
        /// </summary>
        public static ILogger ForType(Type type)
        {
            return _manager.GetLogger(type);
        }

        /// <summary>
        /// Gets a logger for the specified context name.
        /// </summary>
        public static ILogger ForContext(string contextName)
        {
            return _manager.GetLogger(contextName);
        }

        /// <summary>
        /// Gets a logger for the calling class.
        /// </summary>
        public static ILogger Here([CallerFilePath] string sourceFilePath = "")
        {
            // Extract class name from file path
            string className = System.IO.Path.GetFileNameWithoutExtension(sourceFilePath);
            return _manager.GetLogger(className);
        }

        /// <summary>
        /// Sets the correlation ID for distributed tracing.
        /// </summary>
        public static void SetCorrelationId(string correlationId)
        {
            _manager.SetCorrelationId(correlationId);
        }

        /// <summary>
        /// Gets the current correlation ID.
        /// </summary>
        public static string GetCorrelationId()
        {
            return _manager.GetCurrentCorrelationId();
        }

        /// <summary>
        /// Sets the current trading session ID.
        /// </summary>
        public static void SetTradingSession(string sessionId)
        {
            _manager.SetCurrentTradingSession(sessionId);
        }

        /// <summary>
        /// Begins a timed operation scope for performance tracking.
        /// </summary>
        /// <param name="operationName">Name of the operation being timed</param>
        /// <param name="properties">Optional additional properties to include in logs</param>
        /// <param name="messageTemplate">Optional message template override</param>
        /// <returns>An IDisposable that must be disposed when the operation completes</returns>
        public static IDisposable TimeOperation(string operationName, 
            IDictionary<string, object> properties = null, 
            string messageTemplate = "Operation {OperationName} completed in {ElapsedMilliseconds}ms")
        {
            var logger = _manager.GetLogger("PerformanceMetrics");
            
            if (properties != null)
            {
                foreach (var prop in properties)
                {
                    logger = logger.ForContext(prop.Key, prop.Value);
                }
            }
            
            return logger.BeginTimedOperation(operationName, messageTemplate);
        }

        /// <summary>
        /// Logs an exception with contextual information.
        /// </summary>
        public static void Exception(Exception ex, string message = null, 
            [CallerMemberName] string memberName = "", 
            [CallerFilePath] string sourceFilePath = "", 
            [CallerLineNumber] int sourceLineNumber = 0)
        {
            var className = System.IO.Path.GetFileNameWithoutExtension(sourceFilePath);
            var logger = _manager.GetLogger(className);
            
            // Add context information
            logger = logger.ForContext("MemberName", memberName)
                          .ForContext("FilePath", sourceFilePath)
                          .ForContext("LineNumber", sourceLineNumber);
            
            // Use provided message or exception message
            logger.Error(ex, message ?? ex.Message);
        }

        /// <summary>
        /// Flushes any buffered log entries.
        /// </summary>
        public static void Flush()
        {
            _manager.Flush();
        }
    }
}