using System;
using System.Collections.Generic;

namespace Quantra.CrossCutting.Logging
{
    /// <summary>
    /// Represents a structured logger interface with enriched context capabilities.
    /// </summary>
    public interface ILogger
    {
        /// <summary>
        /// Logs a verbose message.
        /// </summary>
        void Verbose(string messageTemplate, params object[] propertyValues);
        
        /// <summary>
        /// Logs a debug message.
        /// </summary>
        void Debug(string messageTemplate, params object[] propertyValues);
        
        /// <summary>
        /// Logs an informational message.
        /// </summary>
        void Information(string messageTemplate, params object[] propertyValues);
        
        /// <summary>
        /// Logs a warning message.
        /// </summary>
        void Warning(string messageTemplate, params object[] propertyValues);
        
        /// <summary>
        /// Logs an error message.
        /// </summary>
        void Error(string messageTemplate, params object[] propertyValues);
        
        /// <summary>
        /// Logs an error with exception details.
        /// </summary>
        void Error(Exception exception, string messageTemplate, params object[] propertyValues);
        
        /// <summary>
        /// Logs a fatal error message.
        /// </summary>
        void Fatal(string messageTemplate, params object[] propertyValues);
        
        /// <summary>
        /// Logs a fatal error with exception details.
        /// </summary>
        void Fatal(Exception exception, string messageTemplate, params object[] propertyValues);
        
        /// <summary>
        /// Creates a new logger with added contextual properties.
        /// </summary>
        ILogger ForContext(string propertyName, object value);
        
        /// <summary>
        /// Creates a new logger for a specific source context.
        /// </summary>
        ILogger ForContext<T>();
        
        /// <summary>
        /// Creates a new logger for a specific source context.
        /// </summary>
        ILogger ForContext(Type source);
        
        /// <summary>
        /// Begins a new logical operation scope that will be included in logs.
        /// </summary>
        IDisposable BeginScope(string operationName, IDictionary<string, object> state = null);
        
        /// <summary>
        /// Begins a timed operation that will log elapsed time upon disposal.
        /// </summary>
        IDisposable BeginTimedOperation(string operationName, string messageTemplate = "Operation {OperationName} completed in {ElapsedMilliseconds}ms");
    }
}