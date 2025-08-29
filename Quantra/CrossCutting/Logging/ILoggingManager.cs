using System;
using Quantra.CrossCutting;

namespace Quantra.CrossCutting.Logging
{
    /// <summary>
    /// Centralized logging management interface.
    /// </summary>
    public interface ILoggingManager : ICrossCuttingModule
    {
        /// <summary>
        /// Gets a logger for the specified context.
        /// </summary>
        ILogger GetLogger<T>();
        
        /// <summary>
        /// Gets a logger for the specified context.
        /// </summary>
        ILogger GetLogger(Type contextType);
        
        /// <summary>
        /// Gets a logger for the specified context name.
        /// </summary>
        ILogger GetLogger(string context);
        
        /// <summary>
        /// Adds a trading session context to subsequent logs.
        /// </summary>
        void SetCurrentTradingSession(string sessionId);
        
        /// <summary>
        /// Sets the correlation ID for distributed tracing.
        /// </summary>
        void SetCorrelationId(string correlationId);
        
        /// <summary>
        /// Gets the current correlation ID.
        /// </summary>
        string GetCurrentCorrelationId();
        
        /// <summary>
        /// Enables or disables the asynchronous logging pipeline.
        /// </summary>
        void SetAsyncLogging(bool enabled);
        
        /// <summary>
        /// Flushes any buffered log messages.
        /// </summary>
        void Flush();
    }
}