using System;
using System.Collections.Generic;
using Quantra.CrossCutting;

namespace Quantra.CrossCutting.Security
{
    /// <summary>
    /// Interface for the security manager.
    /// </summary>
    public interface ISecurityManager : ICrossCuttingModule
    {
        /// <summary>
        /// Sanitizes a log message by removing sensitive information.
        /// </summary>
        string SanitizeLogMessage(string message);
        
        /// <summary>
        /// Sanitizes a collection of properties by removing sensitive information.
        /// </summary>
        IDictionary<string, object> SanitizeProperties(IDictionary<string, object> properties);
        
        /// <summary>
        /// Redacts sensitive information from a string.
        /// </summary>
        string RedactSensitiveData(string text);
        
        /// <summary>
        /// Redacts sensitive information from an object.
        /// </summary>
        T RedactSensitiveData<T>(T obj) where T : class;
        
        /// <summary>
        /// Checks if a property name should be treated as sensitive.
        /// </summary>
        bool IsSensitiveProperty(string propertyName);
        
        /// <summary>
        /// Registers a sensitive property name.
        /// </summary>
        void RegisterSensitiveProperty(string propertyName);
        
        /// <summary>
        /// Secures a connection string by removing sensitive information.
        /// </summary>
        string SecureConnectionString(string connectionString);
    }
}