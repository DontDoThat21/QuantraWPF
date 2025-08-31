using System.Collections.Generic;

namespace Quantra.CrossCutting.Security
{
    /// <summary>
    /// Static helper for security operations.
    /// </summary>
    public static class Security
    {
        private static readonly ISecurityManager _manager = SecurityManager.Instance;
        
        /// <summary>
        /// Initializes the security system.
        /// </summary>
        public static void Initialize()
        {
            _manager.Initialize();
        }
        
        /// <summary>
        /// Sanitizes a log message by removing sensitive information.
        /// </summary>
        public static string SanitizeLogMessage(string message)
        {
            return _manager.SanitizeLogMessage(message);
        }
        
        /// <summary>
        /// Sanitizes a collection of properties by removing sensitive information.
        /// </summary>
        public static IDictionary<string, object> SanitizeProperties(IDictionary<string, object> properties)
        {
            return _manager.SanitizeProperties(properties);
        }
        
        /// <summary>
        /// Redacts sensitive information from a string.
        /// </summary>
        public static string RedactSensitiveData(string text)
        {
            return _manager.RedactSensitiveData(text);
        }
        
        /// <summary>
        /// Redacts sensitive information from an object.
        /// </summary>
        public static T RedactSensitiveData<T>(T obj) where T : class
        {
            return _manager.RedactSensitiveData(obj);
        }
        
        /// <summary>
        /// Checks if a property name should be treated as sensitive.
        /// </summary>
        public static bool IsSensitiveProperty(string propertyName)
        {
            return _manager.IsSensitiveProperty(propertyName);
        }
        
        /// <summary>
        /// Registers a sensitive property name.
        /// </summary>
        public static void RegisterSensitiveProperty(string propertyName)
        {
            _manager.RegisterSensitiveProperty(propertyName);
        }
        
        /// <summary>
        /// Secures a connection string by removing sensitive information.
        /// </summary>
        public static string SecureConnectionString(string connectionString)
        {
            return _manager.SecureConnectionString(connectionString);
        }
    }
}