using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Quantra.CrossCutting;
using Quantra.CrossCutting.Logging;

namespace Quantra.CrossCutting.Security
{
    /// <summary>
    /// Manager for security-related concerns like sensitive data handling.
    /// </summary>
    public class SecurityManager : ISecurityManager
    {
        private static readonly Lazy<SecurityManager> _instance = new Lazy<SecurityManager>(() => new SecurityManager());
        private readonly ILogger _logger;
        private readonly HashSet<string> _sensitivePropertyNames;
        private readonly Regex _sensitivePatterns;

        /// <summary>
        /// Gets the singleton instance of the SecurityManager.
        /// </summary>
        public static SecurityManager Instance => _instance.Value;

        /// <inheritdoc />
        public string ModuleName => "Security";

        /// <summary>
        /// Private constructor to enforce singleton pattern.
        /// </summary>
        private SecurityManager()
        {
            _sensitivePropertyNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "password", "apikey", "secret", "token", "credential", "ssn",
                "creditcard", "cvv", "creditcardnumber", "socialsecurity",
                "key", "accesstoken", "refreshtoken", "pwd", "pass", "pin",
                "securityanswer", "privatekey"
            };
            
            // Combine all sensitive patterns
            var patternBuilder = new StringBuilder();
            patternBuilder.Append(@"(");
            patternBuilder.Append(@"\b(?:password|secret|token|api[-_]?key|access[-_]?token|client[-_]?secret)[-_]?['""=:\s]{1,20}['""](.*?)['""]\b");
            patternBuilder.Append(@"|");
            patternBuilder.Append(@"\b(?:(?:\d{4}[-_ ]){3}\d{4}|\d{16})\b"); // Credit card
            patternBuilder.Append(@"|");
            patternBuilder.Append(@"\b\d{3}[-_ ]?\d{2}[-_ ]?\d{4}\b"); // SSN
            patternBuilder.Append(@")");
            
            _sensitivePatterns = new Regex(patternBuilder.ToString(), 
                RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Multiline);
            
            _logger = Log.ForType<SecurityManager>();
        }

        /// <inheritdoc />
        public void Initialize(string configurationSection = null)
        {
            _logger.Information("SecurityManager initialized with {SensitivePropertyCount} sensitive property patterns",
                _sensitivePropertyNames.Count);
        }

        /// <inheritdoc />
        public string SanitizeLogMessage(string message)
        {
            if (string.IsNullOrEmpty(message))
            {
                return message;
            }

            try
            {
                return _sensitivePatterns.Replace(message, m =>
                {
                    // First group is the whole match, subsequent groups are the captures
                    if (m.Groups.Count > 1 && m.Groups[1].Success)
                    {
                        return m.Value.Substring(0, m.Groups[1].Index - m.Index) + "***REDACTED***";
                    }

                    return "***REDACTED***";
                });
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to sanitize log message, returning safe version");
                return "***LOG MESSAGE CONTAINED POTENTIAL SENSITIVE DATA***";
            }
        }

        /// <inheritdoc />
        public IDictionary<string, object> SanitizeProperties(IDictionary<string, object> properties)
        {
            if (properties == null)
            {
                return null;
            }

            var result = new Dictionary<string, object>(properties.Count);
            
            foreach (var kvp in properties)
            {
                if (IsSensitiveProperty(kvp.Key))
                {
                    result[kvp.Key] = "***REDACTED***";
                }
                else if (kvp.Value is string strValue)
                {
                    result[kvp.Key] = SanitizeLogMessage(strValue);
                }
                else
                {
                    result[kvp.Key] = kvp.Value;
                }
            }
            
            return result;
        }

        /// <inheritdoc />
        public string RedactSensitiveData(string text)
        {
            return SanitizeLogMessage(text);
        }

        /// <inheritdoc />
        public T RedactSensitiveData<T>(T obj) where T : class
        {
            if (obj == null)
            {
                return null;
            }

            try
            {
                // For simple string, just use direct redaction
                if (obj is string strValue)
                {
                    return RedactSensitiveData(strValue) as T;
                }
                
                // Serialize to JSON then deserialize to create a deep copy
                var json = JsonConvert.SerializeObject(obj);
                
                // Redact the JSON string
                var redactedJson = RedactSensitiveData(json);
                
                // Deserialize back
                return JsonConvert.DeserializeObject<T>(redactedJson);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to redact sensitive data from object of type {Type}", typeof(T).Name);
                return obj; // Return original if we can't redact
            }
        }

        /// <inheritdoc />
        public bool IsSensitiveProperty(string propertyName)
        {
            if (string.IsNullOrEmpty(propertyName))
            {
                return false;
            }
            
            // Check if it's in our list of sensitive properties
            return _sensitivePropertyNames.Contains(propertyName);
        }

        /// <inheritdoc />
        public void RegisterSensitiveProperty(string propertyName)
        {
            if (!string.IsNullOrEmpty(propertyName))
            {
                _sensitivePropertyNames.Add(propertyName);
                _logger.Debug("Registered {PropertyName} as sensitive property", propertyName);
            }
        }

        /// <inheritdoc />
        public string SecureConnectionString(string connectionString)
        {
            if (string.IsNullOrEmpty(connectionString))
            {
                return connectionString;
            }

            try
            {
                // Extract individual parameters
                var parts = connectionString.Split(';')
                    .Where(p => !string.IsNullOrWhiteSpace(p))
                    .Select(p => p.Trim())
                    .ToList();
                
                // Replace sensitive information
                for (int i = 0; i < parts.Count; i++)
                {
                    var part = parts[i];
                    
                    if (part.StartsWith("Password=", StringComparison.OrdinalIgnoreCase) ||
                        part.StartsWith("Pwd=", StringComparison.OrdinalIgnoreCase) ||
                        part.StartsWith("User ID=", StringComparison.OrdinalIgnoreCase) ||
                        part.StartsWith("UID=", StringComparison.OrdinalIgnoreCase) ||
                        part.StartsWith("User=", StringComparison.OrdinalIgnoreCase) ||
                        part.StartsWith("ApiKey=", StringComparison.OrdinalIgnoreCase) ||
                        part.StartsWith("Secret=", StringComparison.OrdinalIgnoreCase))
                    {
                        var eqIndex = part.IndexOf('=');
                        if (eqIndex >= 0 && eqIndex < part.Length - 1)
                        {
                            parts[i] = part.Substring(0, eqIndex + 1) + "***REDACTED***";
                        }
                    }
                }
                
                return string.Join(";", parts);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to secure connection string");
                return "***REDACTED CONNECTION STRING***";
            }
        }
    }
}