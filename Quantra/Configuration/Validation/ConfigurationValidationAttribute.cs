using System;
using System.ComponentModel.DataAnnotations;

namespace Quantra.Configuration.Validation
{
    /// <summary>
    /// Base class for configuration validation attributes
    /// </summary>
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
    public abstract class ConfigurationValidationAttribute : ValidationAttribute
    {
        /// <summary>
        /// The environment in which this validation applies
        /// </summary>
        public string[] Environments { get; set; }
        
        /// <summary>
        /// Error level for validation failures
        /// </summary>
        public ValidationErrorLevel ErrorLevel { get; set; } = ValidationErrorLevel.Error;
        
        /// <summary>
        /// Description of the validation rule
        /// </summary>
        public string Description { get; set; }
        
        /// <summary>
        /// Constructor
        /// </summary>
        protected ConfigurationValidationAttribute()
        {
            Environments = new[] { "Development", "Staging", "Production" };
        }
        
        /// <summary>
        /// Constructor with environments
        /// </summary>
        protected ConfigurationValidationAttribute(params string[] environments)
        {
            Environments = environments;
        }
        
        /// <summary>
        /// Checks if this validation applies to the current environment
        /// </summary>
        /// <param name="environment">Current environment</param>
        /// <returns>True if applies, false otherwise</returns>
        public bool AppliesTo(string environment)
        {
            if (Environments == null || Environments.Length == 0)
                return true;
                
            foreach (var env in Environments)
            {
                if (string.Equals(env, environment, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            
            return false;
        }
    }
    
    /// <summary>
    /// Error level for validation failures
    /// </summary>
    public enum ValidationErrorLevel
    {
        /// <summary>
        /// Warning, doesn't prevent application from running
        /// </summary>
        Warning,
        
        /// <summary>
        /// Error, prevents the configuration from being applied
        /// </summary>
        Error,
        
        /// <summary>
        /// Critical error, prevents the application from starting
        /// </summary>
        Critical
    }
    
    /// <summary>
    /// Validation attribute for range values
    /// </summary>
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
    public class ConfigurationRangeAttribute : ConfigurationValidationAttribute
    {
        /// <summary>
        /// Minimum value
        /// </summary>
        public object Minimum { get; }
        
        /// <summary>
        /// Maximum value
        /// </summary>
        public object Maximum { get; }
        
        /// <summary>
        /// Constructor
        /// </summary>
        public ConfigurationRangeAttribute(object minimum, object maximum)
        {
            Minimum = minimum;
            Maximum = maximum;
        }
        
        /// <summary>
        /// Validate the value
        /// </summary>
        public override bool IsValid(object value)
        {
            if (value == null)
                return true; // Skip validation for null values
                
            var type = value.GetType();
            
            if (type == typeof(int))
            {
                int min = Convert.ToInt32(Minimum);
                int max = Convert.ToInt32(Maximum);
                int val = (int)value;
                return val >= min && val <= max;
            }
            else if (type == typeof(double))
            {
                double min = Convert.ToDouble(Minimum);
                double max = Convert.ToDouble(Maximum);
                double val = (double)value;
                return val >= min && val <= max;
            }
            else if (type == typeof(decimal))
            {
                decimal min = Convert.ToDecimal(Minimum);
                decimal max = Convert.ToDecimal(Maximum);
                decimal val = (decimal)value;
                return val >= min && val <= max;
            }
            
            return false;
        }
    }
    
    /// <summary>
    /// Validation attribute for requiring a value in specific environments
    /// </summary>
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
    public class RequiredInEnvironmentAttribute : ConfigurationValidationAttribute
    {
        /// <summary>
        /// Constructor
        /// </summary>
        public RequiredInEnvironmentAttribute(params string[] environments) : base(environments)
        {
        }
        
        /// <summary>
        /// Validate the value
        /// </summary>
        public override bool IsValid(object value)
        {
            // Only validate in environments where this is required
            string currentEnvironment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") 
                ?? Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT")
                ?? "Production";
                
            if (!AppliesTo(currentEnvironment))
                return true;
                
            if (value == null)
                return false;
                
            if (value is string s)
                return !string.IsNullOrWhiteSpace(s);
                
            return true;
        }
    }
}