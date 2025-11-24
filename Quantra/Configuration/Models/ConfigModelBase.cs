using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;

namespace Quantra.Configuration.Models
{
    /// <summary>
    /// Base class for all configuration models with change notification and validation
    /// </summary>
    public abstract class ConfigModelBase : INotifyPropertyChanged, IDataErrorInfo
    {
        private readonly Dictionary<string, object> _properties = new Dictionary<string, object>();
        private readonly Dictionary<string, string> _errors = new Dictionary<string, string>();

        /// <summary>
        /// Event that fires when a property changes
        /// </summary>
        public event PropertyChangedEventHandler? PropertyChanged;

        /// <summary>
        /// Gets the error message for the property with the given name.
        /// </summary>
        public string this[string propertyName]
        {
            get
            {
                ValidateProperty(propertyName);
                return _errors.TryGetValue(propertyName, out string error) ? error : string.Empty;
            }
        }

        /// <summary>
        /// Gets an error message indicating what is wrong with this object.
        /// </summary>
        public string Error => string.Join(Environment.NewLine, _errors.Values);

        /// <summary>
        /// Protected method to get a property value
        /// </summary>
        /// <typeparam name="T">Type of property</typeparam>
        /// <param name="defaultValue">Default value if not set</param>
        /// <param name="propertyName">Name of the property</param>
        /// <returns>The property value</returns>
        protected T Get<T>(T defaultValue = default, [CallerMemberName] string propertyName = null)
        {
            if (string.IsNullOrEmpty(propertyName))
                throw new ArgumentNullException(nameof(propertyName));

            if (_properties.TryGetValue(propertyName, out object value))
                return value is T typedValue ? typedValue : defaultValue;

            return defaultValue;
        }

        /// <summary>
        /// Protected method to set a property value
        /// </summary>
        /// <typeparam name="T">Type of property</typeparam>
        /// <param name="value">Value to set</param>
        /// <param name="propertyName">Name of the property</param>
        /// <returns>True if the value changed, false otherwise</returns>
        protected bool Set<T>(T value, [CallerMemberName] string propertyName = null)
        {
            if (string.IsNullOrEmpty(propertyName))
                throw new ArgumentNullException(nameof(propertyName));

            // Check if value has changed
            if (_properties.TryGetValue(propertyName, out object oldValue) &&
                EqualityComparer<T>.Default.Equals((T)oldValue, value))
                return false;

            _properties[propertyName] = value;

            // Validate the new value
            ValidateProperty(propertyName);

            OnPropertyChanged(propertyName);
            return true;
        }

        /// <summary>
        /// Validates a specific property
        /// </summary>
        /// <param name="propertyName">Name of the property to validate</param>
        protected virtual void ValidateProperty(string propertyName)
        {
            // Remove existing error
            _errors.Remove(propertyName);

            // Get property info
            var propertyInfo = GetType().GetProperty(propertyName);
            if (propertyInfo == null) return;

            // Get property value
            object value = propertyInfo.GetValue(this);

            // Get validation attributes
            var validationAttributes = (ValidationAttribute[])propertyInfo.GetCustomAttributes(typeof(ValidationAttribute), true);

            // Validate each attribute
            foreach (var attribute in validationAttributes)
            {
                if (!attribute.IsValid(value))
                {
                    _errors[propertyName] = attribute.FormatErrorMessage(propertyName);
                    break;
                }
            }
        }

        /// <summary>
        /// Validates all properties in the model
        /// </summary>
        /// <returns>True if valid, false if any properties are invalid</returns>
        public virtual bool Validate()
        {
            _errors.Clear();

            foreach (var property in GetType().GetProperties())
            {
                ValidateProperty(property.Name);
            }

            return _errors.Count == 0;
        }

        /// <summary>
        /// Raises the PropertyChanged event
        /// </summary>
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}