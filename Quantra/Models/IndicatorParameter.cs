using System;
using System.ComponentModel;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a configurable parameter for a technical indicator
    /// </summary>
    public class IndicatorParameter : INotifyPropertyChanged
    {
        private string _name;
        public string Name
        {
            get => _name;
            set
            {
                if (_name != value)
                {
                    _name = value;
                    OnPropertyChanged(nameof(Name));
                }
            }
        }

        private string _description;
        public string Description
        {
            get => _description;
            set
            {
                if (_description != value)
                {
                    _description = value;
                    OnPropertyChanged(nameof(Description));
                }
            }
        }

        private object _value;
        public object Value
        {
            get => _value;
            set
            {
                if (!Equals(_value, value))
                {
                    if (!IsValidValue(value))
                        throw new ArgumentException($"Invalid value for parameter {Name}");
                    
                    _value = value;
                    OnPropertyChanged(nameof(Value));
                }
            }
        }

        private object _defaultValue;
        public object DefaultValue
        {
            get => _defaultValue;
            set
            {
                if (!Equals(_defaultValue, value))
                {
                    _defaultValue = value;
                    OnPropertyChanged(nameof(DefaultValue));
                }
            }
        }

        private object _minValue;
        public object MinValue
        {
            get => _minValue;
            set
            {
                if (!Equals(_minValue, value))
                {
                    _minValue = value;
                    OnPropertyChanged(nameof(MinValue));
                }
            }
        }

        private object _maxValue;
        public object MaxValue
        {
            get => _maxValue;
            set
            {
                if (!Equals(_maxValue, value))
                {
                    _maxValue = value;
                    OnPropertyChanged(nameof(MaxValue));
                }
            }
        }

        public Type ParameterType { get; set; }

        private bool _isOptional;
        public bool IsOptional
        {
            get => _isOptional;
            set
            {
                if (_isOptional != value)
                {
                    _isOptional = value;
                    OnPropertyChanged(nameof(IsOptional));
                }
            }
        }

        private string[] _options;
        public string[] Options
        {
            get => _options;
            set
            {
                if (_options != value)
                {
                    _options = value;
                    OnPropertyChanged(nameof(Options));
                }
            }
        }

        public IndicatorParameter()
        {
            // Default constructor
        }

        public IndicatorParameter(string name, string description, object defaultValue, Type parameterType)
        {
            Name = name;
            Description = description;
            DefaultValue = defaultValue;
            Value = defaultValue;
            ParameterType = parameterType;
            IsOptional = false;
        }

        public bool IsValidValue(object value)
        {
            // If value is null but parameter is optional, allow it
            if (value == null && IsOptional)
                return true;

            // Validate type
            if (value == null || !ParameterType.IsInstanceOfType(value))
                return false;

            // Validate numeric range
            if (value is IComparable comparableValue)
            {
                if (MinValue != null && comparableValue.CompareTo(MinValue) < 0)
                    return false;
                
                if (MaxValue != null && comparableValue.CompareTo(MaxValue) > 0)
                    return false;
            }

            // Validate options
            if (Options != null && Options.Length > 0)
            {
                if (value is string stringValue)
                    return Array.IndexOf(Options, stringValue) >= 0;
            }

            return true;
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}