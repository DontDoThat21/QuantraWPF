using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using Newtonsoft.Json;

namespace Quantra.DAL.Models
{
    /// <summary>
    /// Represents a trading signal configuration with multiple indicators and conditions
    /// </summary>
    public class TradingSignal : INotifyPropertyChanged
    {
        private int _id;
        private string _name = string.Empty;
        private string _description = string.Empty;
        private bool _isEnabled = true;
        private DateTime _createdDate = DateTime.Now;
        private DateTime _lastModified = DateTime.Now;
        private ObservableCollection<SignalSymbol> _symbols = new ObservableCollection<SignalSymbol>();
        private ObservableCollection<SignalIndicator> _indicators = new ObservableCollection<SignalIndicator>();
        private ObservableCollection<SignalCondition> _conditions = new ObservableCollection<SignalCondition>();
        private SignalAlertConfiguration _alertConfiguration = new SignalAlertConfiguration();

        public int Id
        {
            get => _id;
            set => SetField(ref _id, value);
        }

        public string Name
        {
            get => _name;
            set => SetField(ref _name, value);
        }

        public string Description
        {
            get => _description;
            set => SetField(ref _description, value);
        }

        public bool IsEnabled
        {
            get => _isEnabled;
            set => SetField(ref _isEnabled, value);
        }

        public DateTime CreatedDate
        {
            get => _createdDate;
            set => SetField(ref _createdDate, value);
        }

        public DateTime LastModified
        {
            get => _lastModified;
            set => SetField(ref _lastModified, value);
        }

        public ObservableCollection<SignalSymbol> Symbols
        {
            get => _symbols;
            set => SetField(ref _symbols, value);
        }

        public ObservableCollection<SignalIndicator> Indicators
        {
            get => _indicators;
            set => SetField(ref _indicators, value);
        }

        public ObservableCollection<SignalCondition> Conditions
        {
            get => _conditions;
            set => SetField(ref _conditions, value);
        }

        public SignalAlertConfiguration AlertConfiguration
        {
            get => _alertConfiguration;
            set => SetField(ref _alertConfiguration, value);
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>
        /// Validates the signal configuration
        /// </summary>
        public bool Validate(out string errorMessage)
        {
            errorMessage = string.Empty;

            if (string.IsNullOrWhiteSpace(Name))
            {
                errorMessage = "Signal name is required.";
                return false;
            }

            if (Symbols == null || Symbols.Count == 0)
            {
                errorMessage = "At least one symbol is required.";
                return false;
            }

            if (Indicators == null || Indicators.Count == 0)
            {
                errorMessage = "At least one indicator must be selected.";
                return false;
            }

            // Validate total allocation percentage
            double totalAllocation = 0;
            foreach (var symbol in Symbols)
            {
                if (string.IsNullOrWhiteSpace(symbol.Symbol))
                {
                    errorMessage = "All symbols must have a valid ticker.";
                    return false;
                }
                totalAllocation += symbol.AllocationPercentage;
            }

            if (Math.Abs(totalAllocation - 100) > 0.01 && Symbols.Count > 1)
            {
                errorMessage = $"Total allocation must equal 100%. Current: {totalAllocation:F1}%";
                return false;
            }

            return true;
        }
    }

    /// <summary>
    /// Represents a symbol within a trading signal
    /// </summary>
    public class SignalSymbol : INotifyPropertyChanged
    {
        private string _symbol = string.Empty;
        private double _allocationPercentage = 100;
        private bool _isActive = true;
        private bool _isValidated;
        private string _validationMessage = string.Empty;

        public string Symbol
        {
            get => _symbol;
            set => SetField(ref _symbol, value?.ToUpper() ?? string.Empty);
        }

        public double AllocationPercentage
        {
            get => _allocationPercentage;
            set => SetField(ref _allocationPercentage, Math.Max(0, Math.Min(100, value)));
        }

        public bool IsActive
        {
            get => _isActive;
            set => SetField(ref _isActive, value);
        }

        public bool IsValidated
        {
            get => _isValidated;
            set => SetField(ref _isValidated, value);
        }

        public string ValidationMessage
        {
            get => _validationMessage;
            set => SetField(ref _validationMessage, value);
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    /// <summary>
    /// Represents a technical indicator configuration within a trading signal
    /// </summary>
    public class SignalIndicator : INotifyPropertyChanged
    {
        private string _indicatorType = string.Empty;
        private string _displayName = string.Empty;
        private bool _isSelected;
        private Dictionary<string, object> _parameters = new Dictionary<string, object>();

        public string IndicatorType
        {
            get => _indicatorType;
            set => SetField(ref _indicatorType, value);
        }

        public string DisplayName
        {
            get => _displayName;
            set => SetField(ref _displayName, value);
        }

        public bool IsSelected
        {
            get => _isSelected;
            set => SetField(ref _isSelected, value);
        }

        public Dictionary<string, object> Parameters
        {
            get => _parameters;
            set => SetField(ref _parameters, value);
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>
        /// Gets the default parameters for the indicator type
        /// </summary>
        public static Dictionary<string, object> GetDefaultParameters(string indicatorType)
        {
            return indicatorType switch
            {
                "RSI" => new Dictionary<string, object> { { "Period", 14 }, { "OverboughtLevel", 70.0 }, { "OversoldLevel", 30.0 } },
                "MACD" => new Dictionary<string, object> { { "FastPeriod", 12 }, { "SlowPeriod", 26 }, { "SignalPeriod", 9 } },
                "VWAP" => new Dictionary<string, object> { { "AnchorPeriod", "Session" } },
                "ADX" => new Dictionary<string, object> { { "Period", 14 }, { "TrendThreshold", 25.0 } },
                "BollingerBands" => new Dictionary<string, object> { { "Period", 20 }, { "StdDevMultiplier", 2.0 } },
                "EMA" => new Dictionary<string, object> { { "Period", 20 } },
                "SMA" => new Dictionary<string, object> { { "Period", 20 } },
                "StochasticRSI" => new Dictionary<string, object> { { "RSIPeriod", 14 }, { "StochPeriod", 14 }, { "KPeriod", 3 }, { "DPeriod", 3 } },
                "OBV" => new Dictionary<string, object>(),
                "Momentum" => new Dictionary<string, object> { { "Period", 10 } },
                "CCI" => new Dictionary<string, object> { { "Period", 20 }, { "OverboughtLevel", 100.0 }, { "OversoldLevel", -100.0 } },
                "ATR" => new Dictionary<string, object> { { "Period", 14 } },
                _ => new Dictionary<string, object>()
            };
        }
    }

    /// <summary>
    /// Represents a condition for triggering a trading signal
    /// </summary>
    public class SignalCondition : INotifyPropertyChanged
    {
        private string _indicator = string.Empty;
        private string _comparisonOperator = ">";
        private double _thresholdValue;
        private string _logicalOperator = "AND";
        private int _order;

        public string Indicator
        {
            get => _indicator;
            set => SetField(ref _indicator, value);
        }

        public string ComparisonOperator
        {
            get => _comparisonOperator;
            set => SetField(ref _comparisonOperator, value);
        }

        public double ThresholdValue
        {
            get => _thresholdValue;
            set => SetField(ref _thresholdValue, value);
        }

        public string LogicalOperator
        {
            get => _logicalOperator;
            set => SetField(ref _logicalOperator, value);
        }

        public int Order
        {
            get => _order;
            set => SetField(ref _order, value);
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>
        /// Gets the available comparison operators
        /// </summary>
        public static List<string> GetComparisonOperators()
        {
            return new List<string>
            {
                ">",
                "<",
                ">=",
                "<=",
                "=",
                "!=",
                "Crosses Above",
                "Crosses Below"
            };
        }

        /// <summary>
        /// Gets the available logical operators
        /// </summary>
        public static List<string> GetLogicalOperators()
        {
            return new List<string> { "AND", "OR" };
        }

        /// <summary>
        /// Returns the condition as a human-readable string
        /// </summary>
        public override string ToString()
        {
            return $"{Indicator} {ComparisonOperator} {ThresholdValue}";
        }
    }

    /// <summary>
    /// Configuration for signal alerts
    /// </summary>
    public class SignalAlertConfiguration : INotifyPropertyChanged
    {
        private bool _emailNotificationEnabled;
        private bool _smsNotificationEnabled;
        private string _alertSeverity = "Normal";

        public bool EmailNotificationEnabled
        {
            get => _emailNotificationEnabled;
            set => SetField(ref _emailNotificationEnabled, value);
        }

        public bool SmsNotificationEnabled
        {
            get => _smsNotificationEnabled;
            set => SetField(ref _smsNotificationEnabled, value);
        }

        public string AlertSeverity
        {
            get => _alertSeverity;
            set => SetField(ref _alertSeverity, value);
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>
        /// Gets the available severity levels
        /// </summary>
        public static List<string> GetSeverityLevels()
        {
            return new List<string> { "Low", "Normal", "High", "Critical" };
        }
    }
}
