using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Models
{
    /// <summary>
    /// Base class for all custom technical indicators
    /// </summary>
    public abstract class CustomIndicator : IIndicator, INotifyPropertyChanged
    {
        private string _id;
        public string Id
        {
            get => _id;
            protected set
            {
                if (_id != value)
                {
                    _id = value;
                    OnPropertyChanged(nameof(Id));
                }
            }
        }

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

        private string _category;
        public string Category
        {
            get => _category;
            set
            {
                if (_category != value)
                {
                    _category = value;
                    OnPropertyChanged(nameof(Category));
                }
            }
        }

        private bool _isComposable;
        public bool IsComposable
        {
            get => _isComposable;
            protected set
            {
                if (_isComposable != value)
                {
                    _isComposable = value;
                    OnPropertyChanged(nameof(IsComposable));
                }
            }
        }

        protected Dictionary<string, IndicatorParameter> _parameters;
        public Dictionary<string, IndicatorParameter> Parameters => _parameters;

        protected List<string> _dependencies;

        public CustomIndicator()
        {
            _parameters = new Dictionary<string, IndicatorParameter>();
            _dependencies = new List<string>();
            _isComposable = true;

            // Generate a unique ID if one isn't provided
            _id = Guid.NewGuid().ToString();
        }

        public virtual IEnumerable<string> GetDependencies()
        {
            return _dependencies;
        }

        public abstract Task<Dictionary<string, double>> CalculateAsync(List<HistoricalPrice> historicalData);

        /// <summary>
        /// Validates that the required parameters are set and valid before calculation
        /// </summary>
        protected virtual void ValidateParameters()
        {
            foreach (var param in Parameters.Values.Where(p => !p.IsOptional))
            {
                if (param.Value == null)
                    throw new InvalidOperationException($"Required parameter '{param.Name}' is not set");

                if (!param.IsValidValue(param.Value))
                    throw new InvalidOperationException($"Parameter '{param.Name}' has an invalid value");
            }
        }

        /// <summary>
        /// Helper method to register a dependency on another indicator
        /// </summary>
        /// <param name="indicatorId">The ID of the indicator this indicator depends on</param>
        protected void AddDependency(string indicatorId)
        {
            if (!_dependencies.Contains(indicatorId))
                _dependencies.Add(indicatorId);
        }

        /// <summary>
        /// Helper method to register a parameter for this indicator
        /// </summary>
        protected void RegisterParameter(IndicatorParameter parameter)
        {
            if (_parameters.ContainsKey(parameter.Name))
                throw new InvalidOperationException($"Parameter '{parameter.Name}' is already registered");

            _parameters.Add(parameter.Name, parameter);
        }

        /// <summary>
        /// Helper method to register a simple numeric parameter
        /// </summary>
        protected void RegisterParameter(string name, string description, double defaultValue,
                                      double? minValue = null, double? maxValue = null, bool isOptional = false)
        {
            var param = new IndicatorParameter
            {
                Name = name,
                Description = description,
                DefaultValue = defaultValue,
                Value = defaultValue,
                ParameterType = typeof(double),
                IsOptional = isOptional
            };

            if (minValue.HasValue)
                param.MinValue = minValue.Value;

            if (maxValue.HasValue)
                param.MaxValue = maxValue.Value;

            RegisterParameter(param);
        }

        /// <summary>
        /// Helper method to register an integer parameter
        /// </summary>
        protected void RegisterIntParameter(string name, string description, int defaultValue,
                                      int? minValue = null, int? maxValue = null, bool isOptional = false)
        {
            var param = new IndicatorParameter
            {
                Name = name,
                Description = description,
                DefaultValue = defaultValue,
                Value = defaultValue,
                ParameterType = typeof(int),
                IsOptional = isOptional
            };

            if (minValue.HasValue)
                param.MinValue = minValue.Value;

            if (maxValue.HasValue)
                param.MaxValue = maxValue.Value;

            RegisterParameter(param);
        }

        /// <summary>
        /// Helper method to register a string selection parameter
        /// </summary>
        protected void RegisterSelectionParameter(string name, string description, string defaultValue,
                                               string[] options, bool isOptional = false)
        {
            var param = new IndicatorParameter
            {
                Name = name,
                Description = description,
                DefaultValue = defaultValue,
                Value = defaultValue,
                ParameterType = typeof(string),
                IsOptional = isOptional,
                Options = options
            };

            RegisterParameter(param);
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}