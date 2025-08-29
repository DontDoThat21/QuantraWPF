using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Quantra.Services;
using Quantra.Services.Interfaces;

namespace Quantra.Models
{
    /// <summary>
    /// An indicator that is composed of multiple other indicators
    /// </summary>
    public class CompositeIndicator : CustomIndicator
    {
        /// <summary>
        /// Function delegate for composite calculations
        /// </summary>
        public delegate double CalculationDelegate(Dictionary<string, Dictionary<string, double>> inputs);

        /// <summary>
        /// The indicator calculation method
        /// </summary>
        private CalculationDelegate _calculationMethod;

        /// <summary>
        /// Output keys for this indicator (if more than one output value)
        /// </summary>
        private readonly List<string> _outputKeys;

        /// <summary>
        /// Service for accessing other indicators
        /// </summary>
        private readonly ITechnicalIndicatorService _indicatorService;

        /// <summary>
        /// Get a reference to all indicators used in this composite indicator
        /// </summary>
        public IReadOnlyList<string> SourceIndicators => _dependencies;

        /// <summary>
        /// Constructor for a composite indicator
        /// </summary>
        /// <param name="name">The display name of the indicator</param>
        /// <param name="calculationMethod">The calculation delegate to use for computation</param>
        /// <param name="dependencies">List of indicators this composite depends on</param>
        /// <param name="outputKey">The primary output key (default is "Value")</param>
        /// <param name="category">The category for this indicator</param>
        /// <param name="description">Description of what this indicator measures</param>
        public CompositeIndicator(
            string name,
            CalculationDelegate calculationMethod,
            List<string> dependencies,
            string outputKey = "Value",
            string category = "Custom",
            string description = "Custom composite indicator")
        {
            Name = name;
            Description = description;
            Category = category;
            _calculationMethod = calculationMethod;
            _outputKeys = new List<string> { outputKey };
            
            // Add all dependencies to our tracking
            foreach (var dependency in dependencies)
            {
                AddDependency(dependency);
            }
            
            // Get the indicator service via dependency injection or service locator
            _indicatorService = ServiceLocator.GetService<ITechnicalIndicatorService>();
        }

        /// <summary>
        /// Constructor for a multi-output composite indicator 
        /// </summary>
        /// <param name="name">The display name of the indicator</param>
        /// <param name="calculationMethod">The calculation delegate to use for computation</param>
        /// <param name="dependencies">List of indicators this composite depends on</param>
        /// <param name="outputKeys">The output keys for multiple values</param>
        /// <param name="category">The category for this indicator</param>
        /// <param name="description">Description of what this indicator measures</param>
        public CompositeIndicator(
            string name,
            CalculationDelegate calculationMethod,
            List<string> dependencies,
            List<string> outputKeys,
            string category = "Custom",
            string description = "Custom composite indicator")
            : this(name, calculationMethod, dependencies, outputKeys[0], category, description)
        {
            _outputKeys = outputKeys;
        }

        public override async Task<Dictionary<string, double>> CalculateAsync(List<HistoricalPrice> historicalData)
        {
            ValidateParameters();

            // If there's no data, return default values
            if (historicalData == null || historicalData.Count == 0)
            {
                return _outputKeys.ToDictionary(key => key, _ => 0.0);
            }

            // First, calculate all dependencies
            var dependencyValues = new Dictionary<string, Dictionary<string, double>>();
            
            foreach (var dependency in _dependencies)
            {
                try
                {
                    // Retrieve the dependent indicator
                    var indicator = await _indicatorService.GetIndicatorAsync(dependency);
                    if (indicator == null)
                    {
                        throw new InvalidOperationException($"Dependent indicator '{dependency}' not found");
                    }
                    
                    // Calculate its values using the same historical data
                    var values = await indicator.CalculateAsync(historicalData);
                    dependencyValues[dependency] = values;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to calculate dependency '{dependency}'", ex);
                }
            }

            // Calculate the composite value(s)
            var result = new Dictionary<string, double>();
            
            try
            {
                // Call the calculation delegate with all dependency values
                var calculatedValue = _calculationMethod(dependencyValues);
                
                // For simple outputs, assign the single calculated value to the output key
                if (_outputKeys.Count == 1)
                {
                    result[_outputKeys[0]] = calculatedValue;
                }
                else
                {
                    // For complex outputs, we need to implement additional logic or require 
                    // a more complex delegate that returns a Dictionary<string, double>
                    // This is a simplified implementation
                    foreach (var key in _outputKeys)
                    {
                        result[key] = calculatedValue;
                    }
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Failed to calculate composite indicator", ex);
            }

            return result;
        }

        /// <summary>
        /// Sets a new calculation method for this composite indicator
        /// </summary>
        /// <param name="calculationMethod">The new calculation delegate</param>
        public void SetCalculationMethod(CalculationDelegate calculationMethod)
        {
            _calculationMethod = calculationMethod ?? throw new ArgumentNullException(nameof(calculationMethod));
        }

        /// <summary>
        /// Adds a new source indicator as a dependency
        /// </summary>
        /// <param name="indicatorId">The ID of the indicator to add</param>
        public void AddSourceIndicator(string indicatorId)
        {
            if (string.IsNullOrWhiteSpace(indicatorId))
                throw new ArgumentNullException(nameof(indicatorId));

            if (!_dependencies.Contains(indicatorId))
            {
                AddDependency(indicatorId);
            }
        }

        /// <summary>
        /// Removes a source indicator from this composite's dependencies
        /// </summary>
        /// <param name="indicatorId">The ID of the indicator to remove</param>
        public bool RemoveSourceIndicator(string indicatorId)
        {
            return _dependencies.Remove(indicatorId);
        }
        
        /// <summary>
        /// Returns a string representation of this composite indicator
        /// </summary>
        /// <returns>A string description of the indicator and its dependencies</returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Composite Indicator: {Name}");
            sb.AppendLine($"Description: {Description}");
            sb.AppendLine($"Category: {Category}");
            sb.AppendLine("Dependencies:");
            
            foreach (var dependency in _dependencies)
            {
                sb.AppendLine($"  - {dependency}");
            }
            
            return sb.ToString();
        }
    }
}