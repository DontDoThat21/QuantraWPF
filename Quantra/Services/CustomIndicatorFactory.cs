using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Quantra.Models;
using System.Linq;

namespace Quantra.Services
{
    /// <summary>
    /// Factory for creating indicator objects from their definitions
    /// </summary>
    public static class CustomIndicatorFactory
    {
        /// <summary>
        /// Create a CustomIndicator instance from a definition
        /// </summary>
        /// <param name="definition">The indicator definition</param>
        /// <returns>The created indicator instance</returns>
        public static CustomIndicator CreateIndicator(CustomIndicatorDefinition definition)
        {
            if (definition == null)
                throw new ArgumentNullException(nameof(definition));

            // Handle different indicator types
            switch (definition.IndicatorType)
            {
                case "Composite":
                    return CreateCompositeIndicator(definition);
                case "StandardTemplate":
                    return CreateStandardTemplateIndicator(definition);
                case "Custom":
                    return CreateCustomCodeIndicator(definition);
                default:
                    throw new NotSupportedException($"Indicator type '{definition.IndicatorType}' is not supported");
            }
        }

        /// <summary>
        /// Create a composite indicator from a definition
        /// </summary>
        private static CompositeIndicator CreateCompositeIndicator(CustomIndicatorDefinition definition)
        {
            // Create the calculation delegate based on the formula
            var calculationDelegate = BuildCalculationDelegate(definition.Formula);
            
            // Create the composite indicator
            var indicator = new CompositeIndicator(
                definition.Name,
                calculationDelegate,
                definition.Dependencies,
                definition.OutputKeys.Count > 0 ? definition.OutputKeys : new List<string> { "Value" },
                definition.Category,
                definition.Description);
                
            // Set up parameters
            foreach (var paramDef in definition.Parameters.Values)
            {
                var param = ConvertToIndicatorParameter(paramDef);
                indicator.Parameters[param.Name] = param;
            }
            
            return indicator;
        }
        
        /// <summary>
        /// Create a standard template indicator from definition
        /// </summary>
        private static CustomIndicator CreateStandardTemplateIndicator(CustomIndicatorDefinition definition)
        {
            // Here we would instantiate built-in templates like EMA, SMA, etc.
            // For now, we'll just create a placeholder implementation
            // In a real implementation, we would have a registry of template types
            switch (definition.Name.ToLowerInvariant())
            {
                case "sma":
                case "simple moving average":
                    return new SimpleMovingAverageIndicator
                    {
                        Name = definition.Name,
                        Description = definition.Description,
                        Category = definition.Category
                        // Set parameters from definition
                    };
                case "ema":
                case "exponential moving average":
                    return new ExponentialMovingAverageIndicator
                    {
                        Name = definition.Name,
                        Description = definition.Description,
                        Category = definition.Category
                        // Set parameters from definition
                    };
                default:
                    throw new NotSupportedException($"Template indicator '{definition.Name}' is not supported");
            }
        }
        
        /// <summary>
        /// Create a custom code indicator from the definition
        /// </summary>
        private static CustomIndicator CreateCustomCodeIndicator(CustomIndicatorDefinition definition)
        {
            // In a real implementation, we would compile and load the custom code
            // For now, we'll just throw an exception
            throw new NotImplementedException("Custom code indicators are not implemented yet");
        }
        
        /// <summary>
        /// Build a calculation delegate from a formula
        /// </summary>
        private static CompositeIndicator.CalculationDelegate BuildCalculationDelegate(string formula)
        {
            // This is a simple implementation that would need to be expanded
            // In a real implementation, we would parse the formula and build a calculation tree
            return (inputs) =>
            {
                // Simple parser for basic operations
                // Format: indicator1.value [operator] indicator2.value
                // Example: "RSI.Value * 0.5 + MACD.Value * 0.5"
                
                try
                {
                    // This is a very simplified implementation
                    // In a real implementation, you would need a proper expression parser
                    var parts = formula.Split(new[] { '+', '-', '*', '/' }, StringSplitOptions.RemoveEmptyEntries);
                    var operators = formula.Where(c => c == '+' || c == '-' || c == '*' || c == '/').ToArray();
                    
                    if (parts.Length != operators.Length + 1)
                        throw new FormatException("Invalid formula format");
                    
                    double result = GetValueFromPart(parts[0].Trim(), inputs);
                    
                    for (int i = 0; i < operators.Length; i++)
                    {
                        var rightValue = GetValueFromPart(parts[i + 1].Trim(), inputs);
                        
                        switch (operators[i])
                        {
                            case '+':
                                result += rightValue;
                                break;
                            case '-':
                                result -= rightValue;
                                break;
                            case '*':
                                result *= rightValue;
                                break;
                            case '/':
                                if (Math.Abs(rightValue) < 0.000001)
                                    throw new DivideByZeroException("Division by zero in formula");
                                result /= rightValue;
                                break;
                        }
                    }
                    
                    return result;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Error evaluating formula: {formula}", ex);
                }
            };
        }
        
        /// <summary>
        /// Helper method to get a value from a formula part
        /// </summary>
        private static double GetValueFromPart(string part, Dictionary<string, Dictionary<string, double>> inputs)
        {
            // Check if it's a numeric constant
            if (double.TryParse(part, out var numericValue))
                return numericValue;
                
            // Otherwise it should be in the format "indicator.output"
            var dotIndex = part.IndexOf('.');
            if (dotIndex <= 0)
                throw new FormatException($"Invalid formula part: {part}");
                
            var indicatorId = part.Substring(0, dotIndex);
            var outputKey = part.Substring(dotIndex + 1);
            
            if (!inputs.TryGetValue(indicatorId, out var indicatorValues))
                throw new KeyNotFoundException($"Indicator '{indicatorId}' not found in inputs");
                
            if (!indicatorValues.TryGetValue(outputKey, out var value))
                throw new KeyNotFoundException($"Output key '{outputKey}' not found in indicator '{indicatorId}'");
                
            return value;
        }
        
        /// <summary>
        /// Convert an IndicatorParameterDefinition to an IndicatorParameter
        /// </summary>
        private static IndicatorParameter ConvertToIndicatorParameter(IndicatorParameterDefinition definition)
        {
            var type = GetType(definition.ParameterType);
            
            var param = new IndicatorParameter
            {
                Name = definition.Name,
                Description = definition.Description,
                ParameterType = type,
                IsOptional = definition.IsOptional,
                Options = definition.Options
            };
            
            // Convert values to the appropriate type
            if (definition.DefaultValue != null)
            {
                param.DefaultValue = Convert.ChangeType(definition.DefaultValue, type);
                param.Value = param.DefaultValue;
            }
            
            if (definition.Value != null)
            {
                param.Value = Convert.ChangeType(definition.Value, type);
            }
            
            if (definition.MinValue != null)
            {
                param.MinValue = Convert.ChangeType(definition.MinValue, type);
            }
            
            if (definition.MaxValue != null)
            {
                param.MaxValue = Convert.ChangeType(definition.MaxValue, type);
            }
            
            return param;
        }
        
        /// <summary>
        /// Get a Type from a string type name
        /// </summary>
        private static Type GetType(string typeName)
        {
            switch (typeName.ToLowerInvariant())
            {
                case "int":
                case "integer":
                    return typeof(int);
                case "double":
                case "float":
                case "decimal":
                    return typeof(double);
                case "bool":
                case "boolean":
                    return typeof(bool);
                case "string":
                    return typeof(string);
                case "datetime":
                    return typeof(DateTime);
                default:
                    throw new NotSupportedException($"Type '{typeName}' is not supported");
            }
        }
    }

    /// <summary>
    /// Example indicator class for Simple Moving Average
    /// </summary>
    public class SimpleMovingAverageIndicator : CustomIndicator
    {
        public SimpleMovingAverageIndicator()
        {
            Category = "Moving Averages";
            Description = "Simple Moving Average";
            
            // Register parameters
            RegisterIntParameter("Period", "Number of periods to average", 14, 1, 500);
            RegisterSelectionParameter("PriceType", "Price type to use for calculation", "Close", 
                new[] { "Open", "High", "Low", "Close", "Volume", "HLC3", "OHLC4" });
        }
        
        public override async Task<Dictionary<string, double>> CalculateAsync(List<HistoricalPrice> historicalData)
        {
            ValidateParameters();
            
            if (historicalData == null || historicalData.Count == 0)
                return new Dictionary<string, double> { { "Value", 0 } };
                
            int period = (int)Parameters["Period"].Value;
            string priceType = (string)Parameters["PriceType"].Value;
            
            if (historicalData.Count < period)
                return new Dictionary<string, double> { { "Value", 0 } };
                
            // Calculate SMA
            double sum = 0;
            for (int i = historicalData.Count - period; i < historicalData.Count; i++)
            {
                sum += GetPrice(historicalData[i], priceType);
            }
            
            double sma = sum / period;
            return new Dictionary<string, double> { { "Value", sma } };
        }
        
        private double GetPrice(HistoricalPrice price, string priceType)
        {
            return priceType switch
            {
                "Open" => price.Open,
                "High" => price.High,
                "Low" => price.Low,
                "Close" => price.Close,
                "Volume" => price.Volume,
                "HLC3" => (price.High + price.Low + price.Close) / 3,
                "OHLC4" => (price.Open + price.High + price.Low + price.Close) / 4,
                _ => price.Close
            };
        }
    }
    
    /// <summary>
    /// Example indicator class for Exponential Moving Average
    /// </summary>
    public class ExponentialMovingAverageIndicator : CustomIndicator
    {
        public ExponentialMovingAverageIndicator()
        {
            Category = "Moving Averages";
            Description = "Exponential Moving Average";
            
            // Register parameters
            RegisterIntParameter("Period", "Number of periods for EMA calculation", 14, 1, 500);
            RegisterSelectionParameter("PriceType", "Price type to use for calculation", "Close", 
                new[] { "Open", "High", "Low", "Close", "Volume", "HLC3", "OHLC4" });
        }
        
        public override async Task<Dictionary<string, double>> CalculateAsync(List<HistoricalPrice> historicalData)
        {
            ValidateParameters();
            
            if (historicalData == null || historicalData.Count == 0)
                return new Dictionary<string, double> { { "Value", 0 } };
                
            int period = (int)Parameters["Period"].Value;
            string priceType = (string)Parameters["PriceType"].Value;
            
            if (historicalData.Count < period)
                return new Dictionary<string, double> { { "Value", 0 } };
                
            // Calculate EMA
            double multiplier = 2.0 / (period + 1);
            
            // First, calculate SMA as the first EMA value
            double sum = 0;
            for (int i = 0; i < period; i++)
            {
                sum += GetPrice(historicalData[i], priceType);
            }
            double ema = sum / period;
            
            // Then calculate EMA for the remaining periods
            for (int i = period; i < historicalData.Count; i++)
            {
                double price = GetPrice(historicalData[i], priceType);
                ema = (price - ema) * multiplier + ema;
            }
            
            return new Dictionary<string, double> { { "Value", ema } };
        }
        
        private double GetPrice(HistoricalPrice price, string priceType)
        {
            return priceType switch
            {
                "Open" => price.Open,
                "High" => price.High,
                "Low" => price.Low,
                "Close" => price.Close,
                "Volume" => price.Volume,
                "HLC3" => (price.High + price.Low + price.Close) / 3,
                "OHLC4" => (price.Open + price.High + price.Low + price.Close) / 4,
                _ => price.Close
            };
        }
    }
}