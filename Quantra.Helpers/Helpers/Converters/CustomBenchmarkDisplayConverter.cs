using System;
using System.Globalization;
using System.Windows.Data;
using Quantra.Models;

namespace Quantra.Converters
{
    /// <summary>
    /// Converter to ensure proper display of CustomBenchmark objects in ComboBox
    /// </summary>
    public class CustomBenchmarkDisplayConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is CustomBenchmark benchmark)
            {
                // Return a formatted string with Name and DisplaySymbol
                return $"{benchmark.Name} ({benchmark.DisplaySymbol})";
            }
            
            // Fallback to string representation
            return value?.ToString() ?? "Unknown Benchmark";
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}