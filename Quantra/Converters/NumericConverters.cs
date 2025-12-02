using System;
using System.Globalization;
using System.Windows.Data;

namespace Quantra.Converters
{
    /// <summary>
    /// Converter that checks if a value is greater than a parameter
    /// </summary>
    public class GreaterThanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null || parameter == null)
                return false;

            double doubleValue;
            double compareValue;

            if (double.TryParse(value.ToString(), out doubleValue) && double.TryParse(parameter.ToString(), out compareValue))
            {
                return doubleValue > compareValue;
            }

            return false;
        }

        /// <summary>
        /// ConvertBack is not supported for this converter. This method will throw a NotImplementedException if called.
        /// </summary>
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Converter that checks if a value is less than a parameter
    /// </summary>
    public class LessThanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null || parameter == null)
                return false;

            double doubleValue;
            double compareValue;

            if (double.TryParse(value.ToString(), out doubleValue) && double.TryParse(parameter.ToString(), out compareValue))
            {
                return doubleValue < compareValue;
            }

            return false;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Converter that formats currency values with appropriate suffixes (K, M, B)
    /// </summary>
    public class CurrencyFormatConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null)
                return "$0";

            double amount;
            if (!double.TryParse(value.ToString(), out amount))
                return "$0";

            string sign = amount < 0 ? "-" : "";
            amount = Math.Abs(amount);

            if (amount >= 1_000_000_000)
                return $"{sign}${amount / 1_000_000_000:F1}B";
            else if (amount >= 1_000_000)
                return $"{sign}${amount / 1_000_000:F1}M";
            else if (amount >= 1_000)
                return $"{sign}${amount / 1_000:F0}K";
            else
                return $"{sign}${amount:F0}";
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Multi-value converter that calculates the width of a confidence bar
    /// based on confidence value (0-1) and container width
    /// </summary>
    public class ConfidenceWidthConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            if (values == null || values.Length < 2)
                return 0.0;

            if (values[0] == null || values[1] == null)
                return 0.0;

            // First value is confidence (0-1), second is container width
            if (double.TryParse(values[0].ToString(), out double confidence) &&
                double.TryParse(values[1].ToString(), out double containerWidth))
            {
                // Clamp confidence between 0 and 1
                confidence = Math.Max(0, Math.Min(1, confidence));
                
                // Calculate width as percentage of container
                double width = containerWidth * confidence;
                
                // Ensure minimum visible width for very small values
                return width > 0 ? Math.Max(width, 3) : 0;
            }

            return 0.0;
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}