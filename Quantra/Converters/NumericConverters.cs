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
}