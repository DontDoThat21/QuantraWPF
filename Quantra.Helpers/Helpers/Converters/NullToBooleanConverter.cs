using System;
using System.Globalization;
using System.Windows.Data;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts a null/non-null object to a boolean value.
    /// Returns true when the value is not null, false otherwise.
    /// </summary>
    public class NullToBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value != null;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException("ConvertBack not implemented for NullToBooleanConverter");
        }
    }
}
