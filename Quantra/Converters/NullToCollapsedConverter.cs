using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace Quantra.Converters
{
    /// <summary>
    /// Converter that returns Collapsed for null values and Visible for non-null values
    /// </summary>
    public class NullToCollapsedConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value == null ? Visibility.Collapsed : Visibility.Visible;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}