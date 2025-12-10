using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;
using Quantra.Models;

namespace Quantra.Converters
{
    /// <summary>
    /// Converter that returns background color based on option moneyness (ITM/ATM/OTM)
    /// </summary>
    public class OptionMoneynessColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is OptionData option)
            {
                if (option.IsITM)
                    return new SolidColorBrush(Color.FromRgb(0x2D, 0x50, 0x16)); // Green for ITM
                else if (option.IsATM)
                    return new SolidColorBrush(Color.FromRgb(0x4D, 0x4D, 0x00)); // Yellow for ATM
                else
                    return new SolidColorBrush(Color.FromRgb(0x1E, 0x1E, 0x1E)); // Default dark for OTM
            }

            return new SolidColorBrush(Color.FromRgb(0x1E, 0x1E, 0x1E));
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Converter that returns foreground color for high/low values
    /// </summary>
    public class GreekValueColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is double doubleValue)
            {
                // Color code based on magnitude
                var absValue = Math.Abs(doubleValue);
                
                if (absValue > 0.7)
                    return Brushes.Red;
                else if (absValue > 0.4)
                    return Brushes.Yellow;
                else
                    return Brushes.Cyan;
            }

            return Brushes.White;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
