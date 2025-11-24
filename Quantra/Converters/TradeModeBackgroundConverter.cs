using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    public class TradeModeBackgroundConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool isPaperTrade)
            {
                return isPaperTrade
                    ? new SolidColorBrush(Color.FromRgb(45, 106, 76)) // Dark green
                    : new SolidColorBrush(Color.FromRgb(106, 45, 45)); // Dark red
            }

            return new SolidColorBrush(Colors.Transparent);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
