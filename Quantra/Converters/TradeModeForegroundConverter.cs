using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    public class TradeModeForegroundConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool isPaperTrade)
            {
                return isPaperTrade
                    ? new SolidColorBrush(Color.FromRgb(80, 224, 112)) // Light green
                    : new SolidColorBrush(Color.FromRgb(224, 80, 80));  // Light red
            }

            return new SolidColorBrush(Colors.White); // Default
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
