using System;
using System.Globalization;
using System.Windows.Data;

namespace Quantra.Converters
{
    public class TradeModeTextConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool isPaperTrade)
            {
                return isPaperTrade ? "Paper Trade" : "Real Trade";
            }

            return "Unknown";
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string text)
            {
                return text.Equals("Paper Trade", StringComparison.OrdinalIgnoreCase);
            }

            return false;
        }
    }
}
