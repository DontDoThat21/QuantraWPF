using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    public class OrderStatusForegroundConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string status)
            {
                return status.ToLower() switch
                {
                    "executed" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#50E070")), // Green
                    "pending" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FFCC44")),  // Yellow
                    "failed" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E05050")),   // Red
                    "canceled" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#888888")), // Gray
                    "new" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#99CCFF")),      // Light blue
                    _ => new SolidColorBrush(Colors.White)                                                 // Default
                };
            }

            return new SolidColorBrush(Colors.White); // Default
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
