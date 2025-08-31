using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    public class OrderStatusBackgroundConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string status)
            {
                return status.ToLower() switch
                {
                    "executed" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1D4A2C")), // Dark green
                    "pending" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#4A451D")),  // Dark yellow
                    "failed" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#4A1D1D")),   // Dark red
                    "canceled" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#333333")), // Dark gray
                    "new" => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#1D304A")),      // Dark blue
                    _ => new SolidColorBrush((Color)ColorConverter.ConvertFromString("#333344"))           // Default
                };
            }
            
            return new SolidColorBrush((Color)ColorConverter.ConvertFromString("#333344")); // Default
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
