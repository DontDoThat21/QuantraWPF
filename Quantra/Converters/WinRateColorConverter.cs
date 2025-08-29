using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    public class WinRateColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null)
                return new SolidColorBrush(Colors.White);

            double winRate;
            
            if (value is double doubleValue)
            {
                winRate = doubleValue;
            }
            else if (double.TryParse(value.ToString(), out double parsedValue))
            {
                winRate = parsedValue;
            }
            else
            {
                return new SolidColorBrush(Colors.White);
            }

            // Color based on win rate percentage
            if (winRate >= 0.7) // 70%+ is excellent
                return new SolidColorBrush(Color.FromRgb(50, 205, 50)); // LimeGreen
            else if (winRate >= 0.5) // 50-70% is good
                return new SolidColorBrush(Color.FromRgb(46, 139, 87)); // SeaGreen
            else if (winRate >= 0.4) // 40-50% is average
                return new SolidColorBrush(Color.FromRgb(255, 165, 0)); // Orange
            else // Below 40% is poor
                return new SolidColorBrush(Color.FromRgb(220, 20, 60)); // Crimson
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
