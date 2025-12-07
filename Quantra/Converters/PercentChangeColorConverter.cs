using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts percent change to color
    /// Green for positive changes, red for negative changes, white for zero
    /// </summary>
    public class PercentChangeColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null || !double.TryParse(value.ToString(), out double change))
                return new SolidColorBrush(Colors.White);

            if (change > 0)
            {
                // Positive change - green
                return Brushes.Green;
            }
            else if (change < 0)
            {
                // Negative change - red
                return Brushes.Red;
            }
            else
            {
                // Zero change - white
                return new SolidColorBrush(Colors.White);
            }
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}