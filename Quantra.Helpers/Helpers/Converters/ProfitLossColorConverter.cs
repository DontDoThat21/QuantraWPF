using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    public class ProfitLossColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null)
                return new SolidColorBrush(Colors.White);

            double number;
            
            if (value is double doubleValue)
            {
                number = doubleValue;
            }
            else if (double.TryParse(value.ToString(), out double parsedValue))
            {
                number = parsedValue;
            }
            else
            {
                return new SolidColorBrush(Colors.White);
            }

            // Return green for positive values, red for negative, white for zero
            if (number > 0)
                return new SolidColorBrush(Color.FromRgb(50, 205, 50)); // LimeGreen
            else if (number < 0)
                return new SolidColorBrush(Color.FromRgb(220, 20, 60)); // Crimson
            else
                return new SolidColorBrush(Colors.White);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
