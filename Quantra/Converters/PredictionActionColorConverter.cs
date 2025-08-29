using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts prediction action values to colors
    /// BUY actions = green, SELL actions = red, HOLD/neutral = gray
    /// </summary>
    public class PredictionActionColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null)
                return new SolidColorBrush(Colors.White);

            string action = value.ToString()?.ToUpperInvariant().Trim();

            return action switch
            {
                "BUY" or "STRONG BUY" => Brushes.Green,
                "SELL" or "STRONG SELL" => Brushes.Red,
                "HOLD" or "NEUTRAL" => Brushes.Gray,
                _ => Brushes.White
            };
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}