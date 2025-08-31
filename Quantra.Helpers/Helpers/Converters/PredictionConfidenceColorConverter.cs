using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts prediction confidence values to colors based on confidence level
    /// High confidence (>=0.8) = dark green, medium confidence (>=0.6) = green,
    /// low confidence (>=0.4) = orange, very low confidence (<0.4) = red
    /// </summary>
    public class PredictionConfidenceColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null || !double.TryParse(value.ToString(), out double confidence))
                return new SolidColorBrush(Colors.White);

            // Confidence is typically 0.0 to 1.0
            return confidence switch
            {
                >= 0.8 => Brushes.DarkGreen,
                >= 0.6 => Brushes.Green,
                >= 0.4 => Brushes.Orange,
                _ => Brushes.Red
            };
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}