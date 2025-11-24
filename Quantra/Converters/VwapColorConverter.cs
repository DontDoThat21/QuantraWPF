using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts VWAP and Price values to colors based on price relative to VWAP
    /// Red when price is above VWAP, white when close, green when below
    /// </summary>
    public class VwapColorConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            if (values == null || values.Length < 2)
                return new SolidColorBrush(Colors.White);

            if (!double.TryParse(values[0]?.ToString(), out double price) ||
                !double.TryParse(values[1]?.ToString(), out double vwap))
                return new SolidColorBrush(Colors.White);

            // Avoid division by zero
            if (vwap == 0)
                return new SolidColorBrush(Colors.White);

            // Calculate percentage difference
            double percentDiff = (price - vwap) / vwap;

            // Define threshold for "close to VWAP" - within 0.5%
            const double closeThreshold = 0.005;

            if (Math.Abs(percentDiff) <= closeThreshold)
            {
                // Close to VWAP - white
                return new SolidColorBrush(Colors.White);
            }
            else if (percentDiff > 0)
            {
                // Price above VWAP - red intensity based on difference
                double intensity = Math.Min(Math.Abs(percentDiff) * 10, 1.0); // Scale and cap at 1.0
                byte redComponent = (byte)(155 + (65 * intensity)); // Range from 155 to 220
                return new SolidColorBrush(Color.FromRgb(redComponent, 20, 60));
            }
            else
            {
                // Price below VWAP - green intensity based on difference
                double intensity = Math.Min(Math.Abs(percentDiff) * 10, 1.0); // Scale and cap at 1.0
                byte greenComponent = (byte)(50 + (155 * intensity)); // Range from 50 to 205
                return new SolidColorBrush(Color.FromRgb(50, greenComponent, 50));
            }
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}