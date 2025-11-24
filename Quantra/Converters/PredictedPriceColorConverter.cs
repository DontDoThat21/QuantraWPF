using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts predicted price to colors based on comparison with current price
    /// Green when predicted price is higher than current price (bullish),
    /// Red when predicted price is lower than current price (bearish),
    /// White when prices are very close (within 1%)
    /// </summary>
    public class PredictedPriceColorConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            if (values == null || values.Length < 2)
                return new SolidColorBrush(Colors.White);

            if (!double.TryParse(values[0]?.ToString(), out double predictedPrice) ||
                !double.TryParse(values[1]?.ToString(), out double currentPrice))
                return new SolidColorBrush(Colors.White);

            // Avoid division by zero
            if (currentPrice == 0 || predictedPrice == 0)
                return new SolidColorBrush(Colors.White);

            // Calculate percentage difference
            double percentDiff = (predictedPrice - currentPrice) / currentPrice;

            // Define threshold for "close" - within 1%
            const double closeThreshold = 0.01;

            if (Math.Abs(percentDiff) <= closeThreshold)
            {
                // Prices are close - white
                return new SolidColorBrush(Colors.White);
            }
            else if (percentDiff > 0)
            {
                // Predicted price higher than current - bullish (green)
                return Brushes.Green;
            }
            else
            {
                // Predicted price lower than current - bearish (red)
                return Brushes.Red;
            }
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}