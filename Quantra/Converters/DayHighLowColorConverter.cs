using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts Day High/Low values to colors based on performance relative to current price
    /// For Day High: Green when current price is close to day high (performing well)
    /// For Day Low: Red when current price is close to day low (performing poorly)
    /// </summary>
    public class DayHighLowColorConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            if (values == null || values.Length < 3)
                return new SolidColorBrush(Colors.White);

            if (!double.TryParse(values[0]?.ToString(), out double currentPrice) ||
                !double.TryParse(values[1]?.ToString(), out double dayHigh) ||
                !double.TryParse(values[2]?.ToString(), out double dayLow))
                return new SolidColorBrush(Colors.White);

            // Get the column type from parameter
            string columnType = parameter?.ToString()?.ToUpperInvariant();
            
            // Avoid invalid scenarios
            if (dayHigh <= dayLow || currentPrice <= 0)
                return new SolidColorBrush(Colors.White);
            
            // Calculate the range and position
            double range = dayHigh - dayLow;
            double positionFromLow = currentPrice - dayLow;
            double positionRatio = positionFromLow / range; // 0.0 = at low, 1.0 = at high
            
            if (columnType == "HIGH")
            {
                // For Day High column: Green when price is close to the high
                // Exponential intensity based on how close to high
                double distanceFromHigh = Math.Abs(dayHigh - currentPrice) / range;
                double intensity = Math.Pow(1.0 - distanceFromHigh, 3.0); // Exponential curve
                
                if (positionRatio > 0.8) // Only color when price is in top 20% of range
                {
                    byte greenComponent = (byte)(155 + (100 * intensity)); // Range from 155 to 255
                    return new SolidColorBrush(Color.FromRgb(50, greenComponent, 50));
                }
            }
            else if (columnType == "LOW")
            {
                // For Day Low column: Red when price is close to the low
                // Exponential intensity based on how close to low
                double distanceFromLow = Math.Abs(currentPrice - dayLow) / range;
                double intensity = Math.Pow(1.0 - distanceFromLow, 3.0); // Exponential curve
                
                if (positionRatio < 0.2) // Only color when price is in bottom 20% of range
                {
                    byte redComponent = (byte)(155 + (100 * intensity)); // Range from 155 to 255
                    return new SolidColorBrush(Color.FromRgb(redComponent, 20, 60));
                }
            }
            
            return new SolidColorBrush(Colors.White);
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}