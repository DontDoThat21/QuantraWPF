using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts RSI values to colors that fade to yellow as RSI moves outside 40-60 range
    /// Exponential intensity ramp, capping at RSI values of 25 or 75
    /// </summary>
    public class RsiColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null || !double.TryParse(value.ToString(), out double rsi))
                return new SolidColorBrush(Colors.White);

            // Special case: RSI = 0 should display as white
            if (rsi == 0)
                return new SolidColorBrush(Colors.White);

            // RSI is typically 0-100, neutral zone is 40-60
            const double neutralLow = 40.0;
            const double neutralHigh = 60.0;
            const double extremeLow = 25.0;
            const double extremeHigh = 75.0;

            // If RSI is in neutral zone (40-60), keep it white
            if (rsi >= neutralLow && rsi <= neutralHigh)
                return new SolidColorBrush(Colors.White);

            double intensity = 0.0;

            if (rsi < neutralLow)
            {
                // RSI below 40 - calculate intensity from 40 down to 25
                double range = neutralLow - extremeLow; // 15
                double distance = Math.Max(0, neutralLow - rsi);
                intensity = Math.Min(distance / range, 1.0);
            }
            else if (rsi > neutralHigh)
            {
                // RSI above 60 - calculate intensity from 60 up to 75
                double range = extremeHigh - neutralHigh; // 15
                double distance = Math.Max(0, rsi - neutralHigh);
                intensity = Math.Min(distance / range, 1.0);
            }

            // Apply exponential curve for more dramatic effect
            intensity = Math.Pow(intensity, 2.0);

            // Interpolate from white (255,255,255) to yellow (255,255,0)
            byte red = 255;
            byte green = 255;
            byte blue = (byte)(255 * (1.0 - intensity));

            return new SolidColorBrush(Color.FromRgb(red, green, blue));
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}