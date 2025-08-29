using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts timestamp to color that fades to beige as data gets older
    /// White for recent data, fading to beige for data >= 1 day old
    /// </summary>
    public class TimestampColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null || !(value is DateTime timestamp))
                return new SolidColorBrush(Colors.White);

            var now = DateTime.Now;
            var age = now - timestamp;
            
            // Calculate age in hours
            double ageInHours = age.TotalHours;
            
            // If data is less than 1 hour old, keep it white
            if (ageInHours < 1.0)
                return new SolidColorBrush(Colors.White);
            
            // If data is more than 24 hours old, make it beige
            if (ageInHours >= 24.0)
                return new SolidColorBrush(Color.FromRgb(245, 245, 220)); // Beige
            
            // Fade from white to beige based on age (1-24 hours)
            double fadeRatio = (ageInHours - 1.0) / 23.0; // 0.0 to 1.0
            fadeRatio = Math.Min(Math.Max(fadeRatio, 0.0), 1.0); // Ensure bounds
            
            // Interpolate between white (255,255,255) and beige (245,245,220)
            byte red = (byte)(255 - (10 * fadeRatio));
            byte green = (byte)(255 - (10 * fadeRatio));
            byte blue = (byte)(255 - (35 * fadeRatio));
            
            return new SolidColorBrush(Color.FromRgb(red, green, blue));
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}