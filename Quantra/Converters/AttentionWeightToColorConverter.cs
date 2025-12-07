using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts attention weight (0-1) to a color for heatmap visualization
    /// Higher weights = darker/more intense colors
    /// </summary>
    public class AttentionWeightToColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is double weight && weight >= 0)
            {
                // Normalize weight to 0-1 range if needed
                double normalizedWeight = Math.Min(1.0, Math.Max(0.0, weight));

                // Create gradient from light yellow to dark orange/red
                // Low attention: Light yellow (#FFFFCC)
                // High attention: Dark orange (#FF4500)

                byte r = (byte)(255);
                byte g = (byte)(255 - (normalizedWeight * 186)); // 255 -> 69
                byte b = (byte)(204 - (normalizedWeight * 204)); // 204 -> 0

                return new SolidColorBrush(Color.FromRgb(r, g, b));
            }

            return Brushes.Transparent;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Converts attention weight to cell opacity for emphasis
    /// </summary>
    public class AttentionWeightToOpacityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is double weight)
            {
                // Map weight to opacity: 0.3 (low) to 1.0 (high)
                double opacity = 0.3 + (weight * 0.7);
                return Math.Min(1.0, Math.Max(0.3, opacity));
            }

            return 0.5;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
