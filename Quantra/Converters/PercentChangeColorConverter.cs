using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows.Data;
using System.Windows.Media;

namespace Quantra.Converters
{
    /// <summary>
    /// Converts percent change to color based on intensity relative to all visible values
    /// Green for positive changes, red for negative changes, intensity based on relative magnitude
    /// </summary>
    public class PercentChangeColorConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            if (values == null || values.Length < 2)
                return new SolidColorBrush(Colors.White);

            // First value is the current percent change
            if (!double.TryParse(values[0]?.ToString(), out double currentChange))
                return new SolidColorBrush(Colors.White);

            // Second value should be the collection of all visible items
            var allItems = values[1] as IEnumerable;
            if (allItems == null)
                return new SolidColorBrush(Colors.White);

            // Extract all percent changes from the collection
            var allChanges = new List<double>();
            // Cache PropertyInfo for ChangePercent if items are homogeneous
            var firstItem = allItems.Cast<object>().FirstOrDefault();
            var changeProperty = firstItem?.GetType().GetProperty("ChangePercent");

            foreach (var item in allItems)
            {
                if (changeProperty != null)
                {
                    var changeValue = changeProperty.GetValue(item);
                    if (changeValue != null && double.TryParse(changeValue.ToString(), out double change))
                    {
                        allChanges.Add(change);
                    }
                }
            }

            if (allChanges.Count == 0)
                return new SolidColorBrush(Colors.White);

            // Calculate the range of changes
            double minChange = allChanges.Min();
            double maxChange = allChanges.Max();
            double range = maxChange - minChange;

            // If no range, return white
            if (range == 0)
                return new SolidColorBrush(Colors.White);

            // Calculate intensity based on position in the range
            double intensity;
            if (currentChange > 0)
            {
                // Positive change - calculate intensity relative to max positive
                intensity = maxChange > 0 ? Math.Min(currentChange / maxChange, 1.0) : 0.0;

                // Apply exponential curve for more dramatic effect
                intensity = Math.Pow(intensity, 1.5);

                // Green color with intensity
                byte greenComponent = (byte)(155 + (100 * intensity)); // Range from 155 to 255
                return new SolidColorBrush(Color.FromRgb(50, greenComponent, 50));
            }
            else if (currentChange < 0)
            {
                // Negative change - calculate intensity relative to max negative
                intensity = minChange < 0 ? Math.Min(Math.Abs(currentChange) / Math.Abs(minChange), 1.0) : 0.0;

                // Apply exponential curve for more dramatic effect
                intensity = Math.Pow(intensity, 1.5);

                // Red color with intensity
                byte redComponent = (byte)(155 + (100 * intensity)); // Range from 155 to 255
                return new SolidColorBrush(Color.FromRgb(redComponent, 20, 60));
            }
            else
            {
                // Zero change - white
                return new SolidColorBrush(Colors.White);
            }
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}