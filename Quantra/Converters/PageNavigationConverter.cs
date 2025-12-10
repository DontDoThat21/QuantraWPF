using System;
using System.Globalization;
using System.Windows.Data;

namespace Quantra.Converters
{
    /// <summary>
    /// Converter for handling page navigation button enabled states (single value)
    /// </summary>
    public class PageNavigationConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int currentPage && parameter is string direction)
            {
                if (direction == "Previous")
                {
                    // Previous button enabled if current page > 1
                    return currentPage > 1;
                }
                // For "Next" and other directions, let the HasMorePages binding handle it
                return true;
            }
            return false;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Multi-value converter for handling page navigation button enabled states with filtering support
    /// Takes navigation state (CurrentPage or HasMorePages) and IsPaginationEnabled flag
    /// </summary>
    public class PageNavigationMultiConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            if (values == null || values.Length < 2 || parameter is not string direction)
                return false;

            // First value is the navigation state (CurrentPage or HasMorePages)
            // Second value is IsPaginationEnabled (should be true when not filtering)
            var isPaginationEnabled = values[1] is bool enabled && enabled;
            
            if (!isPaginationEnabled)
                return false; // Disable pagination when filtering is active

            if (direction == "Previous")
            {
                // Previous button enabled if current page > 1 AND pagination is enabled
                return values[0] is int currentPage && currentPage > 1;
            }
            else if (direction == "Next")
            {
                // Next button enabled if HasMorePages is true AND pagination is enabled
                return values[0] is bool hasMorePages && hasMorePages;
            }

            return false;
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
