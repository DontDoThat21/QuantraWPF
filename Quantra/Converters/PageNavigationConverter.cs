using System;
using System.Globalization;
using System.Windows.Data;

namespace Quantra.Converters
{
    /// <summary>
    /// Converter for handling page navigation button enabled states
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
}
