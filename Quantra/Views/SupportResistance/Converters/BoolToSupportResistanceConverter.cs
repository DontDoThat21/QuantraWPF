using System;
using System.Globalization;
using System.Windows.Data;

namespace Quantra.Views.SupportResistance.Converters
{
    /// <summary>
    /// Converts a boolean value to a support/resistance text or empty string
    /// </summary>
    public class BoolToSupportResistanceConverter : IValueConverter
    {
        /// <summary>
        /// The type text to display (Support or Resistance)
        /// </summary>
        public string TypeText { get; set; }
        
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool isType && isType)
            {
                return TypeText;
            }
            return string.Empty;
        }
        
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}