using System;
using System.Globalization;
using System.Windows.Data;

namespace Quantra.Converters
{
    public class BoolToStringConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            bool boolValue = (bool)value;
            string paramString = parameter as string;

            if (paramString != null && paramString.Contains("|"))
            {
                string[] options = paramString.Split('|');
                return boolValue ? options[0] : options[1];
            }

            return boolValue ? "True" : "False";
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
