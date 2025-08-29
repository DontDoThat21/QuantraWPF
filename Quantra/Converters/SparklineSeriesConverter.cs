using System;
using System.Globalization;
using System.Windows.Data;
using LiveCharts;
using LiveCharts.Wpf;
using System.Collections.Generic;
using System.Linq;

namespace Quantra.Converters
{
    public class SparklineSeriesConverter : IValueConverter
    {
        // If this converter is used to generate series for a chart, ensure it provides X values as DateTime or formatted date strings

        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var closes = value as IEnumerable<double>;
            if (closes == null)
                return null;

            var series = new SeriesCollection
            {
                new LineSeries
                {
                    Values = new ChartValues<double>(closes),
                    StrokeThickness = 2,
                    Fill = System.Windows.Media.Brushes.Transparent,
                    PointGeometry = null
                }
            };
            return series;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
