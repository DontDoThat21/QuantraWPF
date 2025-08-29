using System;
using System.Collections.Generic;
using LiveCharts.Defaults; // Add OhlcPoint reference 

namespace Quantra.Models
{
    public class StockData
    {
        public List<double> Prices { get; set; } = new List<double>();
        public List<double> UpperBand { get; set; } = new List<double>();
        public List<double> MiddleBand { get; set; } = new List<double>();
        public List<double> LowerBand { get; set; } = new List<double>();
        public List<double> RSI { get; set; } = new List<double>();
        public List<DateTime> Dates { get; set; } = new List<DateTime>();
        // Add OHLC data for candle chart
        public List<OhlcPoint> CandleData { get; set; } = new List<OhlcPoint>();
        // Add Volumes property for volume data
        public List<double> Volumes { get; set; } = new List<double>();

    }

    public class StockIndicator
    {
        public string Name { get; set; }
        public string Value { get; set; }
        public string Description { get; set; }
    }
}
