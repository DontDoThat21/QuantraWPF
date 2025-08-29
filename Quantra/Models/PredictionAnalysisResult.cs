using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    public class PredictionAnalysisResult
    {
        public int Id { get; set; }
        public string Symbol { get; set; }
        public string PredictedAction { get; set; }
        public double Confidence { get; set; }
        public double CurrentPrice { get; set; }
        public double TargetPrice { get; set; }
        public double PotentialReturn { get; set; }
        public string TradingRule { get; set; }
        public DateTime AnalysisTime { get; set; }
        public Dictionary<string, double> Indicators { get; set; }
    }
}
