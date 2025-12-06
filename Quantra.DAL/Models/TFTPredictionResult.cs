using System;
using System.Collections.Generic;

namespace Quantra.DAL.Models
{
    /// <summary>
    /// TFT (Temporal Fusion Transformer) prediction result with multi-horizon forecasts and uncertainty quantification.
    /// Based on the paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    /// https://arxiv.org/abs/1912.09363
    /// </summary>
    public class TFTPredictionResult
    {
        /// <summary>
        /// Stock symbol for this prediction.
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Recommended trading action: "BUY", "SELL", or "HOLD".
        /// </summary>
        public string Action { get; set; }

        /// <summary>
        /// Overall confidence score for the prediction (0.0 to 1.0).
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Current price of the stock at prediction time.
        /// </summary>
        public double CurrentPrice { get; set; }

        /// <summary>
        /// Target price prediction (median of quantile predictions for nearest horizon).
        /// </summary>
        public double TargetPrice { get; set; }

        /// <summary>
        /// Lower bound of the prediction confidence interval (10th percentile).
        /// </summary>
        public double LowerBound { get; set; }

        /// <summary>
        /// Upper bound of the prediction confidence interval (90th percentile).
        /// </summary>
        public double UpperBound { get; set; }

        /// <summary>
        /// Uncertainty measure from TFT model (difference between upper and lower bounds).
        /// Higher values indicate less certain predictions.
        /// </summary>
        public double Uncertainty { get; set; }

        /// <summary>
        /// Multi-horizon predictions (5, 10, 20, 30 days ahead).
        /// Each horizon includes median, lower, and upper bounds.
        /// Key format: "5d", "10d", "20d", "30d"
        /// </summary>
        public Dictionary<string, HorizonPredictionData> Horizons { get; set; }

        /// <summary>
        /// Feature importance weights from TFT variable selection networks.
        /// Shows which features contributed most to the prediction.
        /// </summary>
        public Dictionary<string, double> FeatureWeights { get; set; }

        /// <summary>
        /// Temporal attention weights showing which past time steps were most influential.
        /// Key is the relative time offset (e.g., -1 for yesterday, -5 for 5 days ago).
        /// </summary>
        public Dictionary<int, double> TemporalAttention { get; set; }

        /// <summary>
        /// Error message if prediction failed, null otherwise.
        /// </summary>
        public string Error { get; set; }

        /// <summary>
        /// Timestamp when the prediction was generated.
        /// </summary>
        public DateTime PredictionTimestamp { get; set; } = DateTime.Now;

        /// <summary>
        /// Model version or checkpoint used for prediction.
        /// </summary>
        public string ModelVersion { get; set; }

        /// <summary>
        /// Time taken for inference in milliseconds.
        /// </summary>
        public double InferenceTimeMs { get; set; }

        /// <summary>
        /// Indicates if the prediction was successful.
        /// </summary>
        public bool Success => string.IsNullOrEmpty(Error);

        /// <summary>
        /// Calculated potential return based on target price vs current price.
        /// </summary>
        public double PotentialReturn => CurrentPrice > 0 ? (TargetPrice - CurrentPrice) / CurrentPrice : 0;

        /// <summary>
        /// Default constructor.
        /// </summary>
        public TFTPredictionResult()
        {
            Horizons = new Dictionary<string, HorizonPredictionData>();
            FeatureWeights = new Dictionary<string, double>();
            TemporalAttention = new Dictionary<int, double>();
        }
    }

    /// <summary>
    /// Prediction data for a specific time horizon from TFT model.
    /// </summary>
    public class HorizonPredictionData
    {
        /// <summary>
        /// Median predicted price (50th percentile).
        /// </summary>
        public double MedianPrice { get; set; }

        /// <summary>
        /// Lower bound of prediction interval (10th percentile).
        /// </summary>
        public double LowerBound { get; set; }

        /// <summary>
        /// Upper bound of prediction interval (90th percentile).
        /// </summary>
        public double UpperBound { get; set; }

        /// <summary>
        /// 25th percentile prediction.
        /// </summary>
        public double Q25 { get; set; }

        /// <summary>
        /// 75th percentile prediction.
        /// </summary>
        public double Q75 { get; set; }

        /// <summary>
        /// Confidence level for this horizon's prediction.
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Calculated target price at this horizon.
        /// </summary>
        public double TargetPrice { get; set; }
    }

    /// <summary>
    /// Multi-horizon prediction for display in UI.
    /// </summary>
    public class HorizonPrediction
    {
        /// <summary>
        /// Horizon label (e.g., "5 Days", "10 Days").
        /// </summary>
        public string Horizon { get; set; }

        /// <summary>
        /// Median predicted price.
        /// </summary>
        public double MedianPrice { get; set; }

        /// <summary>
        /// Lower bound of prediction interval.
        /// </summary>
        public double LowerBound { get; set; }

        /// <summary>
        /// Upper bound of prediction interval.
        /// </summary>
        public double UpperBound { get; set; }

        /// <summary>
        /// Confidence level for this prediction.
        /// </summary>
        public double Confidence { get; set; }
    }
}
