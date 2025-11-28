using System.Collections.Generic;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for generating plain English explanations of ML model predictions.
    /// Used by Market Chat to help traders understand model reasoning without reading code (MarketChat story 4).
    /// </summary>
    public interface IModelExplainerService
    {
        /// <summary>
        /// Generates a comprehensive explanation of the factors driving a prediction.
        /// Translates technical indicators, feature weights, and risk metrics into plain English.
        /// </summary>
        /// <param name="prediction">The prediction result containing feature weights and metrics</param>
        /// <returns>A detailed plain English explanation of the prediction factors</returns>
        string ExplainPredictionFactors(PredictionResult prediction);

        /// <summary>
        /// Generates a plain English explanation of the confidence score.
        /// </summary>
        /// <param name="confidence">The confidence score (0.0 to 1.0)</param>
        /// <param name="modelType">The type of ML model used</param>
        /// <returns>A plain English explanation of the confidence level</returns>
        string ExplainConfidenceScore(double confidence, string modelType);

        /// <summary>
        /// Generates a plain English explanation of risk metrics.
        /// </summary>
        /// <param name="riskMetrics">The risk metrics from the prediction</param>
        /// <returns>A plain English explanation of the risk assessment</returns>
        string ExplainRiskMetrics(RiskMetrics riskMetrics);

        /// <summary>
        /// Translates a technical indicator name and its weight into plain English.
        /// </summary>
        /// <param name="indicatorName">The name of the technical indicator (e.g., "RSI", "MACD", "VWAP")</param>
        /// <param name="weight">The feature importance weight (0.0 to 1.0)</param>
        /// <param name="value">The optional current value of the indicator</param>
        /// <returns>A plain English explanation of the indicator's influence</returns>
        string TranslateIndicatorToPlainEnglish(string indicatorName, double weight, double? value = null);

        /// <summary>
        /// Compares predictions across different model types and generates an explanation.
        /// </summary>
        /// <param name="predictions">Dictionary of model type to prediction result</param>
        /// <returns>A plain English comparison of the different model predictions</returns>
        string CompareModelPredictions(Dictionary<string, PredictionResult> predictions);

        /// <summary>
        /// Gets the top N most influential factors from the feature weights.
        /// </summary>
        /// <param name="featureWeights">Dictionary of feature names to their weights</param>
        /// <param name="topN">Number of top factors to return</param>
        /// <returns>A formatted string describing the top influential factors</returns>
        string GetTopInfluentialFactors(Dictionary<string, double> featureWeights, int topN = 3);
    }
}
