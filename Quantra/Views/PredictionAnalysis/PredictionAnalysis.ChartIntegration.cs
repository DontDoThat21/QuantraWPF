using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Controls;
using Quantra.DAL.Services;
using Quantra.Models;
using Quantra.DAL.Models;

namespace Quantra.Controls
{
    /// <summary>
    /// Integration methods for combining candlestick chart with TFT predictions
    /// </summary>
    public partial class PredictionAnalysis : UserControl
    {
        /// <summary>
        /// Update both the candlestick chart and TFT predictions for a given symbol
        /// This is the main integration point for displaying OHLCV data with multi-horizon forecasts
        /// </summary>
        /// <param name="symbol">Stock symbol to analyze</param>
        public async Task UpdateChartWithPredictionsAsync(string symbol)
        {
            try
            {
                if (string.IsNullOrEmpty(symbol))
                {
                    _loggingService?.Log("Warning", "Cannot update chart: symbol is null or empty");
                    return;
                }

                _loggingService?.Log("Info", $"Updating candlestick chart with predictions for {symbol}");

                // 1. Fetch historical OHLCV data
                var timeframe = "3month"; // Get 3 months of data for good context
                var historicalData = await _historicalDataService?.GetHistoricalPrices(symbol, timeframe, "1d");

                if (historicalData == null || historicalData.Count == 0)
                {
                    _loggingService?.Log("Warning", $"No historical data found for {symbol}");
                    ClearCandlestickChart();
                    return;
                }

                _loggingService?.Log("Info", $"Retrieved {historicalData.Count} days of historical data for {symbol}");

                // 2. Get TFT predictions if available
                TFTPredictionResult tftResult = null;
                try
                {
                    // Try to get TFT predictions from the RealTimeInferenceService
                    if (_realTimeInferenceService != null)
                    {
                        // Prepare historical data for TFT prediction
                        var indicators = await _indicatorService?.GetIndicatorsForPrediction(symbol, "1d");
                        
                        // Call TFT prediction service
                        // Note: This assumes TFTPredictionService is available through dependency injection
                        // You may need to add it to the constructor
                        var tftService = new TFTPredictionService(_loggingService, _alphaVantageService);
                        var tftServiceResult = await tftService.GetTFTPredictionsAsync(
                            symbol,
                            historicalData,
                            lookbackDays: 60,
                            futureHorizon: 30,
                            forecastHorizons: new List<int> { 5, 10, 20, 30 }
                        );

                        // Convert TFTResult to TFTPredictionResult
                        if (tftServiceResult != null && string.IsNullOrEmpty(tftServiceResult.Error))
                        {
                            tftResult = ConvertTFTResultToTFTPredictionResult(tftServiceResult);
                            _loggingService?.Log("Info", $"Retrieved TFT predictions for {symbol}: {tftResult.Action} with {tftResult.Confidence:P0} confidence");
                        }
                    }
                }
                catch (Exception tftEx)
                {
                    _loggingService?.LogErrorWithContext(tftEx, $"Failed to get TFT predictions for {symbol}");
                    // Continue without predictions - we'll still show the historical chart
                }

                // 3. Update the candlestick chart
                UpdateCandlestickChart(symbol, historicalData, tftResult);

                _loggingService?.Log("Info", $"Successfully updated candlestick chart for {symbol}");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to update chart with predictions for {symbol}");
                ClearCandlestickChart();
            }
        }

        /// <summary>
        /// Update the chart when a new prediction is generated
        /// Called from the main analysis flow
        /// </summary>
        /// <param name="prediction">The prediction model with TFT results</param>
        public async Task OnPredictionGenerated(PredictionModel prediction)
        {
            try
            {
                if (prediction == null || string.IsNullOrEmpty(prediction.Symbol))
                {
                    return;
                }

                // If the prediction has TFT-specific data, extract it
                TFTPredictionResult tftResult = null;
                
                // Check if we need to fetch TFT results separately or if they're embedded in the prediction
                // This depends on how your PredictionModel is structured
                
                // For now, trigger a full chart update
                await UpdateChartWithPredictionsAsync(prediction.Symbol);
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to update chart after prediction generation");
            }
        }

        /// <summary>
        /// Integrate with the existing analysis button click handler
        /// Call this method from AnalyzeButton_Click after predictions are generated
        /// </summary>
        private async Task RefreshChartForCurrentSymbol()
        {
            try
            {
                var symbol = ManualSymbolTextBox?.Text?.Trim()?.ToUpper();
                if (!string.IsNullOrEmpty(symbol))
                {
                    await UpdateChartWithPredictionsAsync(symbol);
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to refresh chart for current symbol");
            }
        }

        /// <summary>
        /// Convert TFTResult from service to TFTPredictionResult for chart display
        /// </summary>
        private TFTPredictionResult ConvertTFTResultToTFTPredictionResult(TFTResult serviceResult)
        {
            var result = new TFTPredictionResult
            {
                Symbol = serviceResult.Action, // Symbol may not be in TFTResult, using Action temporarily
                Action = serviceResult.Action,
                Confidence = serviceResult.Confidence,
                CurrentPrice = serviceResult.CurrentPrice,
                TargetPrice = serviceResult.TargetPrice,
                LowerBound = 0, // Will be filled from horizons
                UpperBound = 0, // Will be filled from horizons
                Uncertainty = 0,
                Horizons = new Dictionary<string, HorizonPredictionData>(),
                FeatureWeights = serviceResult.FeatureAttention ?? new Dictionary<string, double>(),
                PredictionTimestamp = DateTime.Now,
                Error = serviceResult.Error
            };

            // Convert forecast data to horizon predictions
            if (serviceResult.Predictions != null && serviceResult.Predictions.Count > 0)
            {
                foreach (var forecast in serviceResult.Predictions)
                {
                    var daysAhead = (forecast.Timestamp - DateTime.Now).Days;
                    var horizonKey = $"{daysAhead}d";

                    result.Horizons[horizonKey] = new HorizonPredictionData
                    {
                        MedianPrice = forecast.PredictedPrice,
                        LowerBound = forecast.LowerConfidence,
                        UpperBound = forecast.UpperConfidence,
                        TargetPrice = forecast.PredictedPrice,
                        Confidence = result.Confidence
                    };
                }

                // Set bounds from first horizon
                if (result.Horizons.Any())
                {
                    var firstHorizon = result.Horizons.Values.First();
                    result.LowerBound = firstHorizon.LowerBound;
                    result.UpperBound = firstHorizon.UpperBound;
                    result.Uncertainty = firstHorizon.UpperBound - firstHorizon.LowerBound;
                }
            }

            return result;
        }
    }
}
