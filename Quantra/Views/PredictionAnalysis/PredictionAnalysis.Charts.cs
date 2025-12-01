using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Defaults;
using System.Windows.Controls;
using Quantra.Models;

namespace Quantra.Controls
{
    public partial class PredictionAnalysis : UserControl
    {
        private ChartValues<double> adxValues = new ChartValues<double>();
        private ChartValues<double> rocValues = new ChartValues<double>();
        private ChartValues<double> uoValues = new ChartValues<double>();
        private ChartValues<double> cciValues = new ChartValues<double>();
        private ChartValues<double> atrValues = new ChartValues<double>();
        private ChartValues<double> williamsRValues = new ChartValues<double>();
        private ChartValues<double> stochKValues = new ChartValues<double>();
        private ChartValues<double> stochDValues = new ChartValues<double>();
        private ChartValues<double> stochRsiValues = new ChartValues<double>();
        private ChartValues<double> bullPowerValues = new ChartValues<double>();
        private ChartValues<double> bearPowerValues = new ChartValues<double>();

        public ChartValues<double> AdxValues => adxValues;
        public ChartValues<double> RocValues => rocValues;
        public ChartValues<double> UoValues => uoValues;
        public ChartValues<double> CciValues => cciValues;
        public ChartValues<double> AtrValues => atrValues;
        public ChartValues<double> WilliamsRValues => williamsRValues;
        public ChartValues<double> StochKValues => stochKValues;
        public ChartValues<double> StochDValues => stochDValues;
        public ChartValues<double> StochRsiValues => stochRsiValues;
        public ChartValues<double> BullPowerValues => bullPowerValues;
        public ChartValues<double> BearPowerValues => bearPowerValues;

        // Add Breadth Thrust chart values
        public ChartValues<double> BreadthThrustValues { get; set; }

        private void InitializeChartData()
        {
            try
            {
                //DatabaseMonolith.Log("Info", "Initializing chart data collections");
                
                // Initialize all chart data collections
                PriceValues = new ChartValues<double>();
                VwapValues = new ChartValues<double>();
                PredictionValues = new ChartValues<double>();
                TopPerformerValues = new ChartValues<double>();
                TopPerformerLabels = new List<string>();
                PatternCandles = new ChartValues<OhlcPoint>();
                PatternHighlights = new ChartValues<double>();
                BreadthThrustValues = new ChartValues<double>(); // Initialize Breadth Thrust values
                
                // Set flag indicating charts are initialized
                chartsInitialized = true;
                //DatabaseMonolith.Log("Info", "Chart data collections initialized successfully");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to initialize chart data collections", ex.ToString());
                chartsInitialized = false;
            }
        }

        private void UpdatePlaceholderChartData()
        {
            try
            {
                if (!chartsInitialized)
                {
                    //DatabaseMonolith.Log("Warning", "Cannot update chart data - charts not initialized");
                    return;
                }
                
                // Clear existing data first
                PriceValues.Clear();
                VwapValues.Clear();
                PredictionValues.Clear();
                BreadthThrustValues.Clear(); // Clear Breadth Thrust values
                
                // Generate sample data for the charts
                List<double> priceData = new List<double> { 186.5, 185.2, 182.8, 180.1, 181.5, 183.2, 184.5, 185.7, 186.2, 188.0, 190.5, 192.3, 190.1, 189.8, 191.2, 193.4, 195.6, 194.8, 196.2, 198.1, 197.5 };
                List<double> vwapData = new List<double> { 185.0, 184.8, 183.5, 182.0, 181.8, 182.5, 183.8, 184.5, 185.2, 186.5, 188.0, 190.0, 190.8, 190.5, 190.8, 191.5, 193.0, 194.0, 194.5, 195.2, 196.0 };
                List<double> predictionData = Enumerable.Repeat(double.NaN, 10).Concat(new List<double> { 191.2, 193.4, 195.6, 194.8, 196.2, 198.1, 197.5, 199.1, 202.3, 205.0, 207.1 }).ToList();
                
                // Instead of assigning List<double> directly, use AddRange
                PriceValues.AddRange(priceData);
                VwapValues.AddRange(vwapData);
                PredictionValues.AddRange(predictionData);

                // Generate Breadth Thrust data using the specialized class
                BreadthThrustIndicator btIndicator = BreadthThrustIndicator.GenerateSimulatedData(21, 0.2);
                List<double> btData = btIndicator.GetAllValues();
                
                // Scale the Breadth Thrust values to match the price chart's scale
                double minPrice = priceData.Min();
                double maxPrice = priceData.Max();
                double priceRange = maxPrice - minPrice;
                
                List<double> scaledBtData = new List<double>();
                foreach (double btValue in btData)
                {
                    // Scale BT values (typically 0-1) to lower 25% of price chart range
                    double scaledValue = minPrice + (btValue * 0.25 * priceRange);
                    scaledBtData.Add(scaledValue);
                }
                
                // Add data to chart collections
                PriceValues.AddRange(priceData);
                VwapValues.AddRange(vwapData);
                PredictionValues.AddRange(predictionData);
                BreadthThrustValues.AddRange(scaledBtData); // Add scaled Breadth Thrust values
                
                // Setup top performers chart
                TopPerformerValues.Clear();
                TopPerformerLabels.Clear();
                
                TopPerformerValues.AddRange(new[] { 18.5, 15.2, 12.8, 10.5, 8.3 });
                TopPerformerLabels.AddRange(new[] { "NVDA", "AMD", "MSFT", "AAPL", "ADBE" });
                
                // Setup pattern candles
                PatternCandles.Clear();
                PatternHighlights.Clear();
                
                // Create sample candlestick data
                for (int i = 0; i < 30; i++)
                {
                    // Generate candle data with some randomness
                    double open = 100 + i * 0.5 + (new Random().NextDouble() - 0.5) * 5;
                    double close = open + (new Random().NextDouble() - 0.5) * 4;
                    
                    // Higher volatility in the middle for the pattern visualization
                    double highLowModifier = (i > 10 && i < 20) ? 2.0 : 1.0;
                    double high = Math.Max(open, close) + new Random().NextDouble() * 2 * highLowModifier;
                    double low = Math.Min(open, close) - new Random().NextDouble() * 2 * highLowModifier;
                    
                    PatternCandles.Add(new OhlcPoint(open, high, low, close));
                    
                    // Add special pattern highlight for indexes 12-18 (center area)
                    if (i >= 12 && i <= 18)
                    {
                        PatternHighlights.Add(high + 2);
                    }
                    else
                    {
                        // Use NaN for areas where the highlight shouldn't appear
                        PatternHighlights.Add(double.NaN);
                    }
                }
                
                //DatabaseMonolith.Log("Info", "Placeholder chart data updated successfully");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error updating placeholder chart data", ex.ToString());
            }
        }

        private List<double> CalculateMovingAverage(List<double> data, int period)
        {
            var result = new List<double>();
            
            for (int i = 0; i < data.Count; i++)
            {
                if (i < period - 1)
                {
                    // Not enough data for a full window, use available data
                    double sum = 0;
                    for (int j = 0; j <= i; j++)
                    {
                        sum += data[j];
                    }
                    result.Add(sum / (i + 1));
                }
                else
                {
                    // Full window available
                    double sum = 0;
                    for (int j = i - (period - 1); j <= i; j++)
                    {
                        sum += data[j];
                    }
                    result.Add(sum / period);
                }
            }
            
            return result;
        }

        private void UpdatePatternChart(PatternModel pattern)
        {
            try
            {
                // Clear existing data
                PatternCandles?.Clear();
                PatternHighlights?.Clear();
                
                // Generate simulated candlestick data
                Random random = new Random(pattern.Symbol.GetHashCode()); // Use symbol as seed for consistent randomness
                double basePrice = 100.0;
                double volatility = 2.0;
                
                for (int i = 0; i < 20; i++)
                {
                    // Generate OHLC data
                    double open = basePrice;
                    double close = basePrice + (random.NextDouble() * volatility * 2 - volatility);
                    double high = Math.Max(open, close) + random.NextDouble() * volatility;
                    double low = Math.Min(open, close) - random.NextDouble() * volatility;
                    
                    // Add to candles
                    PatternCandles?.Add(new OhlcPoint(open, high, low, close));
                    
                    // Update base price for next iteration
                    basePrice = close;
                    
                    // Add highlight points for pattern visualization
                    if (PatternHighlights != null)
                    {
                        if (pattern.PatternName == "Double Bottom" && (i == 5 || i == 15))
                        {
                            PatternHighlights.Add(low - 1); // Highlight the bottoms
                        }
                        else if (pattern.PatternName == "Cup and Handle" && i > 5 && i < 16)
                        {
                            // Create a cup shape
                            double cupDepth = -5 * Math.Sin(Math.PI * (i - 5) / 10) + 95;
                            PatternHighlights.Add(cupDepth);
                        }
                        else if (pattern.PatternName == "Head and Shoulders")
                        {
                            if (i == 4 || i == 16)
                            {
                                PatternHighlights.Add(high + 2); // Shoulder points
                            }
                            else if (i == 10)
                            {
                                PatternHighlights.Add(high + 4); // Head point
                            }
                            else
                            {
                                PatternHighlights.Add(double.NaN); // No highlight
                            }
                        }
                        else if (pattern.PatternName == "Bull Flag" && i > 10)
                        {
                            // Create a flag pattern
                            PatternHighlights.Add(basePrice + 5 - (i - 10) * 0.5);
                        }
                        else
                        {
                            PatternHighlights.Add(double.NaN); // No highlight
                        }
                    }
                }
                
                // Force chart to update
                OnPropertyChanged(nameof(PatternCandles));
                OnPropertyChanged(nameof(PatternHighlights));
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to update pattern chart data", ex.ToString());
            }
        }

        private void UpdateAllIndicatorDisplays(PredictionModel prediction)
        {
            // Existing indicator updates
            //UpdateRSIDisplay(prediction);
            //UpdateMACDDisplay(prediction);
            
            // New indicator updates
            //UpdateADXDisplay(prediction);
            //UpdateROCDisplay(prediction);
            //UpdateUltimateOscillatorDisplay(prediction);
            //UpdateBullBearPowerDisplay(prediction);
            //UpdateCCIDisplay(prediction);
            //UpdateATRDisplay(prediction);
            //UpdateWilliamsRDisplay(prediction);
            //UpdateStochasticDisplay(prediction);
            //UpdateStochasticRSIDisplay(prediction);
        }
    }
}
