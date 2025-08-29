using System;
using System.Collections.Generic;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using Quantra.Models;

namespace Quantra.Views.ChartExtensions
{
    /// <summary>
    /// Provides extension methods for adding support/resistance levels to LiveCharts
    /// </summary>
    public static class PriceLevelChartExtensions
    {
        /// <summary>
        /// Adds support/resistance price levels to a LiveCharts series collection
        /// </summary>
        /// <param name="seriesCollection">The chart series collection</param>
        /// <param name="levels">Visual price levels to display</param>
        /// <param name="chartStartIndex">Start index of the visible chart area</param>
        /// <param name="chartEndIndex">End index of the visible chart area</param>
        /// <returns>The updated series collection</returns>
        public static SeriesCollection AddPriceLevels(
            this SeriesCollection seriesCollection,
            List<PriceLevelVisualization.VisualPriceLevel> levels,
            int chartStartIndex,
            int chartEndIndex)
        {
            if (levels == null || levels.Count == 0) 
                return seriesCollection;
                
            int chartWidth = chartEndIndex - chartStartIndex + 1;
            
            // Add each level as a horizontal line or dashed line
            foreach (var level in levels)
            {
                // Create constant values array for horizontal line
                var values = new ChartValues<double>();
                for (int i = 0; i < chartWidth; i++)
                {
                    values.Add(level.Price);
                }
                
                // Create line series for the price level
                var levelSeries = new LineSeries
                {
                    Title = level.Label,
                    Values = values,
                    PointGeometry = null, // No points, just a line
                    Stroke = new SolidColorBrush(level.LineColor),
                    StrokeThickness = level.LineThickness,
                    Fill = Brushes.Transparent,
                    LineSmoothness = 0, // Straight line
                    Opacity = level.Opacity
                };
                
                // Apply line style (solid, dashed, dotted)
                switch (level.Style)
                {
                    case PriceLevelVisualization.LineStyle.Dashed:
                        levelSeries.StrokeDashArray = new DoubleCollection(new[] { 4.0, 2.0 });
                        break;
                    case PriceLevelVisualization.LineStyle.Dotted:
                        levelSeries.StrokeDashArray = new DoubleCollection(new[] { 1.0, 2.0 });
                        break;
                }
                
                seriesCollection.Add(levelSeries);
            }
            
            return seriesCollection;
        }
        
        /// <summary>
        /// Sets up a price level visualization series for a real-time chart
        /// </summary>
        /// <param name="chart">The CartesianChart to add levels to</param>
        /// <param name="strategy">The SupportResistanceStrategy with detected levels</param>
        public static void SetupPriceLevelsOnChart(
            this CartesianChart chart, 
            SupportResistanceStrategy strategy)
        {
            // Get visual levels from the strategy
            var visualLevels = strategy.GetVisualLevels();
            
            if (visualLevels == null || visualLevels.Count == 0)
                return;
            
            // Create a new series collection if needed
            if (chart.Series == null)
                chart.Series = new SeriesCollection();
            
            // Remove any existing level series
            for (int i = chart.Series.Count - 1; i >= 0; i--)
            {
                var series = chart.Series[i];
                if (series.Title?.StartsWith("Level:") == true)
                {
                    chart.Series.RemoveAt(i);
                }
            }
            
            // Add each level as a series
            foreach (var level in visualLevels)
            {
                // Set title with "Level:" prefix for easy identification
                var title = $"Level: {level.Label}";
                
                // Create constant values array for horizontal line (for cartesian chart)
                var values = new ChartValues<double> { level.Price };
                
                // Create line series for the price level
                var levelSeries = new LineSeries
                {
                    Title = title,
                    Values = values,
                    PointGeometry = null, // No points, just a line
                    Stroke = new SolidColorBrush(level.LineColor),
                    StrokeThickness = level.LineThickness,
                    Fill = Brushes.Transparent,
                    ScalesYAt = 0, // Use primary Y axis
                    Opacity = level.Opacity,
                    // This is for a constant horizontal line across the chart
                    DataLabels = false
                };
                
                // Apply line style (solid, dashed, dotted)
                switch (level.Style)
                {
                    case PriceLevelVisualization.LineStyle.Dashed:
                        levelSeries.StrokeDashArray = new DoubleCollection(new[] { 4.0, 2.0 });
                        break;
                    case PriceLevelVisualization.LineStyle.Dotted:
                        levelSeries.StrokeDashArray = new DoubleCollection(new[] { 1.0, 2.0 });
                        break;
                }
                
                chart.Series.Add(levelSeries);
            }
        }
    }
}