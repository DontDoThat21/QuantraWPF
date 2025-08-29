using System;
using System.Collections.Generic;
using System.Windows.Media;

namespace Quantra.Models
{
    /// <summary>
    /// Helper class for visualizing support/resistance price levels in charts
    /// </summary>
    public class PriceLevelVisualization
    {
        /// <summary>
        /// Represents a visual representation of a price level for charting
        /// </summary>
        public class VisualPriceLevel
        {
            /// <summary>
            /// The price level
            /// </summary>
            public double Price { get; set; }
            
            /// <summary>
            /// Line color for visualization
            /// </summary>
            public Color LineColor { get; set; }
            
            /// <summary>
            /// Line thickness (1.0-5.0)
            /// </summary>
            public double LineThickness { get; set; }
            
            /// <summary>
            /// Line opacity (0.0-1.0)
            /// </summary>
            public double Opacity { get; set; }
            
            /// <summary>
            /// Line style (Solid, Dashed, Dotted)
            /// </summary>
            public LineStyle Style { get; set; }
            
            /// <summary>
            /// Extended description for tooltips
            /// </summary>
            public string Label { get; set; }
            
            /// <summary>
            /// Whether this is a major (true) or minor (false) level
            /// </summary>
            public bool IsMajorLevel { get; set; }
        }
        
        /// <summary>
        /// Line style for visualization
        /// </summary>
        public enum LineStyle
        {
            Solid,
            Dashed,
            Dotted
        }
        
        /// <summary>
        /// Create visual representation for price levels based on their properties
        /// </summary>
        /// <param name="levels">List of detected price levels</param>
        /// <returns>List of visual price levels for rendering</returns>
        public static List<VisualPriceLevel> CreateVisuals(List<PriceLevelAnalyzer.PriceLevel> levels)
        {
            var visuals = new List<VisualPriceLevel>();
            
            foreach (var level in levels)
            {
                var visual = new VisualPriceLevel
                {
                    Price = level.Price,
                    LineThickness = 1.0 + (level.Strength * 2.0), // 1-3 based on strength
                    Opacity = 0.5 + (level.Strength * 0.5),       // 0.5-1.0 based on strength
                    IsMajorLevel = level.Strength > 0.7,          // Major levels are stronger
                    Label = FormatLevelLabel(level)
                };
                
                // Set color and style based on detection method and type
                (visual.LineColor, visual.Style) = GetVisualStyleForLevel(level);
                
                visuals.Add(visual);
            }
            
            return visuals;
        }
        
        /// <summary>
        /// Get color and style for a price level based on its properties
        /// </summary>
        private static (Color color, LineStyle style) GetVisualStyleForLevel(PriceLevelAnalyzer.PriceLevel level)
        {
            // Base color depends on whether it's support, resistance, or both
            Color baseColor;
            if (level.IsSupport && level.IsResistance)
                baseColor = Colors.Purple;
            else if (level.IsSupport)
                baseColor = Colors.Green;
            else
                baseColor = Colors.Red;
            
            // Style based on detection method
            LineStyle style = LineStyle.Solid;
            
            switch (level.DetectionMethod)
            {
                case PriceLevelAnalyzer.LevelDetectionMethod.PriceAction:
                    // Price action levels use solid lines
                    style = LineStyle.Solid;
                    break;
                    
                case PriceLevelAnalyzer.LevelDetectionMethod.PivotPoint:
                    // Pivot points use dashed lines
                    style = LineStyle.Dashed;
                    // Adjust color for pivot points
                    if (level.Description.Contains("PP"))
                        baseColor = Colors.Blue;
                    break;
                    
                case PriceLevelAnalyzer.LevelDetectionMethod.Fibonacci:
                    // Fibonacci levels use dotted lines
                    style = LineStyle.Dotted;
                    // Use golden color for key Fibonacci levels
                    if (level.Description.Contains("618") || level.Description.Contains("382"))
                        baseColor = Color.FromRgb(212, 175, 55); // Gold color
                    break;
                    
                case PriceLevelAnalyzer.LevelDetectionMethod.VolumeProfile:
                    // Volume profile levels use semi-transparent solid lines
                    style = LineStyle.Solid;
                    baseColor = Colors.DarkOrange;
                    break;
            }
            
            return (baseColor, style);
        }
        
        /// <summary>
        /// Format a descriptive label for the price level
        /// </summary>
        private static string FormatLevelLabel(PriceLevelAnalyzer.PriceLevel level)
        {
            string typeLabel = level.IsSupport && level.IsResistance ? "S/R" : 
                               level.IsSupport ? "Support" : "Resistance";
                               
            string strengthDesc = level.Strength >= 0.8 ? "Strong" :
                                 level.Strength >= 0.5 ? "Moderate" : "Weak";
                                 
            string touchInfo = level.TouchCount > 0 ? $" ({level.TouchCount} touches)" : "";
            
            // Format the basic info
            string label = $"{typeLabel}: {strengthDesc}{touchInfo}";
            
            // Add method-specific details
            if (!string.IsNullOrEmpty(level.Description))
            {
                label += $" - {level.Description}";
            }
            
            return label;
        }
    }
}