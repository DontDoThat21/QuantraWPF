using System.Windows.Media;

namespace Quantra.Models
{
    /// <summary>
    /// Types and models used to visualize detected support/resistance price levels
    /// </summary>
    public static class PriceLevelVisualization
    {
        /// <summary>
        /// Line style for drawing price levels
        /// </summary>
        public enum LineStyle
        {
            Solid,
            Dashed,
            Dotted
        }

        /// <summary>
        /// Visual representation of a price level for charting
        /// </summary>
        public sealed class VisualPriceLevel
        {
            /// <summary>
            /// Price value of the level
            /// </summary>
            public double Price { get; set; }

            /// <summary>
            /// Display label shown on the chart legend/tooltip
            /// </summary>
            public string Label { get; set; } = string.Empty;

            /// <summary>
            /// Color of the line
            /// </summary>
            public Color LineColor { get; set; } = Colors.DodgerBlue;

            /// <summary>
            /// Line thickness
            /// </summary>
            public double LineThickness { get; set; } = 1.5;

            /// <summary>
            /// Line opacity (0..1)
            /// </summary>
            public double Opacity { get; set; } = 0.9;

            /// <summary>
            /// Line style (solid/dashed/dotted)
            /// </summary>
            public LineStyle Style { get; set; } = LineStyle.Solid;
        }
    }
}
