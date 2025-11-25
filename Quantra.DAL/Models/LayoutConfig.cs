using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Quantra.Models
{
    /// <summary>
    /// Model representing chart panel layout configuration
    /// </summary>
    public class ChartPanelLayout
    {
        /// <summary>
        /// Panel identifier (e.g., "Price", "Volume", "RSI", "MACD")
        /// </summary>
        [Required]
        public string PanelId { get; set; } = string.Empty;

        /// <summary>
        /// Display name for the panel
        /// </summary>
        [Required]
        public string DisplayName { get; set; } = string.Empty;

        /// <summary>
        /// Row position in the grid (0-based)
        /// </summary>
        [Range(0, 20)]
        public int Row { get; set; }

        /// <summary>
        /// Column position in the grid (0-based)
        /// </summary>
        [Range(0, 20)]
        public int Column { get; set; }

        /// <summary>
        /// Number of rows this panel spans
        /// </summary>
        [Range(1, 10)]
        public int RowSpan { get; set; } = 1;

        /// <summary>
        /// Number of columns this panel spans
        /// </summary>
        [Range(1, 10)]
        public int ColumnSpan { get; set; } = 1;

        /// <summary>
        /// Height ratio compared to other panels
        /// </summary>
        [Range(0.1, 10.0)]
        public double HeightRatio { get; set; } = 1.0;

        /// <summary>
        /// Whether this panel is visible
        /// </summary>
        public bool IsVisible { get; set; } = true;

        /// <summary>
        /// Display order for panels in the same position
        /// </summary>
        public int DisplayOrder { get; set; }
    }

    /// <summary>
    /// Complete layout configuration for the chart visualization
    /// </summary>
    public class LayoutConfig
    {
        /// <summary>
        /// List of chart panel layouts
        /// </summary>
        public List<ChartPanelLayout> Panels { get; set; } = new List<ChartPanelLayout>();

        /// <summary>
        /// Total number of grid rows
        /// </summary>
        [Range(1, 20)]
        public int TotalRows { get; set; } = 4;

        /// <summary>
        /// Total number of grid columns
        /// </summary>
        [Range(1, 20)]
        public int TotalColumns { get; set; } = 1;

        /// <summary>
        /// Layout name for saving/loading different layouts
        /// </summary>
        [Required]
        public string LayoutName { get; set; } = "Default";

        /// <summary>
        /// Whether to show grid lines during editing
        /// </summary>
        public bool ShowGridLines { get; set; } = true;

        /// <summary>
        /// Grid line color
        /// </summary>
        public string GridLineColor { get; set; } = "#FF00FFFF";

        /// <summary>
        /// Creates a default layout configuration
        /// </summary>
        /// <returns>Default layout with Price, Volume, and RSI panels</returns>
        public static LayoutConfig CreateDefault()
        {
            return new LayoutConfig
            {
                LayoutName = "Default",
                TotalRows = 4,
                TotalColumns = 1,
                Panels = new List<ChartPanelLayout>
                {
                    new ChartPanelLayout
                    {
                        PanelId = "Price",
                        DisplayName = "Price Chart",
                        Row = 0,
                        Column = 0,
                        RowSpan = 3,
                        ColumnSpan = 1,
                        HeightRatio = 3.0,
                        IsVisible = true,
                        DisplayOrder = 1
                    },
                    new ChartPanelLayout
                    {
                        PanelId = "Volume",
                        DisplayName = "Volume",
                        Row = 3,
                        Column = 0,
                        RowSpan = 1,
                        ColumnSpan = 1,
                        HeightRatio = 1.0,
                        IsVisible = true,
                        DisplayOrder = 2
                    }
                }
            };
        }
    }
}