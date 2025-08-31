using System;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a control positioned in a tab's grid
    /// </summary>
    public class ControlModel
    {
        /// <summary>
        /// Type of the control (e.g., "Stock Chart", "News Feed", etc.)
        /// </summary>
        public string Type { get; set; }

        /// <summary>
        /// Row position in the grid (0-based)
        /// </summary>
        public int Row { get; set; }

        /// <summary>
        /// Column position in the grid (0-based)
        /// </summary>
        public int Column { get; set; }

        /// <summary>
        /// Number of rows the control spans
        /// </summary>
        public int RowSpan { get; set; } = 1;

        /// <summary>
        /// Number of columns the control spans
        /// </summary>
        public int ColSpan { get; set; } = 1;

        /// <summary>
        /// Default constructor
        /// </summary>
        public ControlModel() { }

        /// <summary>
        /// Constructor with properties
        /// </summary>
        public ControlModel(string type, int row, int column, int rowSpan = 1, int colSpan = 1)
        {
            Type = type;
            Row = row;
            Column = column;
            RowSpan = rowSpan;
            ColSpan = colSpan;
        }
    }
}
