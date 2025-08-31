using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Model for storing DataGrid layout settings including width and column widths
    /// </summary>
    public class DataGridSettings
    {
        /// <summary>
        /// Width of the DataGrid control
        /// </summary>
        public double DataGridWidth { get; set; } = double.NaN;

        /// <summary>
        /// Height of the DataGrid control  
        /// </summary>
        public double DataGridHeight { get; set; } = double.NaN;

        /// <summary>
        /// Dictionary of column names to their widths
        /// </summary>
        public Dictionary<string, double> ColumnWidths { get; set; } = new Dictionary<string, double>();
    }
}