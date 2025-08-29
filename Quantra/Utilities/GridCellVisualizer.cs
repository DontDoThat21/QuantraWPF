using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Quantra.Utilities
{
    /// <summary>
    /// Provides visualization of grid cells during control resizing
    /// </summary>
    public class GridCellVisualizer
    {
        private readonly Grid targetGrid;
        private readonly Panel visualLayer;
        private readonly SolidColorBrush cellFillBrush;
        private readonly SolidColorBrush cellStrokeBrush;
        
        private int gridRows;
        private int gridColumns;
        private bool isVisible = false;
        private Rectangle[,] cellOverlays; // Removed readonly modifier
        private Rectangle selectionRect;

        /// <summary>
        /// Creates a new grid cell visualizer
        /// </summary>
        /// <param name="grid">Target grid to visualize</param>
        /// <param name="visualLayer">The panel/canvas where visualization elements will be added</param>
        public GridCellVisualizer(Grid grid, Panel visualLayer)
        {
            targetGrid = grid ?? throw new ArgumentNullException(nameof(grid));
            this.visualLayer = visualLayer ?? throw new ArgumentNullException(nameof(visualLayer));
            
            // Set up brushes for visualization
            cellFillBrush = new SolidColorBrush(Color.FromArgb(40, 0, 255, 0)); // Light green semi-transparent
            cellStrokeBrush = new SolidColorBrush(Color.FromArgb(200, 0, 255, 0)); // Brighter green for borders
            
            // Get grid dimensions
            gridRows = grid.RowDefinitions.Count;
            gridColumns = grid.ColumnDefinitions.Count;
            
            // Create cell overlays
            cellOverlays = new Rectangle[gridRows, gridColumns];
            
            // Initialize the cell overlays but don't add them to visual layer yet
            for (int row = 0; row < gridRows; row++)
            {
                for (int col = 0; col < gridColumns; col++)
                {
                    cellOverlays[row, col] = CreateCellOverlay();
                    Grid.SetRow(cellOverlays[row, col], row);
                    Grid.SetColumn(cellOverlays[row, col], col);
                }
            }
            
            // Create selection rectangle
            selectionRect = new Rectangle
            {
                Fill = new SolidColorBrush(Color.FromArgb(100, 0, 255, 0)),
                Stroke = new SolidColorBrush(Colors.LimeGreen),
                StrokeThickness = 2,
                Visibility = Visibility.Collapsed
            };
            
            // Add selection rectangle to visual layer
            visualLayer.Children.Add(selectionRect);
        }
        
        private Rectangle CreateCellOverlay()
        {
            return new Rectangle
            {
                Fill = cellFillBrush,
                Stroke = cellStrokeBrush,
                StrokeThickness = 1.5,
                Visibility = Visibility.Collapsed
            };
        }
        
        /// <summary>
        /// Updates the grid dimensions if they have changed
        /// </summary>
        /// <param name="rows">Number of rows</param>
        /// <param name="columns">Number of columns</param>
        public void UpdateGridDimensions(int rows, int columns)
        {
            // If dimensions have changed, recreate the cell overlay array
            if (rows != gridRows || columns != gridColumns)
            {
                // Remove existing overlays
                if (isVisible)
                {
                    HideVisualizer();
                }
                
                gridRows = rows;
                gridColumns = columns;
                
                // Create new cell overlays
                cellOverlays = new Rectangle[gridRows, gridColumns];
                
                for (int row = 0; row < gridRows; row++)
                {
                    for (int col = 0; col < gridColumns; col++)
                    {
                        cellOverlays[row, col] = CreateCellOverlay();
                        Grid.SetRow(cellOverlays[row, col], row);
                        Grid.SetColumn(cellOverlays[row, col], col);
                    }
                }
                
                // If visualizer was visible, show the new overlays
                if (isVisible)
                {
                    ShowVisualizer();
                }
            }
        }
        
        /// <summary>
        /// Shows the grid cell visualizer
        /// </summary>
        public void ShowVisualizer()
        {
            if (!isVisible)
            {
                // Add all cell overlays to the visual layer
                for (int row = 0; row < gridRows; row++)
                {
                    for (int col = 0; col < gridColumns; col++)
                    {
                        if (!visualLayer.Children.Contains(cellOverlays[row, col]))
                        {
                            visualLayer.Children.Add(cellOverlays[row, col]);
                        }
                        cellOverlays[row, col].Visibility = Visibility.Visible;
                    }
                }
                
                isVisible = true;
            }
        }
        
        /// <summary>
        /// Hides the grid cell visualizer
        /// </summary>
        public void HideVisualizer()
        {
            if (isVisible)
            {
                // Remove all cell overlays from the visual layer
                for (int row = 0; row < gridRows; row++)
                {
                    for (int col = 0; col < gridColumns; col++)
                    {
                        cellOverlays[row, col].Visibility = Visibility.Collapsed;
                        if (visualLayer.Children.Contains(cellOverlays[row, col]))
                        {
                            visualLayer.Children.Remove(cellOverlays[row, col]);
                        }
                    }
                }
                
                isVisible = false;
            }
        }
        
        /// <summary>
        /// Highlights a region of cells in the grid
        /// </summary>
        /// <param name="row">Starting row</param>
        /// <param name="column">Starting column</param>
        /// <param name="rowSpan">Row span</param>
        /// <param name="columnSpan">Column span</param>
        public void HighlightCells(int row, int column, int rowSpan, int columnSpan)
        {
            // Make sure we don't go out of bounds
            rowSpan = Math.Min(rowSpan, gridRows - row);
            columnSpan = Math.Min(columnSpan, gridColumns - column);
            
            // Ensure spans are at least 1
            rowSpan = Math.Max(1, rowSpan);
            columnSpan = Math.Max(1, columnSpan);
            
            // Set selection rectangle
            Grid.SetRow(selectionRect, row);
            Grid.SetColumn(selectionRect, column);
            Grid.SetRowSpan(selectionRect, rowSpan);
            Grid.SetColumnSpan(selectionRect, columnSpan);
            
            selectionRect.Visibility = Visibility.Visible;
        }
        
        /// <summary>
        /// Hides the cell selection highlight
        /// </summary>
        public void HideHighlight()
        {
            selectionRect.Visibility = Visibility.Collapsed;
        }
    }
}
