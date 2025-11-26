using Quantra;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Quantra
{
    /// <summary>
    /// Partial class containing grid visualization and validation functionality
    /// </summary>
    public partial class AddControlWindow
    {
        // NOTE: Field declarations removed to avoid ambiguity with those in AddControlWindow.xaml.cs
        // These fields are already declared in the main file:
        // - private HashSet<(int Row, int Column)> occupiedCells
        // - private Rectangle selectionRect
        // - private bool useOneBased

        // Add new fields for click selection
        private (int Row, int Column)? firstClickCell = null;
        private bool isSelectingSpan = false;

        /// <summary>
        /// Updates the visualization of the grid based on current row/column settings
        /// </summary>
        private void UpdateGridVisualization()
        {
            try
            {
                // Ensure we have a grid to draw on - use the Grid instead of Canvas
                var gridVisualization = FindName("GridVisualization") as Grid;
                if (gridVisualization == null)
                {
                    //DatabaseMonolith.Log("Warning", "GridVisualization not found");
                    return;
                }

                // Clear existing visualization
                gridVisualization.Children.Clear();
                gridVisualization.RowDefinitions.Clear();
                gridVisualization.ColumnDefinitions.Clear();

                // Get the selected tab and its grid config
                var selectedTab = TabComboBox?.SelectedItem as string;
                if (string.IsNullOrEmpty(selectedTab))
                    return;

                var gridConfig = DatabaseMonolith.LoadGridConfig(selectedTab);
                if (gridConfig.Equals(null))
                    return;

                // Set up the grid dimensions
                for (int i = 0; i < gridConfig.Rows; i++)
                {
                    gridVisualization.RowDefinitions.Add(new RowDefinition());
                }
                for (int j = 0; j < gridConfig.Columns; j++)
                {
                    gridVisualization.ColumnDefinitions.Add(new ColumnDefinition());
                }

                // Draw grid cells
                for (int row = 0; row < gridConfig.Rows; row++)
                {
                    for (int col = 0; col < gridConfig.Columns; col++)
                    {
                        // Create cell border
                        var cellBorder = new Border
                        {
                            Background = GetCellBrush(row, col),
                            BorderBrush = new SolidColorBrush(Colors.Gray),
                            BorderThickness = new Thickness(1),
                            Margin = new Thickness(1),
                            Cursor = Cursors.Hand,
                            Tag = new { Row = row, Column = col } // Store row/col for click handling
                        };

                        // Add click event handler
                        cellBorder.MouseLeftButtonDown += CellBorder_MouseLeftButtonDown;
                        cellBorder.MouseEnter += CellBorder_MouseEnter;
                        cellBorder.MouseLeave += CellBorder_MouseLeave;

                        // Add cell label (1-based for display)
                        var label = new TextBlock
                        {
                            Text = $"{row + 1},{col + 1}",
                            Foreground = new SolidColorBrush(Colors.White),
                            FontSize = 10,
                            FontWeight = FontWeights.SemiBold,
                            HorizontalAlignment = HorizontalAlignment.Center,
                            VerticalAlignment = VerticalAlignment.Center,
                            TextAlignment = TextAlignment.Center,
                            IsHitTestVisible = false // Allow clicks to pass through to the border
                        };

                        cellBorder.Child = label;

                        // Set grid placement
                        Grid.SetRow(cellBorder, row);
                        Grid.SetColumn(cellBorder, col);
                        gridVisualization.Children.Add(cellBorder);
                    }
                }

                // Highlight the currently selected position if valid values exist
                HighlightSelectedPosition();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to update grid visualization: {ex.Message}", ex.ToString());
            }
        }

        /// <summary>
        /// Handle cell border mouse enter for hover effects
        /// </summary>
        private void CellBorder_MouseEnter(object sender, MouseEventArgs e)
        {
            if (sender is Border border && border.Tag != null)
            {
                dynamic tag = border.Tag;
                int row = tag.Row;
                int col = tag.Column;

                // Show hover effect if cell is not occupied
                if (!IsOccupied(row, col))
                {
                    border.Background = new SolidColorBrush(Color.FromArgb(100, 0, 255, 0)); // Light green hover
                }
            }
        }

        /// <summary>
        /// Handle cell border mouse leave to restore original color
        /// </summary>
        private void CellBorder_MouseLeave(object sender, MouseEventArgs e)
        {
            if (sender is Border border && border.Tag != null)
            {
                dynamic tag = border.Tag;
                int row = tag.Row;
                int col = tag.Column;

                // Restore original cell color
                border.Background = GetCellBrush(row, col);
            }
        }

        /// <summary>
        /// Handle cell click for span selection
        /// </summary>
        private void CellBorder_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (sender is Border border && border.Tag != null)
            {
                dynamic tag = border.Tag;
                int row = tag.Row;
                int col = tag.Column;

                // Don't allow clicking on occupied cells
                if (IsOccupied(row, col))
                {
                    return;
                }

                if (!isSelectingSpan)
                {
                    // First click - set the starting position
                    firstClickCell = (row, col);
                    isSelectingSpan = true;

                    // Update the position textboxes (convert to 1-based)
                    if (RowTextBox != null)
                        RowTextBox.Text = (row + 1).ToString();
                    if (ColumnTextBox != null)
                        ColumnTextBox.Text = (col + 1).ToString();

                    // Set initial span to 1x1
                    if (RowSpanTextBox != null)
                        RowSpanTextBox.Text = "1";
                    if (ColumnSpanTextBox != null)
                        ColumnSpanTextBox.Text = "1";

                    //DatabaseMonolith.Log("Info", $"First click at cell ({row + 1},{col + 1})");
                }
                else
                {
                    // Second click - calculate span
                    if (firstClickCell.HasValue)
                    {
                        var firstCell = firstClickCell.Value;
                        
                        // Calculate the span (ensure second click is bottom-right)
                        int startRow = Math.Min(firstCell.Row, row);
                        int startCol = Math.Min(firstCell.Column, col);
                        int endRow = Math.Max(firstCell.Row, row);
                        int endCol = Math.Max(firstCell.Column, col);

                        int rowSpan = endRow - startRow + 1;
                        int colSpan = endCol - startCol + 1;

                        // Validate that all cells in the span are available
                        bool allCellsAvailable = true;
                        for (int r = startRow; r <= endRow; r++)
                        {
                            for (int c = startCol; c <= endCol; c++)
                            {
                                if (IsOccupied(r, c))
                                {
                                    allCellsAvailable = false;
                                    break;
                                }
                            }
                            if (!allCellsAvailable) break;
                        }

                        if (allCellsAvailable)
                        {
                            // Update all textboxes with the calculated values (convert to 1-based)
                            if (RowTextBox != null)
                                RowTextBox.Text = (startRow + 1).ToString();
                            if (ColumnTextBox != null)
                                ColumnTextBox.Text = (startCol + 1).ToString();
                            if (RowSpanTextBox != null)
                                RowSpanTextBox.Text = rowSpan.ToString();
                            if (ColumnSpanTextBox != null)
                                ColumnSpanTextBox.Text = colSpan.ToString();

                            //DatabaseMonolith.Log("Info", $"Selected span: ({startRow + 1},{startCol + 1}) to ({endRow + 1},{endCol + 1}) = {rowSpan}x{colSpan}");
                        }
                        else
                        {
                            // Show warning about occupied cells
                            var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
                            if (mainWindow != null)
                            {
                                mainWindow.AppendAlert("Some cells in the selected span are already occupied!", "warning");
                            }
                        }
                    }

                    // Reset selection state
                    firstClickCell = null;
                    isSelectingSpan = false;
                }

                // Update the visualization to show the new selection
                UpdateGridVisualization();
                
                e.Handled = true;
            }
        }

        /// <summary>
        /// Gets the appropriate brush for a cell based on its state (occupied, selected, etc)
        /// </summary>
        private Brush GetCellBrush(int row, int col)
        {
            // Check if this is the first clicked cell during span selection
            if (isSelectingSpan && firstClickCell.HasValue && 
                firstClickCell.Value.Row == row && firstClickCell.Value.Column == col)
            {
                return new SolidColorBrush(Color.FromArgb(150, 255, 165, 0)); // Orange for first click
            }

            // Check if the cell is occupied
            if (IsOccupied(row, col))
            {
                return new SolidColorBrush(Color.FromArgb(100, 255, 0, 0)); // Red for occupied
            }

            // Default cell color (available)
            return new SolidColorBrush(Color.FromArgb(50, 0, 0, 150)); // Dark blue for available
        }

        /// <summary>
        /// Highlight the currently selected position in the grid
        /// </summary>
        private void HighlightSelectedPosition()
        {
            // Parse the current position values
            if (!int.TryParse(RowTextBox?.Text, out int row) ||
                !int.TryParse(ColumnTextBox?.Text, out int col) ||
                !int.TryParse(RowSpanTextBox?.Text, out int rowSpan) ||
                !int.TryParse(ColumnSpanTextBox?.Text, out int colSpan))
            {
                // Hide selection rect if inputs are invalid
                if (selectionRect != null && selectionRect.Visibility == Visibility.Visible)
                {
                    selectionRect.Visibility = Visibility.Collapsed;
                }
                return;
            }

            // Convert from 1-based to 0-based if needed
            if (useOneBased)
            {
                row--;
                col--;
            }

            // Get the selected tab and its grid config
            var selectedTab = TabComboBox?.SelectedItem as string;
            if (string.IsNullOrEmpty(selectedTab))
                return;

            var gridConfig = DatabaseMonolith.LoadGridConfig(selectedTab);
            if (gridConfig.Equals(null))
                return;

            // Validate position is within grid bounds
            if (row < 0 || row >= gridConfig.Rows || col < 0 || col >= gridConfig.Columns)
            {
                // Hide selection rect if position is out of bounds
                if (selectionRect != null && selectionRect.Visibility == Visibility.Visible)
                {
                    selectionRect.Visibility = Visibility.Collapsed;
                }
                return;
            }

            // Ensure spans don't go out of bounds
            rowSpan = Math.Min(rowSpan, gridConfig.Rows - row);
            colSpan = Math.Min(colSpan, gridConfig.Columns - col);

            // Get the grid visualization
            var gridVisualization = FindName("GridVisualization") as Grid;
            if (gridVisualization == null)
                return;

            // Create or update selection rectangle overlay
            if (selectionRect == null)
            {
                selectionRect = new Rectangle
                {
                    Fill = new SolidColorBrush(Color.FromArgb(100, 0, 255, 0)),
                    Stroke = new SolidColorBrush(Colors.LimeGreen),
                    StrokeThickness = 2,
                    IsHitTestVisible = false // Allow clicks to pass through
                };
            }

            // Position the selection rectangle
            Grid.SetRow(selectionRect, row);
            Grid.SetColumn(selectionRect, col);
            Grid.SetRowSpan(selectionRect, rowSpan);
            Grid.SetColumnSpan(selectionRect, colSpan);

            // Add to grid if not already there
            if (!gridVisualization.Children.Contains(selectionRect))
            {
                gridVisualization.Children.Add(selectionRect);
            }

            // Always make the selection rectangle visible when there's a valid selection
            selectionRect.Visibility = Visibility.Visible;
        }

        /// <summary>
        /// Initialize span selection controls
        /// </summary>
        private void InitializeSpanSelection()
        {
            // Reset selection state when initializing
            firstClickCell = null;
            isSelectingSpan = false;
        }

        /// <summary>
        /// Refreshes the list of cells that are already occupied in the tab
        /// </summary>
        private void RefreshOccupiedCells(string tabName)
        {
            if (string.IsNullOrEmpty(tabName))
                return;

            // Clear existing occupation data
            if (occupiedCells == null)
                occupiedCells = new HashSet<(int Row, int Column)>();
            else
                occupiedCells.Clear();

            try
            {
                // Get controls for the tab from the database
                var controls = DatabaseMonolith.LoadControlsForTab(tabName);
                
                // Log the number of controls found for debugging
                //DatabaseMonolith.Log("Debug", $"Found {controls.Count} controls for tab '{tabName}'");

                // Add all occupied cells to our list
                foreach (var control in controls)
                {
                    for (int r = 0; r < control.RowSpan; r++)
                    {
                        for (int c = 0; c < control.ColSpan; c++)
                        {
                            occupiedCells.Add((control.Row + r, control.Column + c));
                        }
                    }
                }
                
                // Log how many cells are now marked as occupied
                //DatabaseMonolith.Log("Debug", $"Marked {occupiedCells.Count} cells as occupied in tab '{tabName}'");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error refreshing occupied cells: {ex.Message}", ex.ToString());
            }
        }

        /// <summary>
        /// Checks if a given cell is already occupied
        /// </summary>
        private bool IsOccupied(int row, int col)
        {
            // Safely handle tuple comparison with null check
            return occupiedCells != null && occupiedCells.Contains((row, col));
        }
    }
}
