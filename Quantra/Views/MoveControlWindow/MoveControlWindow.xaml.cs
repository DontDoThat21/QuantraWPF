using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Quantra.Views.MoveControlWindow
{
    public partial class MoveControlWindow : Window
    {
        // Properties for binding
        public string SourceTabName { get; private set; }
        public string SelectedTabName { get; private set; }
        public int ResultRow { get; private set; }
        public int ResultColumn { get; private set; }
        public bool IsApplied { get; private set; }

        // Private fields
        private readonly int _rowSpan;
        private readonly int _columnSpan;
        private readonly List<(int Row, int Col)> _occupiedCells = new List<(int Row, int Col)>();
        private readonly Dictionary<string, List<(int Row, int Col)>> _tabOccupiedCells = new Dictionary<string, List<(int Row, int Col)>>();

        // Constants
        private const int GridSize = 8; // Show 8x8 grid preview
        private readonly SolidColorBrush _availableCellBrush = new SolidColorBrush(Color.FromArgb(40, 30, 144, 255)); // Light blue
        private readonly SolidColorBrush _occupiedCellBrush = new SolidColorBrush(Color.FromArgb(40, 255, 0, 0)); // Light red
        private readonly SolidColorBrush _selectedCellBrush = new SolidColorBrush(Color.FromArgb(80, 0, 255, 0)); // Light green

        // Parameterless constructor for XAML designer support
        public MoveControlWindow()
        {
            InitializeComponent();
            ResultRow = 0;
            ResultColumn = 0;
        }

        public MoveControlWindow(string sourceTabName, List<string> availableTabs, int currentRow, int currentColumn, int rowSpan, int columnSpan)
        {
            InitializeComponent();

            // Set initial values
            SourceTabName = sourceTabName;
            SelectedTabName = sourceTabName;
            ResultRow = currentRow;
            ResultColumn = currentColumn;
            _rowSpan = rowSpan;
            _columnSpan = columnSpan;

            // Setup UI
            SourceTabTextBlock.Text = sourceTabName;
            RowTextBox.Text = currentRow.ToString();
            ColumnTextBox.Text = currentColumn.ToString();

            // Load available tabs
            if (availableTabs != null && availableTabs.Count > 0)
            {
                DestinationTabComboBox.ItemsSource = availableTabs;
                DestinationTabComboBox.SelectedItem = sourceTabName;
            }

            // Initialize grid preview
            InitializePreviewGrid();

            // Load occupied cells for the current tab
            LoadOccupiedCells(sourceTabName);

            // Update the preview
            UpdatePreview();

            // Set window title to include control position
            Title = $"Move Control - Currently at ({currentRow},{currentColumn})";

            // Set focus to the row textbox
            Loaded += (s, e) => RowTextBox.Focus();
        }

        private void InitializePreviewGrid()
        {
            // Clear existing grid definitions
            PreviewGrid.RowDefinitions.Clear();
            PreviewGrid.ColumnDefinitions.Clear();
            PreviewGrid.Children.Clear();

            // Add row and column definitions
            for (int i = 0; i < GridSize; i++)
            {
                PreviewGrid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
                PreviewGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            }

            // Add cell borders for the grid preview
            for (int row = 0; row < GridSize; row++)
            {
                for (int col = 0; col < GridSize; col++)
                {
                    var cellBorder = new Border
                    {
                        BorderBrush = new SolidColorBrush(Color.FromRgb(70, 70, 100)),
                        BorderThickness = new Thickness(1),
                        Background = _availableCellBrush
                    };

                    // Add row/col number labels
                    var label = new TextBlock
                    {
                        Text = $"{row},{col}",
                        FontSize = 10,
                        Foreground = new SolidColorBrush(Color.FromArgb(120, 255, 255, 255)),
                        VerticalAlignment = VerticalAlignment.Top,
                        HorizontalAlignment = HorizontalAlignment.Left,
                        Margin = new Thickness(2, 2, 0, 0)
                    };

                    cellBorder.Child = label;

                    // Set position in grid
                    Grid.SetRow(cellBorder, row);
                    Grid.SetColumn(cellBorder, col);

                    PreviewGrid.Children.Add(cellBorder);
                }
            }
        }

        private void LoadOccupiedCells(string tabName)
        {
            try
            {
                if (_tabOccupiedCells.ContainsKey(tabName))
                {
                    // Already loaded, use cached cells
                    _occupiedCells.Clear();
                    _occupiedCells.AddRange(_tabOccupiedCells[tabName]);
                    return;
                }

                // Get mainwindow instance to access occupied cells
                var mainWindow = Application.Current.Windows.OfType<Quantra.MainWindow>().FirstOrDefault();
                if (mainWindow != null)
                {
                    var cells = mainWindow.GetOccupiedCellsForTab(tabName);

                    // Cache the result
                    _occupiedCells.Clear();
                    _occupiedCells.AddRange(cells);
                    _tabOccupiedCells[tabName] = new List<(int Row, int Col)>(cells);

                    //DatabaseMonolith.Log("Info", $"Loaded {cells.Count} occupied cells for tab '{tabName}' in MoveControlWindow");
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error loading occupied cells: {ex.Message}", ex.ToString());
            }
        }

        private void UpdatePreview()
        {
            // Reset all cell backgrounds
            foreach (Border cell in PreviewGrid.Children)
            {
                int row = Grid.GetRow(cell);
                int col = Grid.GetColumn(cell);

                // Check if this cell is occupied by another control
                bool isOccupied = _occupiedCells.Contains((row, col));

                // Check if this is the cell that would be occupied by this control (and spans)
                bool isSelectedCell = (row >= ResultRow && row < ResultRow + _rowSpan) &&
                                     (col >= ResultColumn && col < ResultColumn + _columnSpan);

                // Update cell visuals
                if (isSelectedCell)
                {
                    cell.Background = _selectedCellBrush;
                    cell.BorderBrush = new SolidColorBrush(Colors.LimeGreen);
                    cell.BorderThickness = new Thickness(2);
                }
                else if (isOccupied)
                {
                    cell.Background = _occupiedCellBrush;
                    cell.BorderBrush = new SolidColorBrush(Color.FromRgb(70, 70, 100));
                    cell.BorderThickness = new Thickness(1);
                }
                else
                {
                    cell.Background = _availableCellBrush;
                    cell.BorderBrush = new SolidColorBrush(Color.FromRgb(70, 70, 100));
                    cell.BorderThickness = new Thickness(1);
                }
            }

            // Check for overlaps and update Move button state
            bool hasOverlap = CheckForOverlap();
            MoveButton.IsEnabled = !hasOverlap;

            // Visual indication when there's an overlap
            if (hasOverlap)
            {
                MoveButton.Background = new SolidColorBrush(Color.FromRgb(100, 100, 100));
                MoveButton.ToolTip = "Cannot move to this position due to overlap with existing controls";
            }
            else
            {
                MoveButton.Background = new SolidColorBrush(Color.FromRgb(49, 162, 120)); // #31A278
                MoveButton.ToolTip = null;
            }
        }

        private bool CheckForOverlap()
        {
            // Skip the current control position if in the same tab
            if (SelectedTabName == SourceTabName)
            {
                for (int r = ResultRow; r < ResultRow + _rowSpan; r++)
                {
                    for (int c = ResultColumn; c < ResultColumn + _columnSpan; c++)
                    {
                        foreach (var cell in _occupiedCells)
                        {
                            // Skip checking the original position
                            if (cell.Row >= ResultRow && cell.Row < ResultRow + _rowSpan &&
                                cell.Col >= ResultColumn && cell.Col < ResultColumn + _columnSpan)
                            {
                                continue;
                            }

                            if (cell.Row == r && cell.Col == c)
                            {
                                return true; // Overlap found
                            }
                        }
                    }
                }
            }
            else
            {
                // Check for any overlap in a different tab
                for (int r = ResultRow; r < ResultRow + _rowSpan; r++)
                {
                    for (int c = ResultColumn; c < ResultColumn + _columnSpan; c++)
                    {
                        if (_occupiedCells.Contains((r, c)))
                        {
                            return true; // Overlap found
                        }
                    }
                }
            }

            return false; // No overlap
        }

        private void Position_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Try to parse new values
            if (int.TryParse(RowTextBox.Text, out int newRow) && int.TryParse(ColumnTextBox.Text, out int newColumn))
            {
                // Ensure values are non-negative
                if (newRow >= 0 && newColumn >= 0)
                {
                    ResultRow = newRow;
                    ResultColumn = newColumn;
                    UpdatePreview();
                }
            }
        }

        private void DestinationTabComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (DestinationTabComboBox.SelectedItem is string selectedTab)
            {
                SelectedTabName = selectedTab;

                // Load occupied cells for the new tab
                LoadOccupiedCells(selectedTab);

                // Update the preview
                UpdatePreview();
            }
        }

        private void MoveButton_Click(object sender, RoutedEventArgs e)
        {
            if (!CheckForOverlap())
            {
                IsApplied = true;
                DialogResult = true;
                Close();
            }
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            IsApplied = false;
            DialogResult = false;
            Close();
        }
    }
}
