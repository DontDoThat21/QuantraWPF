using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using Quantra.ViewModels;

namespace Quantra.Views.MoveControlWindow
{
    public partial class MoveControlWindow : Window
    {
        private readonly MoveControlWindowViewModel _viewModel;
        public bool IsApplied { get; private set; }

        // Private fields for UI
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
        }

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public MoveControlWindow(MoveControlWindowViewModel viewModel)
        {
            InitializeComponent();

            _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
            DataContext = _viewModel;

            // Subscribe to ViewModel events
            _viewModel.MoveApplied += OnMoveApplied;
            _viewModel.CloseRequested += OnCloseRequested;

            // Setup UI
            SourceTabTextBlock.Text = _viewModel.SourceTabName;
            RowTextBox.Text = _viewModel.ResultRow.ToString();
            ColumnTextBox.Text = _viewModel.ResultColumn.ToString();

            // Load available tabs
            DestinationTabComboBox.ItemsSource = _viewModel.AvailableTabs;
            DestinationTabComboBox.SelectedItem = _viewModel.SelectedTabName;

            // Initialize grid preview
            InitializePreviewGrid();

            // Load occupied cells for the current tab
            LoadOccupiedCells(_viewModel.SourceTabName);

            // Update the preview
            UpdatePreview();

            // Set window title to include control position
            Title = $"Move Control - Currently at ({_viewModel.ResultRow},{_viewModel.ResultColumn})";

            // Set focus to the row textbox
            Loaded += (s, e) => RowTextBox.Focus();
        }

        /// <summary>
        /// Legacy constructor for compatibility
        /// </summary>
        public MoveControlWindow(string sourceTabName, List<string> availableTabs, int currentRow, int currentColumn, int rowSpan, int columnSpan)
            : this(new MoveControlWindowViewModel(sourceTabName, availableTabs, currentRow, currentColumn, rowSpan, columnSpan))
        {
        }

        private void OnMoveApplied(object sender, MoveControlEventArgs e)
        {
            IsApplied = true;
        }

        private void OnCloseRequested(object sender, bool applied)
        {
            IsApplied = applied;
            DialogResult = applied;
            Close();
        }

        protected override void OnClosed(EventArgs e)
        {
            // Unsubscribe from events to prevent memory leaks
            if (_viewModel != null)
            {
                _viewModel.MoveApplied -= OnMoveApplied;
                _viewModel.CloseRequested -= OnCloseRequested;
            }
            base.OnClosed(e);
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
                bool isSelectedCell = (row >= _viewModel.ResultRow && row < _viewModel.ResultRow + _viewModel.RowSpan) &&
                                     (col >= _viewModel.ResultColumn && col < _viewModel.ResultColumn + _viewModel.ColumnSpan);

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
            if (_viewModel.SelectedTabName == _viewModel.SourceTabName)
            {
                for (int r = _viewModel.ResultRow; r < _viewModel.ResultRow + _viewModel.RowSpan; r++)
                {
                    for (int c = _viewModel.ResultColumn; c < _viewModel.ResultColumn + _viewModel.ColumnSpan; c++)
                    {
                        foreach (var cell in _occupiedCells)
                        {
                            // Skip checking the original position
                            if (cell.Row >= _viewModel.ResultRow && cell.Row < _viewModel.ResultRow + _viewModel.RowSpan &&
                                cell.Col >= _viewModel.ResultColumn && cell.Col < _viewModel.ResultColumn + _viewModel.ColumnSpan)
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
                for (int r = _viewModel.ResultRow; r < _viewModel.ResultRow + _viewModel.RowSpan; r++)
                {
                    for (int c = _viewModel.ResultColumn; c < _viewModel.ResultColumn + _viewModel.ColumnSpan; c++)
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
                    _viewModel.ResultRow = newRow;
                    _viewModel.ResultColumn = newColumn;
                    UpdatePreview();
                }
            }
        }

        private void DestinationTabComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (DestinationTabComboBox.SelectedItem is string selectedTab)
            {
                _viewModel.SelectedTabName = selectedTab;

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
