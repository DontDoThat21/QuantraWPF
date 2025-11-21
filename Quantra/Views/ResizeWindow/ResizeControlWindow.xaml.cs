using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Quantra
{
    public partial class ResizeControlWindow : Window
    {
        private int initialRow;
        private int initialColumn;
        private int initialRowSpan;
        private int initialColumnSpan;
        private string tabName;
        private Border controlBorder;
        private int gridMaxRows = 4;
        private int gridMaxCols = 4;
        private List<(int Row, int Col)> occupiedCells = new List<(int Row, int Col)>();
        private bool useOneBased = true; // Always use 1-based indexing now
        private string controlType = "Control"; // Default control type name
        
        // Result properties
        public int ResultRow { get; private set; }
        public int ResultColumn { get; private set; }
        public int ResultRowSpan { get; private set; }
        public int ResultColumnSpan { get; private set; }
        public bool IsApplied { get; private set; }

        private readonly List<Rectangle> occupiedCellIndicators = new List<Rectangle>();
        private readonly List<TextBlock> cellLabels = new List<TextBlock>();
        private bool isDraggingSelection = false;
        private Point dragStartPoint;
        private ResizeModeType currentResizeMode = ResizeModeType.None;
        private enum ResizeModeType { None, Move, ResizeRight, ResizeBottom, ResizeCorner }

        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public ResizeControlWindow()
        {
            InitializeComponent();
            ResultRow = 0;
            ResultColumn = 0;
            ResultRowSpan = 1;
            ResultColumnSpan = 1;
        }

        /// <summary>
        /// Creates a new resize control window
        /// </summary>
        /// <param name="tabName">The tab name containing the control</param>
        /// <param name="controlBorder">The border containing the control</param>
        /// <param name="row">Current row position</param>
        /// <param name="column">Current column position</param>
        /// <param name="rowSpan">Current row span</param>
        /// <param name="columnSpan">Current column span</param>
        public ResizeControlWindow(string tabName, Border controlBorder, int row, int column, int rowSpan, int columnSpan)
        {
            InitializeComponent();
            
            this.tabName = tabName;
            this.controlBorder = controlBorder;
            this.initialRow = row;
            this.initialColumn = column;
            this.initialRowSpan = rowSpan;
            this.initialColumnSpan = columnSpan;
            
            // Initialize properties with current values
            ResultRow = row;
            ResultColumn = column;
            ResultRowSpan = rowSpan;
            ResultColumnSpan = columnSpan;
            
            // Ensure window is modal
            this.Owner = Application.Current.MainWindow;
            
            // Setup keyboard navigation
            this.PreviewKeyDown += ResizeControlWindow_PreviewKeyDown;
            
            // Load grid information
            var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
            if (mainWindow != null)
            {
                var gridConfig = DatabaseMonolith.LoadGridConfig(tabName);
                gridMaxRows = Math.Max(4, gridConfig.Rows);
                gridMaxCols = Math.Max(4, gridConfig.Columns);
                
                // Update grid size text
                GridSizeText.Text = $"{gridMaxRows} x {gridMaxCols}";
                
                // Update tab name
                CurrentTabText.Text = tabName;
                
                // Determine control type from the contained control
                if (controlBorder.Child != null)
                {
                    var controlTypeName = controlBorder.Child.GetType().Name;
                    
                    // Map control types to friendly names
                    controlType = controlTypeName switch
                    {
                        "StockExplorer" => "Symbol Charts",
                        "TransactionsUserControl" => "Transactions",
                        "PredictionAnalysisControl" => "Prediction Analysis",
                        "SectorAnalysisHeatmapControl" => "Sector Momentum Heatmap",
                        _ => controlTypeName
                    };
                }
                
                // Update control type text
                ControlTypeText.Text = $"Resizing: {controlType}";
                
                // Load occupied cells
                occupiedCells = mainWindow.GetOccupiedCellsForTab(tabName);
                
                // Remove the current control's cells from the occupied list
                for (int r = row; r < row + rowSpan; r++)
                {
                    for (int c = column; c < column + columnSpan; c++)
                    {
                        occupiedCells.RemoveAll(cell => cell.Row == r && cell.Col == c);
                    }
                }
            }
            
            // Update the UI to reflect the loaded values
            this.Loaded += ResizeControlWindow_Loaded;
        }

        private void ResizeControlWindow_Loaded(object sender, RoutedEventArgs e)
        {
            EnsureGridExistsForTab(tabName);

            try
            {
                // Configure grid visualizer
                ConfigureGridVisualizer();
                
                // Configure sliders with proper ranges based on grid size
                RowSlider.Maximum = gridMaxRows - 1;
                ColumnSlider.Maximum = gridMaxCols - 1;
                RowSpanSlider.Maximum = gridMaxRows;
                ColumnSpanSlider.Maximum = gridMaxCols;
                
                // Initialize slider values
                UpdateControlValues(initialRow, initialColumn, initialRowSpan, initialColumnSpan);
                
                // Setup direct manipulation on GridVisualizer
                SetupGridVisualizerEvents();
                
                // Draw occupied cells
                DrawOccupiedCells();
                
                // Initial update of the selection indicator
                UpdateSelectionIndicator();
                
                // Update the related control information
                UpdateControlInfo();
                
                // Don't show instructions overlay initially (changed to hide by default)
                InstructionOverlay.Visibility = Visibility.Collapsed;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error initializing resize window: {ex.Message}", 
                    "Initialization Error", MessageBoxButton.OK, MessageBoxImage.Error);
                
                // Log the error
                //DatabaseMonolith.Log("Error", "ResizeControlWindow_Loaded failed", ex.ToString());
                
                // Set default values
                SetDefaultValues();
            }
        }
        
        private void ConfigureGridVisualizer()
        {
            // Clear existing definitions
            GridVisualizer.RowDefinitions.Clear();
            GridVisualizer.ColumnDefinitions.Clear();
            
            // Add the correct number of rows and columns
            for (int i = 0; i < gridMaxRows; i++)
            {
                GridVisualizer.RowDefinitions.Add(new RowDefinition());
            }
            
            for (int i = 0; i < gridMaxCols; i++)
            {
                GridVisualizer.ColumnDefinitions.Add(new ColumnDefinition());
            }
        }
        
        private void ResizeControlWindow_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Enter:
                    if (ApplyButton.IsEnabled)
                    {
                        ApplyButton_Click(this, new RoutedEventArgs());
                        e.Handled = true;
                    }
                    break;
                
                case Key.Escape:
                    CancelButton_Click(this, new RoutedEventArgs());
                    e.Handled = true;
                    break;
                
                case Key.Left:
                    if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                    {
                        // Decrease column span with Shift+Left if possible
                        if (ResultColumnSpan > 1)
                        {
                            UpdateControlValues(ResultRow, ResultColumn, ResultRowSpan, ResultColumnSpan - 1);
                            e.Handled = true;
                        }
                    }
                    else
                    {
                        // Move left if possible
                        if (ResultColumn > 0)
                        {
                            UpdateControlValues(ResultRow, ResultColumn - 1, ResultRowSpan, ResultColumnSpan);
                            e.Handled = true;
                        }
                    }
                    break;
                
                case Key.Right:
                    if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                    {
                        // Increase column span with Shift+Right if possible
                        if (ResultColumn + ResultColumnSpan < gridMaxCols)
                        {
                            UpdateControlValues(ResultRow, ResultColumn, ResultRowSpan, ResultColumnSpan + 1);
                            e.Handled = true;
                        }
                    }
                    else
                    {
                        // Move right if possible
                        if (ResultColumn + ResultColumnSpan < gridMaxCols)
                        {
                            UpdateControlValues(ResultRow, ResultColumn + 1, ResultRowSpan, ResultColumnSpan);
                            e.Handled = true;
                        }
                    }
                    break;
                
                case Key.Up:
                    if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                    {
                        // Decrease row span with Shift+Up if possible
                        if (ResultRowSpan > 1)
                        {
                            UpdateControlValues(ResultRow, ResultColumn, ResultRowSpan - 1, ResultColumnSpan);
                            e.Handled = true;
                        }
                    }
                    else
                    {
                        // Move up if possible
                        if (ResultRow > 0)
                        {
                            UpdateControlValues(ResultRow - 1, ResultColumn, ResultRowSpan, ResultColumnSpan);
                            e.Handled = true;
                        }
                    }
                    break;
                
                case Key.Down:
                    if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                    {
                        // Increase row span with Shift+Down if possible
                        if (ResultRow + ResultRowSpan < gridMaxRows)
                        {
                            UpdateControlValues(ResultRow, ResultColumn, ResultRowSpan + 1, ResultColumnSpan);
                            e.Handled = true;
                        }
                    }
                    else
                    {
                        // Move down if possible
                        if (ResultRow + ResultRowSpan < gridMaxRows)
                        {
                            UpdateControlValues(ResultRow, ResultColumn + 1, ResultRowSpan, ResultColumnSpan);
                            e.Handled = true;
                        }
                    }
                    break;
            }
            
            if (e.Handled)
            {
                // Update UI when keyboard changes were made
                UpdateSelectionIndicator();
                UpdateControlInfo();
            }
        }
        
        private void SetDefaultValues()
        {
            // Set safe default values if there was an error
            ResultRow = initialRow;
            ResultColumn = initialColumn;
            ResultRowSpan = initialRowSpan;
            ResultColumnSpan = initialColumnSpan;
        }
        
        private void UpdateControlValues(int row, int column, int rowSpan, int columnSpan)
        {
            try
            {
                // Record the current values before updating
                ResultRow = row;
                ResultColumn = column;
                ResultRowSpan = rowSpan;
                ResultColumnSpan = columnSpan;
                
                // Update display values (always using 1-based indexing now)
                int displayRow = row + 1;
                int displayColumn = column + 1;
                
                // Update sliders without triggering events (to avoid recursion)
                RowSlider.ValueChanged -= RowSlider_ValueChanged;
                ColumnSlider.ValueChanged -= ColumnSlider_ValueChanged;
                RowSpanSlider.ValueChanged -= RowSpanSlider_ValueChanged;
                ColumnSpanSlider.ValueChanged -= ColumnSpanSlider_ValueChanged;
                
                RowSlider.Value = row;
                ColumnSlider.Value = column;
                RowSpanSlider.Value = rowSpan;
                ColumnSpanSlider.Value = columnSpan;
                
                RowSlider.ValueChanged += RowSlider_ValueChanged;
                ColumnSlider.ValueChanged += ColumnSlider_ValueChanged;
                RowSpanSlider.ValueChanged += RowSpanSlider_ValueChanged;
                ColumnSpanSlider.ValueChanged += ColumnSpanSlider_ValueChanged;
                
                // Update textboxes without triggering events
                RowTextBox.TextChanged -= RowTextBox_TextChanged;
                ColumnTextBox.TextChanged -= ColumnTextBox_TextChanged;
                RowSpanTextBox.TextChanged -= RowSpanTextBox_TextChanged;
                ColumnSpanTextBox.TextChanged -= ColumnSpanTextBox_TextChanged;
                
                RowTextBox.Text = displayRow.ToString();
                ColumnTextBox.Text = displayColumn.ToString();
                RowSpanTextBox.Text = rowSpan.ToString();
                ColumnSpanTextBox.Text = columnSpan.ToString();
                
                RowTextBox.TextChanged += RowTextBox_TextChanged;
                ColumnTextBox.TextChanged += ColumnTextBox_TextChanged;
                RowSpanTextBox.TextChanged += RowSpanTextBox_TextChanged;
                ColumnSpanTextBox.TextChanged += ColumnSpanTextBox_TextChanged;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "UpdateControlValues failed", ex.ToString());
            }
        }

        private void SetupGridVisualizerEvents()
        {
            GridVisualizer.MouseLeftButtonDown += GridVisualizer_MouseLeftButtonDown;
            GridVisualizer.MouseMove += GridVisualizer_MouseMove;
            GridVisualizer.MouseLeftButtonUp += GridVisualizer_MouseLeftButtonUp;
            
            // Add mouse events to the selection indicator for direct manipulation
            SelectionIndicator.MouseLeftButtonDown += SelectionIndicator_MouseLeftButtonDown;
            SelectionIndicator.MouseMove += SelectionIndicator_MouseMove;
            SelectionIndicator.MouseLeftButtonUp += SelectionIndicator_MouseLeftButtonUp;
            
            // Add cursor feedback
            SelectionIndicator.MouseEnter += (s, e) => Cursor = Cursors.SizeAll;
            SelectionIndicator.MouseLeave += (s, e) => Cursor = Cursors.Arrow;
        }
        
        private void SelectionIndicator_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            isDraggingSelection = true;
            SelectionIndicator.CaptureMouse();
            
            // Get drag start point
            dragStartPoint = e.GetPosition(GridVisualizer);
            
            // Determine resize mode based on where the mouse is within the selection
            var pos = e.GetPosition(SelectionIndicator);
            
            // Calculate edge thresholds (12 pixels from the edges)
            double rightEdge = SelectionIndicator.ActualWidth - 12;
            double bottomEdge = SelectionIndicator.ActualHeight - 12;
            bool nearRightEdge = pos.X > rightEdge;
            bool nearBottomEdge = pos.Y > bottomEdge;
            
            if (nearRightEdge && nearBottomEdge)
                currentResizeMode = ResizeModeType.ResizeCorner;
            else if (nearRightEdge)
                currentResizeMode = ResizeModeType.ResizeRight;
            else if (nearBottomEdge)
                currentResizeMode = ResizeModeType.ResizeBottom;
            else
                currentResizeMode = ResizeModeType.Move;
            
            // Update cursor based on mode
            switch (currentResizeMode)
            {
                case ResizeModeType.Move:
                    Cursor = Cursors.SizeAll;
                    break;
                case ResizeModeType.ResizeRight:
                    Cursor = Cursors.SizeWE;
                    break;
                case ResizeModeType.ResizeBottom:
                    Cursor = Cursors.SizeNS;
                    break;
                case ResizeModeType.ResizeCorner:
                    Cursor = Cursors.SizeNWSE;
                    break;
            }
            
            e.Handled = true;
        }

        private void SelectionIndicator_MouseMove(object sender, MouseEventArgs e)
        {
            if (isDraggingSelection)
            {
                Point currentPosition = e.GetPosition(GridVisualizer);
                
                // Calculate the delta movement in grid cell units
                double cellWidth = GridVisualizer.ActualWidth / gridMaxCols;
                double cellHeight = GridVisualizer.ActualHeight / gridMaxRows;
                
                int deltaRows = 0;
                int deltaCols = 0;
                int deltaRowSpan = 0;
                int deltaColSpan = 0;
                
                switch (currentResizeMode)
                {
                    case ResizeModeType.Move:
                        // Calculate movement in cells
                        deltaCols = (int)Math.Round((currentPosition.X - dragStartPoint.X) / cellWidth);
                        deltaRows = (int)Math.Round((currentPosition.Y - dragStartPoint.Y) / cellHeight);
                        
                        if (deltaRows != 0 || deltaCols != 0)
                        {
                            int newRow = Math.Max(0, Math.Min(ResultRow + deltaRows, gridMaxRows - ResultRowSpan));
                            int newCol = Math.Max(0, Math.Min(ResultColumn + deltaCols, gridMaxCols - ResultColumnSpan));
                            
                            // Only update if position has changed
                            if (newRow != ResultRow || newCol != ResultColumn)
                            {
                                UpdateControlValues(newRow, newCol, ResultRowSpan, ResultColumnSpan);
                                dragStartPoint = currentPosition;
                            }
                        }
                        break;
                    
                    case ResizeModeType.ResizeRight:
                        // Calculate width change in cells
                        deltaColSpan = (int)Math.Round((currentPosition.X - dragStartPoint.X) / cellWidth);
                        
                        if (deltaColSpan != 0)
                        {
                            int newColSpan = Math.Max(1, Math.Min(ResultColumnSpan + deltaColSpan, gridMaxCols - ResultColumn));
                            
                            // Only update if span has changed
                            if (newColSpan != ResultColumnSpan)
                            {
                                UpdateControlValues(ResultRow, ResultColumn, ResultRowSpan, newColSpan);
                                dragStartPoint = currentPosition;
                            }
                        }
                        break;
                    
                    case ResizeModeType.ResizeBottom:
                        // Calculate height change in cells
                        deltaRowSpan = (int)Math.Round((currentPosition.Y - dragStartPoint.Y) / cellHeight);
                        
                        if (deltaRowSpan != 0)
                        {
                            int newRowSpan = Math.Max(1, Math.Min(ResultRowSpan + deltaRowSpan, gridMaxRows - ResultRow));
                            
                            // Only update if span has changed
                            if (newRowSpan != ResultRowSpan)
                            {
                                UpdateControlValues(ResultRow, ResultColumn, newRowSpan, ResultColumnSpan);
                                dragStartPoint = currentPosition;
                            }
                        }
                        break;
                    
                    case ResizeModeType.ResizeCorner:
                        // Calculate both width and height changes in cells
                        deltaColSpan = (int)Math.Round((currentPosition.X - dragStartPoint.X) / cellWidth);
                        deltaRowSpan = (int)Math.Round((currentPosition.Y - dragStartPoint.Y) / cellHeight);
                        
                        if (deltaRowSpan != 0 || deltaColSpan != 0)
                        {
                            int newRowSpan = Math.Max(1, Math.Min(ResultRowSpan + deltaRowSpan, gridMaxRows - ResultRow));
                            int newColSpan = Math.Max(1, Math.Min(ResultColumnSpan + deltaColSpan, gridMaxCols - ResultColumn));
                            
                            // Only update if span has changed
                            if (newRowSpan != ResultRowSpan || newColSpan != ResultColumnSpan)
                            {
                                UpdateControlValues(ResultRow, ResultColumn, newRowSpan, newColSpan);
                                dragStartPoint = currentPosition;
                            }
                        }
                        break;
                }
                
                // Update visualization
                UpdateSelectionIndicator();
                UpdateControlInfo();
            }
            else
            {
                // Show appropriate cursor based on position when not dragging
                var pos = e.GetPosition(SelectionIndicator);
                double rightEdge = SelectionIndicator.ActualWidth - 12;
                double bottomEdge = SelectionIndicator.ActualHeight - 12;
                
                if (pos.X > rightEdge && pos.Y > bottomEdge)
                    Cursor = Cursors.SizeNWSE; // Corner resize
                else if (pos.X > rightEdge)
                    Cursor = Cursors.SizeWE; // Horizontal resize
                else if (pos.Y > bottomEdge)
                    Cursor = Cursors.SizeNS; // Vertical resize
                else
                    Cursor = Cursors.SizeAll; // Move
            }
        }

        private void SelectionIndicator_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (isDraggingSelection)
            {
                isDraggingSelection = false;
                currentResizeMode = ResizeModeType.None;
                SelectionIndicator.ReleaseMouseCapture();
                Cursor = Cursors.Arrow;
            }
        }

        private void GridVisualizer_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            // Only handle if we're not already dragging
            if (!isDraggingSelection)
            {
                Point clickPoint = e.GetPosition(GridVisualizer);
                
                // Convert click position to grid cell
                int col = (int)(clickPoint.X / (GridVisualizer.ActualWidth / gridMaxCols));
                int row = (int)(clickPoint.Y / (GridVisualizer.ActualHeight / gridMaxRows));
                
                // Check if this is a valid cell (not occupied)
                if (!occupiedCells.Any(cell => cell.Row == row && cell.Col == col))
                {
                    // Position the control at the clicked cell with default span of 1x1
                    UpdateControlValues(row, col, 1, 1);
                    UpdateSelectionIndicator();
                    UpdateControlInfo();
                }
            }
        }

        private void GridVisualizer_MouseMove(object sender, MouseEventArgs e)
        {
            // Update cursor for grid cells to show where clicks are valid
            if (!isDraggingSelection)
            {
                Point hoverPoint = e.GetPosition(GridVisualizer);
                
                // Convert hover position to grid cell
                int col = (int)(hoverPoint.X / (GridVisualizer.ActualWidth / gridMaxCols));
                int row = (int)(hoverPoint.Y / (GridVisualizer.ActualHeight / gridMaxRows));
                
                // Check if this is a valid cell (not occupied)
                if (col >= 0 && col < gridMaxCols && row >= 0 && row < gridMaxRows)
                {
                    if (!occupiedCells.Any(cell => cell.Row == row && cell.Col == col))
                    {
                        Cursor = Cursors.Hand; // Valid spot to place control
                    }
                    else
                    {
                        Cursor = Cursors.No; // Cell is occupied
                    }
                }
            }
        }

        private void GridVisualizer_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            // Nothing to do here for now
        }

        private void DrawOccupiedCells()
        {
            // Clear existing indicators
            foreach (var indicator in occupiedCellIndicators)
            {
                GridVisualizer.Children.Remove(indicator);
            }
            occupiedCellIndicators.Clear();
            
            // Clear existing cell labels
            foreach (var label in cellLabels)
            {
                GridVisualizer.Children.Remove(label);
            }
            cellLabels.Clear();
            
            // Add new indicators for occupied cells
            foreach (var cell in occupiedCells)
            {
                if (cell.Row < gridMaxRows && cell.Col < gridMaxCols)
                {
                    var indicator = new Rectangle
                    {
                        Fill = new SolidColorBrush(Color.FromArgb(100, 255, 0, 0)),
                        Stroke = new SolidColorBrush(Color.FromRgb(255, 0, 0)),
                        StrokeThickness = 1
                    };
                    
                    Grid.SetRow(indicator, cell.Row);
                    Grid.SetColumn(indicator, cell.Col);
                    
                    GridVisualizer.Children.Add(indicator);
                    occupiedCellIndicators.Add(indicator);
                }
            }
            
            // Add cell numbers for better visualization (always using 1-based indexing)
            for (int r = 0; r < gridMaxRows; r++)
            {
                for (int c = 0; c < gridMaxCols; c++)
                {
                    // Display as 1-based indices
                    int displayRow = r + 1;
                    int displayCol = c + 1;
                    
                    var cellLabel = new TextBlock
                    {
                        Text = $"{displayRow},{displayCol}",
                        Foreground = new SolidColorBrush(Color.FromArgb(150, 255, 255, 255)),
                        FontSize = 10,
                        FontWeight = FontWeights.SemiBold,
                        HorizontalAlignment = HorizontalAlignment.Center,
                        VerticalAlignment = VerticalAlignment.Center
                    };
                    
                    Grid.SetRow(cellLabel, r);
                    Grid.SetColumn(cellLabel, c);
                    
                    GridVisualizer.Children.Add(cellLabel);
                    cellLabels.Add(cellLabel);
                }
            }
        }
        
        private void UpdateSelectionIndicator()
        {
            try
            {
                // Update the visual selection indicator
                Grid.SetRow(SelectionIndicator, ResultRow);
                Grid.SetColumn(SelectionIndicator, ResultColumn);
                Grid.SetRowSpan(SelectionIndicator, ResultRowSpan);
                Grid.SetColumnSpan(SelectionIndicator, ResultColumnSpan);
                
                // Check if the new position is valid
                bool isValid = true;
                
                // Ensure the control stays within grid bounds
                if (ResultRow + ResultRowSpan > gridMaxRows || ResultColumn + ResultColumnSpan > gridMaxCols)
                {
                    isValid = false;
                    WarningTextBlock.Text = "⚠️ Control exceeds grid boundaries";
                }
                // Check for overlaps with other controls
                else
                {
                    for (int r = ResultRow; r < ResultRow + ResultRowSpan; r++)
                    {
                        for (int c = ResultColumn; c < ResultColumn + ResultColumnSpan; c++)
                        {
                            if (occupiedCells.Any(cell => cell.Row == r && cell.Col == c))
                            {
                                isValid = false;
                                WarningTextBlock.Text = "⚠️ This position would overlap with another control";
                                break;
                            }
                        }
                        if (!isValid) break;
                    }
                }
                
                // Update UI based on validity
                WarningBorder.Visibility = isValid ? Visibility.Collapsed : Visibility.Visible;
                WarningTextBlock.Visibility = isValid ? Visibility.Collapsed : Visibility.Visible;
                ApplyButton.IsEnabled = isValid;
                
                // Change indicator color for invalid placement
                if (isValid)
                {
                    SelectionIndicator.Background = new SolidColorBrush(Color.FromArgb(102, 75, 169, 248)); // #664BA9F8
                    SelectionIndicator.BorderBrush = new SolidColorBrush(Color.FromRgb(75, 169, 248)); // #FF4BA9F8
                }
                else
                {
                    SelectionIndicator.Background = new SolidColorBrush(Color.FromArgb(102, 255, 50, 50)); // Red with opacity
                    SelectionIndicator.BorderBrush = new SolidColorBrush(Color.FromRgb(255, 50, 50)); // Red
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "UpdateSelectionIndicator failed", ex.ToString());
            }
        }

        private void UpdateControlInfo()
        {
            // Get display row/col values - always 1-based now
            int displayRow = ResultRow + 1;
            int displayCol = ResultColumn + 1;
            
            // Update the control info display
            ControlInfoTextBlock.Text = $"{controlType} at position: Row {displayRow}, Column {displayCol}\n" +
                                        $"Size: {ResultRowSpan} row(s) × {ResultColumnSpan} column(s)";
        }
        
        private void ShowInstructions_Click(object sender, RoutedEventArgs e)
        {
            InstructionOverlay.Visibility = Visibility.Visible;
        }
        
        private void DismissInstructions_Click(object sender, RoutedEventArgs e)
        {
            InstructionOverlay.Visibility = Visibility.Collapsed;
            
            // Save user preference to not show instructions again
            DatabaseMonolith.SaveUserPreference("SkipResizeInstructions", "true");
        }

        #region Event Handlers for UI Controls
        
        private void RowSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (IsLoaded)
            {
                ResultRow = (int)e.NewValue;
                
                // Update the text box without triggering its event
                RowTextBox.TextChanged -= RowTextBox_TextChanged;
                RowTextBox.Text = (ResultRow + 1).ToString(); // Always 1-based now
                RowTextBox.TextChanged += RowTextBox_TextChanged;
                
                UpdateSelectionIndicator();
                UpdateControlInfo();
            }
        }

        private void ColumnSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (IsLoaded)
            {
                ResultColumn = (int)e.NewValue;
                
                // Update the text box without triggering its event
                ColumnTextBox.TextChanged -= ColumnTextBox_TextChanged;
                ColumnTextBox.Text = (ResultColumn + 1).ToString(); // Always 1-based now
                ColumnTextBox.TextChanged += ColumnTextBox_TextChanged;
                
                UpdateSelectionIndicator();
                UpdateControlInfo();
            }
        }

        private void RowSpanSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (IsLoaded)
            {
                ResultRowSpan = (int)e.NewValue;
                
                // Update the text box without triggering its event
                RowSpanTextBox.TextChanged -= RowSpanTextBox_TextChanged;
                RowSpanTextBox.Text = ResultRowSpan.ToString();
                RowSpanTextBox.TextChanged += RowSpanTextBox_TextChanged;
                
                UpdateSelectionIndicator();
                UpdateControlInfo();
            }
        }

        private void ColumnSpanSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (IsLoaded)
            {
                ResultColumnSpan = (int)e.NewValue;
                
                // Update the text box without triggering its event
                ColumnSpanTextBox.TextChanged -= ColumnSpanTextBox_TextChanged;
                ColumnSpanTextBox.Text = ResultColumnSpan.ToString();
                ColumnSpanTextBox.TextChanged += ColumnSpanTextBox_TextChanged;
                
                UpdateSelectionIndicator();
                UpdateControlInfo();
            }
        }

        private void RowTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (IsLoaded && int.TryParse(RowTextBox.Text, out int textValue))
            {
                // Convert from 1-based to 0-based since we're always using 1-based display now
                int actualValue = textValue - 1;
                
                // Ensure value is in valid range
                if (actualValue >= 0 && actualValue < gridMaxRows)
                {
                    // Update slider value (will trigger the slider's event)
                    RowSlider.Value = actualValue;
                }
            }
        }

        private void ColumnTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (IsLoaded && int.TryParse(ColumnTextBox.Text, out int textValue))
            {
                // Convert from 1-based to 0-based since we're always using 1-based display now
                int actualValue = textValue - 1;
                
                // Ensure value is in valid range
                if (actualValue >= 0 && actualValue < gridMaxCols)
                {
                    // Update slider value (will trigger the slider's event)
                    ColumnSlider.Value = actualValue;
                }
            }
        }

        private void RowSpanTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (IsLoaded && int.TryParse(RowSpanTextBox.Text, out int value))
            {
                if (value >= 1 && value <= gridMaxRows)
                {
                    RowSpanSlider.Value = value;
                }
            }
        }

        private void ColumnSpanTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (IsLoaded && int.TryParse(ColumnSpanTextBox.Text, out int value))
            {
                if (value >= 1 && value <= gridMaxCols)
                {
                    ColumnSpanSlider.Value = value;
                }
            }
        }
        
        private void ApplyButton_Click(object sender, RoutedEventArgs e)
        {
            IsApplied = true;
            this.DialogResult = true;
            this.Close();
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            IsApplied = false;
            this.DialogResult = false;
            this.Close();
        }
        
        #endregion

        // Dynamically create grids if they are missing during resizing
        private void EnsureGridExistsForTab(string tabName)
        {
            var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
            if (mainWindow != null)
            {
                var tabItem = mainWindow.MainTabControl.Items.OfType<TabItem>().FirstOrDefault(t => t.Header.ToString() == tabName);
                if (tabItem != null && tabItem.Content is not Grid)
                {
                    var grid = new Grid();

                    // Default grid dimensions (4x4)
                    for (int i = 0; i < 4; i++)
                    {
                        grid.RowDefinitions.Add(new RowDefinition());
                    }
                    for (int j = 0; j < 4; j++)
                    {
                        grid.ColumnDefinitions.Add(new ColumnDefinition());
                    }

                    tabItem.Content = grid;

                    // Log grid creation
                    //DatabaseMonolith.Log("Info", $"Created grid for tab '{tabName}' with default dimensions 4x4");
                }
            }
        }
    }
}
