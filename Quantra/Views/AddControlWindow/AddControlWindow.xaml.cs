using Quantra.Controls;
using Quantra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;
using System.Runtime.InteropServices;
using static Quantra.ResizableBorder;

namespace Quantra
{
    /// <summary>
    /// Main AddControlWindow class - core logic and properties
    /// Functionalities split into partial classes for better organization
    /// </summary>
    public partial class AddControlWindow : Window
    {
        private string chosenControlDefinition;
        private HashSet<(int Row, int Column)> occupiedCells = new HashSet<(int Row, int Column)>();
        private static AddControlWindow _instance;
        private Rectangle selectionRect; // Represents currently selected position
        private bool isResizing = false;
        private ResizeDirection currentResizeDirection;
        private Point startPoint;
        private Cursor originalCursor;
        private const int resizeBorderThickness = 8;
        private bool useOneBased = true; // Default to 1-based indexing

        public AddControlWindow()
        {
            try
            {
                InitializeComponent();

                // Set window properties to make it non-modal
                this.Owner = Application.Current.MainWindow;
                this.WindowStartupLocation = WindowStartupLocation.CenterOwner;
                this.ShowInTaskbar = false;

                // Initialize resizing support
                this.MouseMove += Window_MouseMove;
                this.MouseLeave += Window_MouseLeave;
                this.MouseLeftButtonDown += Window_MouseLeftButtonDown;
                this.MouseLeftButtonUp += Window_MouseLeftButtonUp;

                // Add KeyDown event handler for Escape key
                this.KeyDown += (s, e) => {
                    if (e.Key == Key.Escape)
                    {
                        this.Close();
                    }
                };

                // Allow keyboard navigation
                this.PreviewKeyDown += AddControlWindow_PreviewKeyDown;

                // Set default values for row and column with 1-based indexing
                if (RowTextBox != null) RowTextBox.Text = "1";
                if (ColumnTextBox != null) ColumnTextBox.Text = "1";
                
                // Set default values for row and column span to 4x4
                if (RowSpanTextBox != null) RowSpanTextBox.Text = "4";
                if (ColumnSpanTextBox != null) ColumnSpanTextBox.Text = "4";

                // Initialize selection rectangle for grid visualization
                selectionRect = new Rectangle
                {
                    Fill = new SolidColorBrush(Color.FromArgb(100, 0, 255, 0)),
                    Stroke = new SolidColorBrush(Colors.LimeGreen),
                    StrokeThickness = 2
                };

                // Use the Loaded event to ensure UI elements are fully initialized before adding event handlers
                this.Loaded += (s, e) => {
                    // Load tabs after UI is initialized
                    LoadTabs();

                    // Handle tab selection change to update grid size display
                    if (TabComboBox != null) TabComboBox.SelectionChanged += TabComboBox_SelectionChanged;

                    // Add validation for position and span inputs
                    if (RowTextBox != null) RowTextBox.TextChanged += ValidatePosition;
                    if (ColumnTextBox != null) ColumnTextBox.TextChanged += ValidatePosition;
                    if (RowSpanTextBox != null) RowSpanTextBox.TextChanged += ValidatePosition;
                    if (ColumnSpanTextBox != null) ColumnSpanTextBox.TextChanged += ValidatePosition;

                    // Create initial grid visualization
                    UpdateGridVisualization();
                    
                    // Initialize span selection
                    InitializeSpanSelection();

                    // Ensure this window can be closed with the keyboard
                    this.Focusable = true;
                    this.Focus();
                };
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error initializing Add Control window: {ex.Message}",
                                "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// Static method to get or create the instance (singleton pattern)
        /// </summary>
        public static AddControlWindow GetInstance()
        {
            if (_instance == null || !_instance.IsLoaded)
            {
                _instance = new AddControlWindow();
                
                // Register for tab added events from MainWindow
                var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
                if (mainWindow != null)
                {
                    mainWindow.TabAdded -= _instance.RefreshTabs;
                    mainWindow.TabAdded += _instance.RefreshTabs;
                    
                    // Log that we're connecting to the event
                    //DatabaseMonolith.Log("Info", "AddControlWindow connected to MainWindow TabAdded event");
                }
            }
            else
            {
                // If window already exists, bring it to front and flash
                _instance.BringToFrontAndFlash();
                
                // Refresh tabs to ensure we have the latest tabs
                _instance.RefreshTabs();
            }
            return _instance;
        }

        private void TabComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Update grid size label when tab selection changes
            var selectedTab = TabComboBox.SelectedItem as string;
            if (selectedTab != null)
            {
                var gridConfig = DatabaseMonolith.LoadGridConfig(selectedTab);
                GridSizeLabel.Content = $"{gridConfig.Rows} x {gridConfig.Columns}";

                // Refresh the occupied cells for the selected tab
                RefreshOccupiedCells(selectedTab);

                // Reset validation since we've changed tabs
                ResetValidation();

                // Update grid visualization
                UpdateGridVisualization();
                
                // NEW: Synchronize tab selection with MainWindow
                SynchronizeMainWindowTab(selectedTab);
            }
        }
        
        /// <summary>
        /// Synchronizes the MainWindow's tab selection with the AddControlWindow's tab selection
        /// </summary>
        /// <param name="tabName">The name of the tab to select in MainWindow</param>
        private void SynchronizeMainWindowTab(string tabName)
        {
            try
            {
                // Get the MainWindow instance
                var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
                if (mainWindow != null)
                {
                    // Find the corresponding TabItem in MainWindow's TabControl
                    var mainTabControl = mainWindow.MainTabControl;
                    if (mainTabControl != null)
                    {
                        var targetTab = mainTabControl.Items.OfType<TabItem>()
                            .FirstOrDefault(t => t.Header?.ToString() == tabName);
                        
                        if (targetTab != null && mainTabControl.SelectedItem != targetTab)
                        {
                            // Select the tab in MainWindow
                            mainTabControl.SelectedItem = targetTab;
                            
                            // Optional: Log the synchronization for debugging
                            //DatabaseMonolith.Log("Info", $"Synchronized MainWindow tab selection to: {tabName}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                // Log error but don't interrupt the user experience
                //DatabaseMonolith.Log("Warning", $"Failed to synchronize MainWindow tab selection: {ex.Message}", ex.ToString());
            }
        }

        private void ResetValidation()
        {
            AddButton.IsEnabled = true;
            OverlapWarningTextBlock.Visibility = Visibility.Collapsed;
        }

        private void AddButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedTab = TabComboBox.SelectedItem as string;
            var selectedControl = ControlComboBox.SelectedItem as ComboBoxItem;
            var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();

            // Validate all input fields
            if (selectedTab == null || selectedControl == null)
            {
                if (mainWindow != null)
                    mainWindow.AppendAlert("Please select a tab and control type.", "warning");
                else
                    MessageBox.Show("Please select a tab and control type.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            if (!int.TryParse(RowTextBox.Text, out int row) ||
                !int.TryParse(ColumnTextBox.Text, out int column) ||
                !int.TryParse(RowSpanTextBox.Text, out int rowSpan) ||
                !int.TryParse(ColumnSpanTextBox.Text, out int columnSpan))
            {
                if (mainWindow != null)
                    mainWindow.AppendAlert("Please provide valid numeric values for row, column, and spans.", "warning");
                else
                    MessageBox.Show("Please provide valid numeric values.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // Convert from 1-based to 0-based indexing if needed
            if (useOneBased)
            {
                row--;
                column--;
            }

            // Ensure spans are at least 1
            if (rowSpan < 1 || columnSpan < 1)
            {
                if (mainWindow != null)
                    mainWindow.AppendAlert("Row span and column span must be at least 1.", "warning");
                else
                    MessageBox.Show("Spans must be at least 1.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // Improved control type extraction with more robust handling
            string controlType = null;
            
            try
            {
                if (selectedControl.Content is TextBlock textBlock)
                {
                    // Direct TextBlock content
                    controlType = textBlock.Text;
                }
                else if (selectedControl.Content is StackPanel stackPanel)
                {
                    // StackPanel with TextBlock inside (as seen in XAML)
                    var textBlockInPanel = stackPanel.Children.OfType<TextBlock>().FirstOrDefault();
                    if (textBlockInPanel != null)
                    {
                        controlType = textBlockInPanel.Text;
                    }
                    else
                    {
                        // Fallback: try to get text from any text-containing element
                        foreach (var child in stackPanel.Children)
                        {
                            if (child is TextBlock tb)
                            {
                                controlType = tb.Text;
                                break;
                            }
                            else if (child is ContentPresenter cp && cp.Content is string str)
                            {
                                controlType = str;
                                break;
                            }
                        }
                    }
                }
                else if (selectedControl.Content is string stringContent)
                {
                    // Direct string content
                    controlType = stringContent;
                }
                else if (selectedControl.Content != null)
                {
                    // Last resort: use ToString() on the content
                    controlType = selectedControl.Content.ToString();
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error extracting control type from ComboBox selection", ex.ToString());
                controlType = null;
            }
            
            // If we couldn't extract the control type, show an error
            if (string.IsNullOrEmpty(controlType))
            {
                if (mainWindow != null)
                    mainWindow.AppendAlert("Could not determine control type.", "warning");
                else
                    MessageBox.Show("Could not determine control type.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            } 

            // Log the control type to help with debugging
            //DatabaseMonolith.Log("Info", $"Adding control of type: '{controlType}'");

            chosenControlDefinition = $"{controlType},{row},{column},{rowSpan},{columnSpan}";

            try
            {
                if (mainWindow != null)
                {
                    // Try to add the control to the tab - the MainWindow.AddControlToTab method
                    // will check for overlaps again as a final validation
                    mainWindow.AddControlToTab(selectedTab, controlType, row, column, rowSpan, columnSpan);

                    // Force refresh the tab to ensure controls render properly - especially for Prediction Analysis
                    if (controlType == "Prediction Analysis")
                    {
                        //DatabaseMonolith.Log("Info", $"Special handling for Prediction Analysis control - ensuring visibility");
                        
                        // Force refresh of the tab immediately to ensure control is properly initialized
                        mainWindow.RefreshTabControls(selectedTab);
                    }

                    // Notify success
                    mainWindow.AppendAlert($"Added {controlType} control to {selectedTab} tab", "positive");

                    // Keep the row and column for easy positioning of multiple controls
                    RowSpanTextBox.Text = "4";  // Reset span to default 4x4
                    ColumnSpanTextBox.Text = "4";  // Reset span to default 4x4

                    // Refresh the occupied cells to reflect the newly added control
                    RefreshOccupiedCells(selectedTab);

                    // Don't close the window - keep it open for additional controls
                }
            }
            catch (Exception ex)
            {
                if (mainWindow != null)
                    mainWindow.AppendAlert($"Error adding control: {ex.Message}", "negative");
                else
                    MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }

        // Override OnClosing to make sure we clean up properly when closing
        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            base.OnClosing(e);

            // Ensure we clear the instance when closing
            if (_instance == this)
            {
                _instance = null;
            }
        }

        // Override the OnClosed method to unregister from the event
        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            
            // Unregister from MainWindow events
            var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
            if (mainWindow != null)
            {
                mainWindow.TabAdded -= this.RefreshTabs;
            }
            
            // Ensure we clear the instance when closing
            if (_instance == this)
            {
                _instance = null;
            }
        }

        /// <summary>
        /// Initializes the window with a specific tab and cell position
        /// </summary>
        /// <param name="tabName">The name of the tab to select</param>
        /// <param name="row">The row position (0-based)</param>
        /// <param name="col">The column position (0-based)</param>
        public void InitializeWithPosition(string tabName, int row, int col)
        {
            try
            {
                // Need to wait for UI to be fully initialized
                if (!this.IsLoaded)
                {
                    this.Loaded += (s, e) => InitializePositionAfterLoaded(tabName, row, col);
                }
                else
                {
                    InitializePositionAfterLoaded(tabName, row, col);
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to initialize AddControlWindow with position: {ex.Message}", ex.ToString());
            }
        }

        /// <summary>
        /// Helper method to set position after window is loaded
        /// </summary>
        private void InitializePositionAfterLoaded(string tabName, int row, int col)
        {
            // Select the tab
            if (TabComboBox != null && TabComboBox.Items.Contains(tabName))
            {
                TabComboBox.SelectedItem = tabName;
            }

            // Convert from 0-based to 1-based if needed
            int displayRow = useOneBased ? row + 1 : row;
            int displayCol = useOneBased ? col + 1 : col;

            // Set the position
            if (RowTextBox != null)
            {
                RowTextBox.Text = displayRow.ToString();
            }

            if (ColumnTextBox != null)
            {
                ColumnTextBox.Text = displayCol.ToString();
            }

            // Set default spans to 4x4
            if (RowSpanTextBox != null)
            {
                RowSpanTextBox.Text = "4";
            }

            if (ColumnSpanTextBox != null)
            {
                ColumnSpanTextBox.Text = "4";
            }

            // Refresh occupation information for validation
            RefreshOccupiedCells(tabName);

            // Update the grid visualization
            UpdateGridVisualization();

            // Validate the position
            UpdatePositionValidation();

            // Log success
            //DatabaseMonolith.Log("Info", $"Initialized AddControlWindow with tab '{tabName}' at position ({displayRow},{displayCol})");
        }

        private void Window_MouseMove(object sender, MouseEventArgs e)
        {
            if (isResizing)
            {
                HandleResize(e.GetPosition(this));
            }
            else
            {
                // Get the mouse position and determine if we're over a resize border
                Point position = e.GetPosition(this);
                ResizeDirection direction = GetResizeDirection(position);
                
                // Update cursor based on resize direction
                if (direction != ResizeDirection.None)
                {
                    this.Cursor = GetCursorForResizeDirection(direction);
                }
                else
                {
                    // Reset cursor when not over a resize edge/corner
                    this.Cursor = originalCursor ?? Cursors.Arrow;
                }
            }
        }

        private void Window_MouseLeave(object sender, MouseEventArgs e)
        {
            // Reset cursor when mouse leaves the window
            this.Cursor = originalCursor ?? Cursors.Arrow;
        }

        private ResizeDirection GetResizeDirection(Point mousePosition)
        {
            // If we're not at a window edge, return None
            if (mousePosition.X > resizeBorderThickness && 
                mousePosition.X < this.ActualWidth - resizeBorderThickness &&
                mousePosition.Y > resizeBorderThickness && 
                mousePosition.Y < this.ActualHeight - resizeBorderThickness)
            {
                return ResizeDirection.None;
            }

            // Determine which edge or corner we're on
            bool onLeftEdge = mousePosition.X <= resizeBorderThickness;
            bool onRightEdge = mousePosition.X >= this.ActualWidth - resizeBorderThickness;
            bool onTopEdge = mousePosition.Y <= resizeBorderThickness;
            bool onBottomEdge = mousePosition.Y >= this.ActualHeight - resizeBorderThickness;

            // Check for corners first
            if (onLeftEdge && onTopEdge) return ResizeDirection.TopLeft;
            if (onRightEdge && onTopEdge) return ResizeDirection.TopRight;
            if (onLeftEdge && onBottomEdge) return ResizeDirection.BottomLeft;
            if (onRightEdge && onBottomEdge) return ResizeDirection.BottomRight;

            // Check for edges
            if (onLeftEdge) return ResizeDirection.Left;
            if (onRightEdge) return ResizeDirection.Right;
            if (onTopEdge) return ResizeDirection.Top;
            if (onBottomEdge) return ResizeDirection.Bottom;

            return ResizeDirection.None;
        }

        private Cursor GetCursorForResizeDirection(ResizeDirection direction)
        {
            switch (direction)
            {
                case ResizeDirection.Left:
                case ResizeDirection.Right:
                    return Cursors.SizeWE;
                case ResizeDirection.Top:
                case ResizeDirection.Bottom:
                    return Cursors.SizeNS;
                case ResizeDirection.TopLeft:
                case ResizeDirection.BottomRight:
                    return Cursors.SizeNWSE;
                case ResizeDirection.TopRight:
                case ResizeDirection.BottomLeft:
                    return Cursors.SizeNESW;
                default:
                    return originalCursor ?? Cursors.Arrow;
            }
        }

        private void HandleResize(Point currentPosition)
        {
            if (!isResizing) return;

            // Calculate the change in position
            double deltaX = currentPosition.X - startPoint.X;
            double deltaY = currentPosition.Y - startPoint.Y;

            // Apply resize based on the current resize direction
            switch (currentResizeDirection)
            {
                case ResizeDirection.Left:
                    if (this.Width - deltaX >= this.MinWidth)
                    {
                        this.Left += deltaX;
                        this.Width -= deltaX;
                    }
                    break;

                case ResizeDirection.Right:
                    if (this.Width + deltaX >= this.MinWidth)
                    {
                        this.Width += deltaX;
                        startPoint.X = currentPosition.X;
                    }
                    break;

                case ResizeDirection.Top:
                    if (this.Height - deltaY >= this.MinHeight)
                    {
                        this.Top += deltaY;
                        this.Height -= deltaY;
                    }
                    break;

                case ResizeDirection.Bottom:
                    if (this.Height + deltaY >= this.MinHeight)
                    {
                        this.Height += deltaY;
                        startPoint.Y = currentPosition.Y;
                    }
                    break;

                case ResizeDirection.TopLeft:
                    if (this.Width - deltaX >= this.MinWidth && this.Height - deltaY >= this.MinHeight)
                    {
                        this.Left += deltaX;
                        this.Width -= deltaX;
                        this.Top += deltaY;
                        this.Height -= deltaY;
                    }
                    break;

                case ResizeDirection.TopRight:
                    if (this.Width + deltaX >= this.MinWidth && this.Height - deltaY >= this.MinHeight)
                    {
                        this.Width += deltaX;
                        startPoint.X = currentPosition.X;
                        this.Top += deltaY;
                        this.Height -= deltaY;
                    }
                    break;

                case ResizeDirection.BottomLeft:
                    if (this.Width - deltaX >= this.MinWidth && this.Height + deltaY >= this.MinHeight)
                    {
                        this.Left += deltaX;
                        this.Width -= deltaX;
                        this.Height += deltaY;
                        startPoint.Y = currentPosition.Y;
                    }
                    break;

                case ResizeDirection.BottomRight:
                    if (this.Width + deltaX >= this.MinWidth && this.Height + deltaY >= this.MinHeight)
                    {
                        this.Width += deltaX;
                        this.Height += deltaY;
                        startPoint.X = currentPosition.X;
                        startPoint.Y = currentPosition.Y;
                    }
                    break;
            }
            
            // Update the grid visualization during resize
            UpdateGridVisualization();
        }

        private void Window_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            // Save the original cursor before we start resizing
            originalCursor = this.Cursor;
            
            // Determine if we're on a resize edge/corner
            Point position = e.GetPosition(this);
            currentResizeDirection = GetResizeDirection(position);
            
            // If we clicked on a resize border, start resizing
            if (currentResizeDirection != ResizeDirection.None)
            {
                isResizing = true;
                startPoint = position;
                this.CaptureMouse();
                e.Handled = true;
            }
        }

        private void Window_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (isResizing)
            {
                isResizing = false;
                this.ReleaseMouseCapture();
                this.Cursor = originalCursor ?? Cursors.Arrow;
                e.Handled = true;
            }
        }
    }
}