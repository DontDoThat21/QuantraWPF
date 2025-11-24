using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Quantra.Utilities
{
    /// <summary>
    /// An adorner that provides resize handles for controls in the grid
    /// </summary>
    public class ControlResizingAdorner : Adorner
    {
        // IMPROVEMENT: Increased handle size for easier targeting
        private const double HANDLE_SIZE = 16; // Increased from 14
        // IMPROVEMENT: Increased corner threshold for more forgiving detection
        private const double CORNER_THRESHOLD = 20; // Increased from 15
        // IMPROVEMENT: Added edge threshold for detecting edges
        private const double EDGE_THRESHOLD = 15; // New threshold for edges

        private Brush handleFill = new SolidColorBrush(Color.FromArgb(180, 75, 169, 248));
        private Brush handleStroke = new SolidColorBrush(Color.FromRgb(75, 169, 248));
        // IMPROVEMENT: Different fill for bottom handles to provide visual feedback
        private Brush bottomHandleFill = new SolidColorBrush(Color.FromArgb(200, 120, 200, 255));
        private Pen previewPen;
        private Brush previewFill = new SolidColorBrush(Color.FromArgb(50, 0, 255, 0)); // Semi-transparent green
        private Brush gridCellBrush = new SolidColorBrush(Color.FromArgb(100, 0, 255, 0)); // Semi-transparent green for cells
        private Pen gridCellPen = new Pen(new SolidColorBrush(Color.FromRgb(0, 255, 0)), 1.5); // Green pen for grid cells

        // Visual elements for resize handles
        private readonly Rectangle topLeftHandle;
        private readonly Rectangle topRightHandle;
        private readonly Rectangle bottomLeftHandle;
        private readonly Rectangle bottomRightHandle;
        // IMPROVEMENT: Added bottom center handle
        private readonly Rectangle bottomCenterHandle;
        // IMPROVEMENT: Added left, right, and top center handles
        private readonly Rectangle leftCenterHandle;
        private readonly Rectangle rightCenterHandle;
        private readonly Rectangle topCenterHandle;
        private readonly VisualCollection visualChildren;

        // Track resize operation
        private ResizeMode currentResizeMode = ResizeMode.None;
        private Point startPoint;
        private Rect originalRect;
        private Rect currentRect;
        private bool isResizing = false;
        private bool showPreview = false;
        private bool showGrid = true;

        // Callback for resizing
        private Action<int, int, int, int> resizeCallback;
        private Grid parentGrid;
        private int gridRows;
        private int gridColumns;

        // IMPROVEMENT: Enhanced resize mode enum with additional edge resize modes
        private enum ResizeMode
        {
            None,
            TopLeft,
            TopRight,
            BottomLeft,
            BottomRight,
            TopEdge,
            BottomEdge,
            LeftEdge,
            RightEdge
        }

        /// <summary>
        /// Creates a new resizing adorner for the specified element
        /// </summary>
        /// <param name="adornedElement">The element being adorned</param>
        /// <param name="parentGrid">The parent grid containing the element</param>
        /// <param name="rows">Number of rows in the grid</param>
        /// <param name="columns">Number of columns in the grid</param>
        /// <param name="callback">Callback function for when resize is completed</param>
        public ControlResizingAdorner(UIElement adornedElement, Grid parentGrid, int rows, int columns,
            Action<int, int, int, int> callback) : base(adornedElement)
        {
            this.parentGrid = parentGrid;
            gridRows = rows;
            gridColumns = columns;
            resizeCallback = callback;

            // Create the preview pen
            previewPen = new Pen(new SolidColorBrush(Color.FromArgb(200, 0, 255, 0)), 2);

            // Initialize the visual collection
            visualChildren = new VisualCollection(this);

            // Create resize handles
            topLeftHandle = CreateHandle();
            topRightHandle = CreateHandle();
            bottomLeftHandle = CreateHandle(true); // Mark as bottom handle
            bottomRightHandle = CreateHandle(true); // Mark as bottom handle
            // IMPROVEMENT: Added bottom center handle
            bottomCenterHandle = CreateHandle(true); // Mark as bottom handle
            // IMPROVEMENT: Added left, right, and top center handles
            leftCenterHandle = CreateHandle();
            rightCenterHandle = CreateHandle();
            topCenterHandle = CreateHandle();


            // Add handles to visual collection
            visualChildren.Add(topLeftHandle);
            visualChildren.Add(topRightHandle);
            visualChildren.Add(bottomLeftHandle);
            visualChildren.Add(bottomRightHandle);
            visualChildren.Add(bottomCenterHandle);
            visualChildren.Add(leftCenterHandle);
            visualChildren.Add(rightCenterHandle);
            visualChildren.Add(topCenterHandle);

            // Set up mouse events
            MouseLeftButtonDown += OnMouseLeftButtonDown;
            MouseLeftButtonUp += OnMouseLeftButtonUp;
            MouseMove += OnMouseMove;

            // Set up cursor feedback
            MouseEnter += OnMouseEnter;
            MouseLeave += OnMouseLeave;

            // Initialize with current rect
            originalRect = new Rect(AdornedElement.RenderSize);
            currentRect = originalRect;

            // Disable automatic grid and preview - only show when actively resizing
            showPreview = false;
            showGrid = false;
        }

        // IMPROVEMENT: Modified to create either regular or bottom handle with optional parameters
        private Rectangle CreateHandle(bool isBottomOrEdgeHandle = false)
        {
            return new Rectangle
            {
                Width = HANDLE_SIZE,
                Height = HANDLE_SIZE,
                Fill = isBottomOrEdgeHandle ? bottomHandleFill : handleFill, // Use different fill for bottom/edge handles
                Stroke = handleStroke,
                StrokeThickness = 1.5, // IMPROVEMENT: Slightly thicker stroke for better visibility
                RadiusX = 3, // IMPROVEMENT: Slightly more rounded corners
                RadiusY = 3
            };
        }

        /// <summary>
        /// Shows or hides the preview rectangle
        /// </summary>
        public void ShowPreview(bool show)
        {
            showPreview = show;
            InvalidateVisual();
        }

        /// <summary>
        /// Shows or hides the grid visualization
        /// </summary>
        public void ShowGridVisuals(bool show)
        {
            showGrid = show;
            InvalidateVisual();
        }

        /// <summary>
        /// Gets the grid cell position and span from a rectangle
        /// </summary>
        private (int Row, int Column, int RowSpan, int ColumnSpan) GetCellsFromRect(Rect rect)
        {
            if (parentGrid == null || parentGrid.ActualWidth == 0 || parentGrid.ActualHeight == 0)
                return (0, 0, 1, 1);

            double cellWidth = parentGrid.ActualWidth / gridColumns;
            double cellHeight = parentGrid.ActualHeight / gridRows;

            int row = Math.Max(0, Math.Min(gridRows - 1, (int)(rect.Top / cellHeight)));
            int column = Math.Max(0, Math.Min(gridColumns - 1, (int)(rect.Left / cellWidth)));

            // Calculate rowSpan and columnSpan, ensuring at least 1 and not exceeding grid bounds
            int rowSpan = Math.Max(1, Math.Min(gridRows - row, (int)Math.Ceiling(rect.Height / cellHeight)));
            int columnSpan = Math.Max(1, Math.Min(gridColumns - column, (int)Math.Ceiling(rect.Width / cellWidth)));

            return (row, column, rowSpan, columnSpan);
        }

        /// <summary>
        /// Gets a rectangle based on grid cell positions
        /// </summary>
        private Rect GetRectFromCells(int row, int column, int rowSpan, int columnSpan)
        {
            if (parentGrid == null || parentGrid.ActualWidth == 0 || parentGrid.ActualHeight == 0)
                return new Rect();

            double cellWidth = parentGrid.ActualWidth / gridColumns;
            double cellHeight = parentGrid.ActualHeight / gridRows;

            double left = column * cellWidth;
            double top = row * cellHeight;
            double width = columnSpan * cellWidth;
            double height = rowSpan * cellHeight;

            return new Rect(left, top, width, height);
        }

        protected override void OnRender(DrawingContext drawingContext)
        {
            if (AdornedElement != null)
            {
                // Calculate control's bounds
                Rect adornedRect = new Rect(AdornedElement.RenderSize);

                // Position the resize handles
                PositionHandles(adornedRect);

                // Only draw grid lines when actively resizing and showGrid is enabled
                if (isResizing && showGrid && parentGrid != null)
                {
                    double cellWidth = parentGrid.ActualWidth / gridColumns;
                    double cellHeight = parentGrid.ActualHeight / gridRows;

                    // Draw all grid cells with light lines
                    for (int r = 0; r <= gridRows; r++)
                    {
                        // Draw horizontal grid lines
                        drawingContext.DrawLine(
                            new Pen(new SolidColorBrush(Color.FromArgb(60, 255, 255, 255)), 1),
                            new Point(0, r * cellHeight),
                            new Point(parentGrid.ActualWidth, r * cellHeight));
                    }

                    for (int c = 0; c <= gridColumns; c++)
                    {
                        // Draw vertical grid lines
                        drawingContext.DrawLine(
                            new Pen(new SolidColorBrush(Color.FromArgb(60, 255, 255, 255)), 1),
                            new Point(c * cellWidth, 0),
                            new Point(c * cellWidth, parentGrid.ActualHeight));
                    }
                }

                // IMPROVEMENT: Draw a subtle highlight along the bottom edge for better visibility
                if (!isResizing)
                {
                    drawingContext.DrawLine(
                        new Pen(new SolidColorBrush(Color.FromArgb(120, 120, 200, 255)), 2),
                        new Point(adornedRect.Left, adornedRect.Bottom),
                        new Point(adornedRect.Right, adornedRect.Bottom));
                }

                // Draw preview rectangle if resizing
                if (isResizing && showPreview)
                {
                    // Get cell-aligned rectangle for the current drag position
                    var (row, col, rowSpan, colSpan) = GetCellsFromRect(currentRect);
                    var cellAlignedRect = GetRectFromCells(row, col, rowSpan, colSpan);

                    // Draw the preview rectangle
                    drawingContext.DrawRectangle(previewFill, previewPen, cellAlignedRect);

                    if (showGrid)
                    {
                        // Draw grid visualization
                        double cellWidth = parentGrid.ActualWidth / gridColumns;
                        double cellHeight = parentGrid.ActualHeight / gridRows;

                        // Draw highlighted cells
                        for (int r = row; r < row + rowSpan; r++)
                        {
                            for (int c = col; c < col + colSpan; c++)
                            {
                                Rect cellRect = new Rect(
                                    c * cellWidth,
                                    r * cellHeight,
                                    cellWidth,
                                    cellHeight);

                                drawingContext.DrawRectangle(null, gridCellPen, cellRect);
                            }
                        }
                    }

                    // Draw coordinates text
                    FormattedText text = new FormattedText(
                        $"({row + 1},{col + 1}) {rowSpan}x{colSpan}",
                        System.Globalization.CultureInfo.CurrentCulture,
                        FlowDirection.LeftToRight,
                        new Typeface("Segoe UI"),
                        12,
                        Brushes.White,
                        VisualTreeHelper.GetDpi(this).PixelsPerDip);

                    Point textPos = new Point(cellAlignedRect.Left + 4, cellAlignedRect.Top + 4);
                    drawingContext.DrawRectangle(
                        new SolidColorBrush(Color.FromArgb(150, 0, 0, 0)),
                        null,
                        new Rect(textPos, new Size(text.Width + 8, text.Height + 2)));
                    drawingContext.DrawText(text, textPos);
                }
            }
        }

        private void PositionHandles(Rect adornedRect)
        {
            // Position resize handles at the corners
            topLeftHandle.SetValue(Canvas.LeftProperty, adornedRect.Left - HANDLE_SIZE / 2);
            topLeftHandle.SetValue(Canvas.TopProperty, adornedRect.Top - HANDLE_SIZE / 2);

            topRightHandle.SetValue(Canvas.LeftProperty, adornedRect.Right - HANDLE_SIZE / 2);
            topRightHandle.SetValue(Canvas.TopProperty, adornedRect.Top - HANDLE_SIZE / 2);

            bottomLeftHandle.SetValue(Canvas.LeftProperty, adornedRect.Left - HANDLE_SIZE / 2);
            bottomLeftHandle.SetValue(Canvas.TopProperty, adornedRect.Bottom - HANDLE_SIZE / 2);

            bottomRightHandle.SetValue(Canvas.LeftProperty, adornedRect.Right - HANDLE_SIZE / 2);
            bottomRightHandle.SetValue(Canvas.TopProperty, adornedRect.Bottom - HANDLE_SIZE / 2);

            // IMPROVEMENT: Position bottom center handle
            bottomCenterHandle.SetValue(Canvas.LeftProperty, adornedRect.Left + adornedRect.Width / 2 - HANDLE_SIZE / 2);
            bottomCenterHandle.SetValue(Canvas.TopProperty, adornedRect.Bottom - HANDLE_SIZE / 2);

            // IMPROVEMENT: Position top, left, and right center handles
            topCenterHandle.SetValue(Canvas.LeftProperty, adornedRect.Left + adornedRect.Width / 2 - HANDLE_SIZE / 2);
            topCenterHandle.SetValue(Canvas.TopProperty, adornedRect.Top - HANDLE_SIZE / 2);

            leftCenterHandle.SetValue(Canvas.LeftProperty, adornedRect.Left - HANDLE_SIZE / 2);
            leftCenterHandle.SetValue(Canvas.TopProperty, adornedRect.Top + adornedRect.Height / 2 - HANDLE_SIZE / 2);

            rightCenterHandle.SetValue(Canvas.LeftProperty, adornedRect.Right - HANDLE_SIZE / 2);
            rightCenterHandle.SetValue(Canvas.TopProperty, adornedRect.Top + adornedRect.Height / 2 - HANDLE_SIZE / 2);
        }

        private void OnMouseEnter(object sender, MouseEventArgs e)
        {
            // Only update cursor if actively resizing
            if (isResizing)
            {
                UpdateCursor(e.GetPosition(this));
            }
        }

        private void OnMouseLeave(object sender, MouseEventArgs e)
        {
            // Always reset cursor when leaving the adorner
            Cursor = Cursors.Arrow;
        }

        private void OnMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (AdornedElement == null)
                return;

            Point position = e.GetPosition(this);
            Rect adornedRect = new Rect(AdornedElement.RenderSize);

            // IMPROVEMENT: Enhanced corner and edge detection with more forgiving thresholds
            currentResizeMode = GetResizeMode(position, adornedRect);

            // If not on a handle or edge, do nothing
            if (currentResizeMode == ResizeMode.None)
                return;

            // Start resize operation
            startPoint = position;
            originalRect = adornedRect;
            currentRect = adornedRect;
            isResizing = true;
            ShowPreview(true);
            ShowGridVisuals(true);
            CaptureMouse();
            e.Handled = true;
        }

        // IMPROVEMENT: New helper method to determine resize mode with enhanced detection
        private ResizeMode GetResizeMode(Point position, Rect adornedRect)
        {
            // Check corners first (with enhanced detection area)
            if (IsCloseToPoint(position, adornedRect.TopLeft, CORNER_THRESHOLD))
                return ResizeMode.TopLeft;
            if (IsCloseToPoint(position, adornedRect.TopRight, CORNER_THRESHOLD))
                return ResizeMode.TopRight;
            if (IsCloseToPoint(position, adornedRect.BottomLeft, CORNER_THRESHOLD))
                return ResizeMode.BottomLeft;
            if (IsCloseToPoint(position, adornedRect.BottomRight, CORNER_THRESHOLD))
                return ResizeMode.BottomRight;

            // IMPROVEMENT: Check edges
            if (IsCloseToHorizontalEdge(position, adornedRect.Top, adornedRect.Left, adornedRect.Right, EDGE_THRESHOLD))
                return ResizeMode.TopEdge;
            if (IsCloseToHorizontalEdge(position, adornedRect.Bottom, adornedRect.Left, adornedRect.Right, EDGE_THRESHOLD))
                return ResizeMode.BottomEdge;
            if (IsCloseToVerticalEdge(position, adornedRect.Left, adornedRect.Top, adornedRect.Bottom, EDGE_THRESHOLD))
                return ResizeMode.LeftEdge;
            if (IsCloseToVerticalEdge(position, adornedRect.Right, adornedRect.Top, adornedRect.Bottom, EDGE_THRESHOLD))
                return ResizeMode.RightEdge;

            return ResizeMode.None;
        }

        // IMPROVEMENT: New helper method to check if point is close to an edge
        private bool IsCloseToHorizontalEdge(Point point, double yEdge, double xStart, double xEnd, double threshold)
        {
            return Math.Abs(point.Y - yEdge) <= threshold && point.X >= xStart + CORNER_THRESHOLD && point.X <= xEnd - CORNER_THRESHOLD;
        }

        private bool IsCloseToVerticalEdge(Point point, double xEdge, double yStart, double yEnd, double threshold)
        {
            return Math.Abs(point.X - xEdge) <= threshold && point.Y >= yStart + CORNER_THRESHOLD && point.Y <= yEnd - CORNER_THRESHOLD;
        }

        private void OnMouseMove(object sender, MouseEventArgs e)
        {
            if (AdornedElement == null)
                return;

            Point position = e.GetPosition(this);

            if (isResizing)
            {
                // Calculate the change in position
                Vector delta = position - startPoint;
                Rect newRect = originalRect; // Start with the original rect for calculations

                // Update the new rectangle based on which handle is being dragged
                switch (currentResizeMode)
                {
                    case ResizeMode.TopLeft:
                        newRect = new Rect(
                            originalRect.Left + delta.X,
                            originalRect.Top + delta.Y,
                            Math.Max(0, originalRect.Width - delta.X),
                            Math.Max(0, originalRect.Height - delta.Y));
                        break;

                    case ResizeMode.TopRight:
                        newRect = new Rect(
                            originalRect.Left,
                            originalRect.Top + delta.Y,
                            Math.Max(0, originalRect.Width + delta.X),
                            Math.Max(0, originalRect.Height - delta.Y));
                        break;

                    case ResizeMode.BottomLeft:
                        newRect = new Rect(
                            originalRect.Left + delta.X,
                            originalRect.Top,
                            Math.Max(0, originalRect.Width - delta.X),
                            Math.Max(0, originalRect.Height + delta.Y));
                        break;

                    case ResizeMode.BottomRight:
                        newRect = new Rect(
                            originalRect.Left,
                            originalRect.Top,
                            Math.Max(0, originalRect.Width + delta.X),
                            Math.Max(0, originalRect.Height + delta.Y));
                        break;
                    case ResizeMode.TopEdge:
                        newRect = new Rect(
                            originalRect.Left,
                            originalRect.Top + delta.Y,
                            Math.Max(0, originalRect.Width),
                            Math.Max(0, originalRect.Height - delta.Y));
                        break;
                    case ResizeMode.BottomEdge:
                        newRect = new Rect(
                            originalRect.Left,
                            originalRect.Top,
                            Math.Max(0, originalRect.Width),
                            Math.Max(0, originalRect.Height + delta.Y));
                        break;
                    case ResizeMode.LeftEdge:
                        newRect = new Rect(
                            originalRect.Left + delta.X,
                            originalRect.Top,
                            Math.Max(0, originalRect.Width - delta.X),
                            Math.Max(0, originalRect.Height));
                        break;
                    case ResizeMode.RightEdge:
                        newRect = new Rect(
                            originalRect.Left,
                            originalRect.Top,
                            Math.Max(0, originalRect.Width + delta.X),
                            Math.Max(0, originalRect.Height));
                        break;
                }

                // Validate bounds of the freely-dragged rectangle before snapping
                ValidateResizeBounds(ref newRect);

                // Snap the validated newRect to the grid cells to determine the currentRect for preview and finalization
                var (row, col, rowSpan, colSpan) = GetCellsFromRect(newRect);
                currentRect = GetRectFromCells(row, col, rowSpan, colSpan);

                // Redraw using the snapped currentRect
                InvalidateVisual();
            }
            else
            {
                // Update cursor based on position
                UpdateCursor(position);
            }
        }

        private void OnMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (!isResizing)
                return;

            isResizing = false;
            ReleaseMouseCapture();
            ShowPreview(false);
            ShowGridVisuals(false);

            // Calculate the final grid cell coordinates and spans
            var (row, column, rowSpan, columnSpan) = GetCellsFromRect(currentRect);

            // Don't allow resizing to zero dimensions
            if (rowSpan < 1) rowSpan = 1;
            if (columnSpan < 1) columnSpan = 1;

            // Call the resize callback with the new position and size
            resizeCallback?.Invoke(row, column, rowSpan, columnSpan);

            // Reset cursor
            Cursor = Cursors.Arrow;

            // IMPROVEMENT: Update currentRect to the new size of the adorned element
            // This ensures that the hover preview is correct after resizing.
            if (AdornedElement != null)
            {
                currentRect = new Rect(AdornedElement.RenderSize);
            }

            e.Handled = true;
        }

        private void UpdateCursor(Point position)
        {
            if (AdornedElement == null)
                return;

            // Only show resize cursors when actively resizing
            if (!isResizing)
            {
                Cursor = Cursors.Arrow;
                return;
            }

            Rect adornedRect = new Rect(AdornedElement.RenderSize);

            // IMPROVEMENT: Use the same enhanced detection for cursor updates
            var hoverMode = GetResizeMode(position, adornedRect);

            switch (hoverMode)
            {
                case ResizeMode.TopLeft:
                case ResizeMode.BottomRight:
                    Cursor = Cursors.SizeNWSE;
                    break;
                case ResizeMode.TopRight:
                case ResizeMode.BottomLeft:
                    Cursor = Cursors.SizeNESW;
                    break;
                case ResizeMode.TopEdge:
                case ResizeMode.BottomEdge:
                    Cursor = Cursors.SizeNS;
                    break;
                case ResizeMode.LeftEdge:
                case ResizeMode.RightEdge:
                    Cursor = Cursors.SizeWE;
                    break;
                default:
                    Cursor = Cursors.SizeAll; // Middle (for dragging)
                    break;
            }
        }

        private bool IsCloseToPoint(Point point, Point targetPoint, double threshold)
        {
            // IMPROVEMENT: Use distance calculation for smoother corner detection
            return (point - targetPoint).Length <= threshold;
        }

        protected override Size MeasureOverride(Size constraint)
        {
            // Size of the adorner is determined by the AdornedElement
            return AdornedElement.RenderSize;
        }

        protected override Size ArrangeOverride(Size finalSize)
        {
            // Make sure each handle is properly positioned
            Rect adornedRect = new Rect(AdornedElement.RenderSize);
            PositionHandles(adornedRect);
            return finalSize;
        }

        protected override Visual GetVisualChild(int index)
        {
            return visualChildren[index];
        }

        protected override int VisualChildrenCount
        {
            get { return visualChildren.Count; }
        }

        // Add validation to ensure resizing respects grid boundaries
        private void ValidateResizeBounds(ref Rect rect)
        {
            if (parentGrid == null || parentGrid.ActualWidth == 0 || parentGrid.ActualHeight == 0)
                return;

            double cellWidth = parentGrid.ActualWidth / gridColumns;
            double cellHeight = parentGrid.ActualHeight / gridRows;

            // IMPROVEMENT: Enforce minimum size of at least one grid cell for better UX
            rect.Width = Math.Max(rect.Width, cellWidth * 0.8);
            rect.Height = Math.Max(rect.Height, cellHeight * 0.8);

            // Ensure the rectangle stays within grid bounds
            rect.X = Math.Max(0, Math.Min(rect.X, parentGrid.ActualWidth - rect.Width));
            rect.Y = Math.Max(0, Math.Min(rect.Y, parentGrid.ActualHeight - rect.Height));
            rect.Width = Math.Min(rect.Width, parentGrid.ActualWidth - rect.X);
            rect.Height = Math.Min(rect.Height, parentGrid.ActualHeight - rect.Y);
        }
    }
}
