using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Quantra
{
    /// <summary>
    /// A customizable border control that provides visual feedback for resizable areas
    /// </summary>
    public class ResizableBorder : Border
    {
        private Window parentWindow;
        private ResizeDirection currentResizeDirection = ResizeDirection.None;
        private bool isResizing = false;
        private Point startPoint;
        private Cursor originalCursor;

        // Increaed ResizeBorderThickness to ensure easier resizing with rounded corners
        public int ResizeBorderThickness { get; set; } = 10;

        public enum ResizeDirection
        {
            None,
            Left,
            Right,
            Top,
            Bottom,
            TopLeft,
            TopRight,
            BottomLeft,
            BottomRight
        }

        internal enum SC
        {
            SIZE = 0xF000
        }

        internal enum WM
        {
            SYSCOMMAND = 0x0112
        }

        /// <summary>
        /// Initializes a new instance of the ResizableBorder class
        /// </summary>
        public ResizableBorder()
        {
            this.Loaded += ResizableBorder_Loaded;
            this.Background = Brushes.Transparent;
        }

        private void ResizableBorder_Loaded(object sender, RoutedEventArgs e)
        {
            parentWindow = Window.GetWindow(this);
            if (parentWindow != null)
            {
                this.MouseLeftButtonDown += ResizableBorder_MouseLeftButtonDown;
                this.MouseLeftButtonUp += ResizableBorder_MouseLeftButtonUp;
                this.MouseMove += ResizableBorder_MouseMove;
                this.MouseLeave += ResizableBorder_MouseLeave;

                // Make sure the parent window has minimum dimensions
                if (parentWindow.MinWidth <= 0)
                    parentWindow.MinWidth = 300;
                if (parentWindow.MinHeight <= 0)
                    parentWindow.MinHeight = 200;
            }
        }

        private void ResizableBorder_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (parentWindow == null) return;

            ResizeDirection direction = GetResizeDirection(e.GetPosition(this));
            if (direction != ResizeDirection.None)
            {
                isResizing = true;
                currentResizeDirection = direction;
                startPoint = e.GetPosition(parentWindow);
                originalCursor = this.Cursor;
                this.CaptureMouse();
                e.Handled = true;
            }
        }

        private void ResizableBorder_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (isResizing)
            {
                isResizing = false;
                currentResizeDirection = ResizeDirection.None;
                this.ReleaseMouseCapture();
                this.Cursor = originalCursor;
                e.Handled = true;
            }
        }

        private void ResizableBorder_MouseMove(object sender, MouseEventArgs e)
        {
            if (parentWindow == null) return;

            if (isResizing)
            {
                HandleResize(e.GetPosition(parentWindow));
                return;
            }

            // Update cursor based on resize direction
            ResizeDirection direction = GetResizeDirection(e.GetPosition(this));
            UpdateCursor(direction);
        }

        private void ResizableBorder_MouseLeave(object sender, MouseEventArgs e)
        {
            if (!isResizing)
            {
                this.Cursor = Cursors.Arrow;
            }
        }

        private ResizeDirection GetResizeDirection(Point position)
        {
            if (parentWindow == null) return ResizeDirection.None;

            double width = this.ActualWidth;
            double height = this.ActualHeight;

            bool nearLeft = position.X <= ResizeBorderThickness;
            bool nearRight = position.X >= width - ResizeBorderThickness;
            bool nearTop = position.Y <= ResizeBorderThickness;
            bool nearBottom = position.Y >= height - ResizeBorderThickness;

            if (nearTop && nearLeft)
                return ResizeDirection.TopLeft;
            else if (nearTop && nearRight)
                return ResizeDirection.TopRight;
            else if (nearBottom && nearLeft)
                return ResizeDirection.BottomLeft;
            else if (nearBottom && nearRight)
                return ResizeDirection.BottomRight;
            else if (nearLeft)
                return ResizeDirection.Left;
            else if (nearRight)
                return ResizeDirection.Right;
            else if (nearTop)
                return ResizeDirection.Top;
            else if (nearBottom)
                return ResizeDirection.Bottom;
            else
                return ResizeDirection.None;
        }

        private void UpdateCursor(ResizeDirection direction)
        {
            switch (direction)
            {
                case ResizeDirection.Left:
                case ResizeDirection.Right:
                    this.Cursor = Cursors.SizeWE;
                    break;
                case ResizeDirection.Top:
                case ResizeDirection.Bottom:
                    this.Cursor = Cursors.SizeNS;
                    break;
                case ResizeDirection.TopLeft:
                case ResizeDirection.BottomRight:
                    this.Cursor = Cursors.SizeNWSE;
                    break;
                case ResizeDirection.TopRight:
                case ResizeDirection.BottomLeft:
                    this.Cursor = Cursors.SizeNESW;
                    break;
                default:
                    this.Cursor = Cursors.Arrow;
                    break;
            }
        }

        private void HandleResize(Point currentPosition)
        {
            if (parentWindow == null) return;

            double deltaX = currentPosition.X - startPoint.X;
            double deltaY = currentPosition.Y - startPoint.Y;
            double newWidth = parentWindow.Width;
            double newHeight = parentWindow.Height;

            switch (currentResizeDirection)
            {
                case ResizeDirection.Left:
                    newWidth = Math.Max(parentWindow.MinWidth, parentWindow.Width - deltaX);
                    if (newWidth != parentWindow.Width)
                    {
                        parentWindow.Width = newWidth;
                        parentWindow.Left += deltaX;
                    }
                    break;

                case ResizeDirection.Right:
                    newWidth = Math.Max(parentWindow.MinWidth, parentWindow.Width + deltaX);
                    parentWindow.Width = newWidth;
                    break;

                case ResizeDirection.Top:
                    newHeight = Math.Max(parentWindow.MinHeight, parentWindow.Height - deltaY);
                    if (newHeight != parentWindow.Height)
                    {
                        parentWindow.Height = newHeight;
                        parentWindow.Top += deltaY;
                    }
                    break;

                case ResizeDirection.Bottom:
                    newHeight = Math.Max(parentWindow.MinHeight, parentWindow.Height + deltaY);
                    parentWindow.Height = newHeight;
                    break;

                case ResizeDirection.TopLeft:
                    newWidth = Math.Max(parentWindow.MinWidth, parentWindow.Width - deltaX);
                    newHeight = Math.Max(parentWindow.MinHeight, parentWindow.Height - deltaY);

                    if (newWidth != parentWindow.Width)
                    {
                        parentWindow.Width = newWidth;
                        parentWindow.Left += deltaX;
                    }

                    if (newHeight != parentWindow.Height)
                    {
                        parentWindow.Height = newHeight;
                        parentWindow.Top += deltaY;
                    }
                    break;

                case ResizeDirection.TopRight:
                    newWidth = Math.Max(parentWindow.MinWidth, parentWindow.Width + deltaX);
                    newHeight = Math.Max(parentWindow.MinHeight, parentWindow.Height - deltaY);

                    parentWindow.Width = newWidth;

                    if (newHeight != parentWindow.Height)
                    {
                        parentWindow.Height = newHeight;
                        parentWindow.Top += deltaY;
                    }
                    break;

                case ResizeDirection.BottomLeft:
                    newWidth = Math.Max(parentWindow.MinWidth, parentWindow.Width - deltaX);
                    newHeight = Math.Max(parentWindow.MinHeight, parentWindow.Height + deltaY);

                    if (newWidth != parentWindow.Width)
                    {
                        parentWindow.Width = newWidth;
                        parentWindow.Left += deltaX;
                    }

                    parentWindow.Height = newHeight;
                    break;

                case ResizeDirection.BottomRight:
                    newWidth = Math.Max(parentWindow.MinWidth, parentWindow.Width + deltaX);
                    newHeight = Math.Max(parentWindow.MinHeight, parentWindow.Height + deltaY);

                    parentWindow.Width = newWidth;
                    parentWindow.Height = newHeight;
                    break;
            }

            startPoint = currentPosition;
        }
    }
}
