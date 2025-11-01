using Quantra.Controls;
using Quantra;
using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;

namespace Quantra
{
    /// <summary>
    /// Partial class for Window initialization and management functionality
    /// </summary>
    public partial class AddControlWindow : Window
    {
        // Constructor removed to avoid duplication with the one in AddControlWindow.xaml.cs

        private void AddControlWindow_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                this.Close();
                e.Handled = true;
            }
            else if (e.Key == Key.Enter && AddButton.IsEnabled)
            {
                // When Enter is pressed and the Add button is enabled, click it
                AddButton_Click(AddButton, new RoutedEventArgs());
                e.Handled = true;
            }
        }

        // Method to bring window to front and flash to get attention
        private void BringToFrontAndFlash()
        {
            // Bring to front
            this.Activate();
            this.Focus();

            // Flash the window (change background and revert)
            var originalBackground = this.Background;

            var animation = new ColorAnimation
            {
                From = Colors.Yellow,
                To = (originalBackground as SolidColorBrush)?.Color ?? Colors.DarkBlue,
                Duration = new Duration(TimeSpan.FromSeconds(0.5)),
                AutoReverse = true,
                RepeatBehavior = new RepeatBehavior(3)
            };

            var brush = new SolidColorBrush(Colors.Yellow);
            this.Background = brush;
            brush.BeginAnimation(SolidColorBrush.ColorProperty, animation);

            // Flash the window using Win32 API (more noticeable)
            var wih = new WindowInteropHelper(this);
            FlashWindow.Flash(wih.Handle, 5);
        }

        private void LoadTabs()
        {
            RefreshTabs();
        }

        // Add this method to refresh the tab list
        public void RefreshTabs(string newTabName = null)
        {
            // Execute on UI thread
            this.Dispatcher.BeginInvoke(new Action(() =>
            {
                try
                {
                    // Store the current selection if any
                    string currentSelection = TabComboBox?.SelectedItem as string;

                    // Retrieve tabs from the MainWindow
                    var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
                    if (mainWindow != null)
                    {
                        var tabs = mainWindow.GetTabNames();

                        // Update the tab list
                        if (TabComboBox != null)
                        {
                            TabComboBox.ItemsSource = tabs;

                            // Select the new tab if specified, otherwise restore the previous selection
                            if (!string.IsNullOrEmpty(newTabName) && tabs.Contains(newTabName))
                            {
                                TabComboBox.SelectedItem = newTabName;
                            }
                            else if (!string.IsNullOrEmpty(currentSelection) && tabs.Contains(currentSelection))
                            {
                                TabComboBox.SelectedItem = currentSelection;
                            }
                            else if (tabs.Any())
                            {
                                TabComboBox.SelectedIndex = 0;
                            }

                            // Log the refresh
                            //DatabaseMonolith.Log("Info", $"Refreshed tab list in AddControlWindow. Tab count: {tabs.Count}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", $"Error refreshing tabs in AddControlWindow: {ex.Message}", ex.ToString());
                }
            }));
        }

        // IMPORTANT: WindowResizeDirection and WindowHandleResize are renamed to avoid ambiguity
        // with similarly named methods in the main AddControlWindow.xaml.cs

        private ResizableBorder.ResizeDirection GetWindowResizeDirection(Point mousePosition)
        {
            double width = this.ActualWidth;
            double height = this.ActualHeight;

            // Check if mouse is in bottom-left corner
            if (mousePosition.Y > height - resizeBorderThickness && mousePosition.X < resizeBorderThickness)
                return ResizableBorder.ResizeDirection.BottomLeft;

            // Check if mouse is in bottom-right corner
            if (mousePosition.Y > height - resizeBorderThickness && mousePosition.X > width - resizeBorderThickness)
                return ResizableBorder.ResizeDirection.BottomRight;

            // Check if mouse is at bottom edge
            if (mousePosition.Y > height - resizeBorderThickness)
                return ResizableBorder.ResizeDirection.Bottom;

            return ResizableBorder.ResizeDirection.None;
        }

        // Renamed to WindowHandleResize to avoid ambiguity
        private void WindowHandleResize(Point currentPosition)
        {
            double deltaY = currentPosition.Y - startPoint.Y;
            double deltaX = currentPosition.X - startPoint.X;
            double newHeight = this.Height;
            double newWidth = this.Width;

            switch (currentResizeDirection)
            {
                case ResizableBorder.ResizeDirection.Bottom:
                    newHeight = Math.Max(this.MinHeight, this.Height + deltaY);
                    this.Height = newHeight;
                    break;

                case ResizableBorder.ResizeDirection.BottomLeft:
                    newHeight = Math.Max(this.MinHeight, this.Height + deltaY);
                    newWidth = Math.Max(this.MinWidth, this.Width - deltaX);

                    this.Height = newHeight;
                    this.Width = newWidth;
                    this.Left += this.Width != newWidth ? deltaX : 0;
                    break;

                case ResizableBorder.ResizeDirection.BottomRight:
                    newHeight = Math.Max(this.MinHeight, this.Height + deltaY);
                    newWidth = Math.Max(this.MinWidth, this.Width + deltaX);

                    this.Height = newHeight;
                    this.Width = newWidth;
                    break;
            }

            startPoint = currentPosition;
        }

        // IMPORTANT: Removing the duplicate event handlers that conflict with AddControlWindow.xaml.cs
        // The event wiring will remain in the constructor, but the implementation will come from xaml.cs
    }
}
