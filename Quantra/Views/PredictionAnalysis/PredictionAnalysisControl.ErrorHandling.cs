using System;
using System.Diagnostics;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Quantra.Controls
{
    public partial class PredictionAnalysisControl : UserControl
    {
        // Helper method to display chart initialization errors
        private void DisplayChartInitializationError(string errorMessage)
        {
            try
            {
                // Create error message for charts
                //if (PredictionChart != null)
                //{
                //    var errorBlock = new TextBlock
                //    {
                //        Text = errorMessage,
                //        Foreground = Brushes.Red,
                //        Background = new SolidColorBrush(Color.FromArgb(50, 0, 0, 0)),
                //        TextWrapping = TextWrapping.Wrap,
                //        TextAlignment = TextAlignment.Center,
                //        VerticalAlignment = VerticalAlignment.Center,
                //        HorizontalAlignment = HorizontalAlignment.Center,
                //        Padding = new Thickness(10)
                //    };
                    
                //    // Replace chart content with error message
                //    Grid.SetRow(errorBlock, Grid.GetRow(PredictionChart));
                //    Grid.SetColumn(errorBlock, Grid.GetColumn(PredictionChart));
                    
                //    // Get the parent of the chart
                //    var parent = PredictionChart.Parent as Grid;
                //    if (parent != null)
                //    {
                //        parent.Children.Remove(PredictionChart);
                //        parent.Children.Add(errorBlock);
                //    }
                //}
                
                // Update status
                if (StatusText != null)
                {
                    StatusText.Text = "Error: Chart components failed to load";
                }
            }
            catch (Exception ex)
            {
                // Log with file and method info
                var method = MethodBase.GetCurrentMethod();
                string file = method.DeclaringType?.Name ?? "UnknownFile";
                string methodName = method.Name;
                string details = $"File: {file}, Method: {methodName}, Exception: {ex}";
                DatabaseMonolith.Log("Error", "Failed to display chart error UI", details);
            }
        }
        
        // Helper method to display general initialization errors
        private void DisplayInitializationError(string errorMessage)
        {
            // If running in the debugger, throw instead of displaying in the UI
            if (Debugger.IsAttached)
            {
                var method = MethodBase.GetCurrentMethod();
                string file = method.DeclaringType?.Name ?? "UnknownFile";
                string methodName = method.Name;
                string details = $"File: {file}, Method: {methodName}, Error: {errorMessage}";
                DatabaseMonolith.Log("Error", "Initialization error (debugger attached)", details);
                throw new Exception(errorMessage);
            }
            // Create a minimal UI with just an error message
            var grid = new Grid
            {
                Background = new SolidColorBrush(Color.FromRgb(30, 30, 48))
            };
            
            var errorBlock = new TextBlock
            {
                Text = errorMessage,
                Foreground = Brushes.Red,
                Background = new SolidColorBrush(Color.FromArgb(50, 0, 0, 0)),
                TextWrapping = TextWrapping.Wrap,
                TextAlignment = TextAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                HorizontalAlignment = HorizontalAlignment.Center,
                Padding = new Thickness(20),
                FontSize = 16
            };
            
            grid.Children.Add(errorBlock);
            
            // Replace all content with the error grid
            this.Content = grid;
        }
    }
}
