using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using Quantra;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for SectorAnalysisHeatmapControl.xaml
    /// </summary>
    public partial class SectorAnalysisHeatmapControl : UserControl, INotifyPropertyChanged, IDisposable
    {
        private SectorMomentumService _sectorService;
        private string _currentTimeframe = "1m"; // Default to 1 month
        private bool _isUpdating = false;
        private Dictionary<string, List<SectorMomentumModel>> _sectorData;
        private DispatcherTimer _autoRefreshTimer;
        private bool _disposed = false;

        public SectorAnalysisHeatmapControl()
        {
            try
            {
                InitializeComponent();
                _sectorService = new SectorMomentumService();
                DataContext = this;

                // Initialize the refresh timer (refresh every 5 minutes)
                _autoRefreshTimer = new DispatcherTimer
                {
                    Interval = TimeSpan.FromMinutes(5)
                };
                _autoRefreshTimer.Tick += AutoRefreshTimer_Tick;
                _autoRefreshTimer.Start();

                // Load initial data
                LoadHeatmapData();

                DatabaseMonolith.Log("Info", "SectorAnalysisHeatmapControl initialized successfully");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error initializing SectorAnalysisHeatmapControl", ex.ToString());
                DisplayInitializationError("Failed to initialize control: " + ex.Message);
            }
        }

        private void DisplayInitializationError(string errorMessage)
        {
            try
            {
                // Clear any existing content
                if (HeatmapGrid != null)
                    HeatmapGrid.Children.Clear();

                // Add error message
                var errorTextBlock = new TextBlock
                {
                    Text = errorMessage,
                    Foreground = Brushes.Red,
                    FontWeight = FontWeights.Bold,
                    TextWrapping = TextWrapping.Wrap,
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center,
                    Margin = new Thickness(20)
                };

                if (HeatmapContainer != null)
                {
                    // Replace the grid with the error message
                    if (HeatmapContainer.Children.Count > 1)
                        HeatmapContainer.Children[1] = errorTextBlock;
                    else
                        HeatmapContainer.Children.Add(errorTextBlock);
                }

                if (StatusText != null)
                    StatusText.Text = "Error: Failed to load heatmap";
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error displaying initialization error", ex.ToString());
            }
        }

        private void TimeframeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            try
            {
                if (TimeframeComboBox.SelectedItem is ComboBoxItem selectedItem)
                {
                    string timeframe = selectedItem.Tag?.ToString();
                    if (!string.IsNullOrEmpty(timeframe) && timeframe != _currentTimeframe)
                    {
                        _currentTimeframe = timeframe;
                        LoadHeatmapData();
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error in TimeframeComboBox_SelectionChanged", ex.ToString());
                StatusText.Text = "Error changing timeframe";
            }
        }

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_isUpdating)
                    return;

                _isUpdating = true;
                StatusText.Text = "Refreshing data...";

                // Refresh data using async pattern instead of Task.Run
                await LoadHeatmapDataAsync(true);

                StatusText.Text = "Data refreshed successfully";
                _isUpdating = false;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error refreshing heatmap data", ex.ToString());
                StatusText.Text = "Error refreshing data";
                _isUpdating = false;
            }
        }

        private void AutoRefreshTimer_Tick(object sender, EventArgs e)
        {
            try
            {
                if (!_isUpdating)
                {
                    LoadHeatmapData();
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error in auto-refresh timer", ex.ToString());
            }
        }

        private async Task LoadHeatmapDataAsync(bool forceRefresh = false)
        {
            try
            {
                if (_isUpdating)
                    return;

                _isUpdating = true;
                StatusText.Text = "Loading sector data...";

                // Get sector data from service (assuming this might have async operations)
                await Task.Run(() => {
                    _sectorData = _sectorService.GetSectorMomentumData(_currentTimeframe, forceRefresh);
                });
                
                // Update the heatmap UI (now on UI thread)
                UpdateHeatmap();
                
                LastUpdatedText.Text = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";
                StatusText.Text = "Ready";

                _isUpdating = false;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error loading heatmap data", ex.ToString());
                StatusText.Text = "Error loading sector data";
                _isUpdating = false;
            }
        }

        // Keep the original method for backward compatibility
        private void LoadHeatmapData(bool forceRefresh = false)
        {
            try
            {
                if (_isUpdating)
                    return;

                _isUpdating = true;
                StatusText.Text = "Loading sector data...";

                // Get sector data from service
                _sectorData = _sectorService.GetSectorMomentumData(_currentTimeframe, forceRefresh);
                
                // Update the heatmap UI directly
                UpdateHeatmap();
                
                LastUpdatedText.Text = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";
                StatusText.Text = "Ready";

                _isUpdating = false;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error loading heatmap data", ex.ToString());
                StatusText.Text = "Error loading sector data";
                _isUpdating = false;
            }
        }

        private void UpdateHeatmap()
        {
            try
            {
                // Clear existing heatmap
                HeatmapGrid.Children.Clear();
                
                if (_sectorData == null || !_sectorData.Any())
                {
                    // Display placeholder data if no real data is available
                    CreatePlaceholderHeatmap();
                    return;
                }

                // Get all sectors and subsectors
                var sectors = _sectorData.Keys.OrderBy(s => s).ToList();
                
                // Set up the uniform grid dimensions
                int totalColumns = sectors.Count;
                int maxRowHeight = _sectorData.Values.Max(list => list.Count);
                
                HeatmapGrid.Rows = maxRowHeight + 1; // +1 for header row
                HeatmapGrid.Columns = totalColumns;
                
                // Add sector headers
                for (int i = 0; i < sectors.Count; i++)
                {
                    var sector = sectors[i];
                    
                    // Create sector header
                    var headerBorder = new Border
                    {
                        Background = new SolidColorBrush(Color.FromRgb(60, 60, 90)),
                        BorderBrush = new SolidColorBrush(Color.FromRgb(80, 80, 120)),
                        BorderThickness = new Thickness(1),
                        Padding = new Thickness(5),
                        Margin = new Thickness(1)
                    };
                    
                    var headerText = new TextBlock
                    {
                        Text = sector,
                        Foreground = Brushes.White,
                        FontWeight = FontWeights.Bold,
                        TextWrapping = TextWrapping.Wrap,
                        TextAlignment = TextAlignment.Center
                    };
                    
                    headerBorder.Child = headerText;
                    HeatmapGrid.Children.Add(headerBorder);
                }
                
                // Add subsector cells with momentum coloring
                for (int row = 1; row < HeatmapGrid.Rows; row++)
                {
                    for (int col = 0; col < sectors.Count; col++)
                    {
                        var sector = sectors[col];
                        var subsectors = _sectorData[sector];
                        
                        if (row - 1 < subsectors.Count)
                        {
                            var subsector = subsectors[row - 1];
                            var cellBorder = CreateHeatmapCell(subsector);
                            HeatmapGrid.Children.Add(cellBorder);
                        }
                        else
                        {
                            // Empty cell for sectors with fewer subsectors
                            var emptyBorder = new Border
                            {
                                Background = new SolidColorBrush(Color.FromRgb(30, 30, 45)),
                                BorderBrush = new SolidColorBrush(Color.FromRgb(40, 40, 55)),
                                BorderThickness = new Thickness(1),
                                Margin = new Thickness(1)
                            };
                            HeatmapGrid.Children.Add(emptyBorder);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error updating heatmap", ex.ToString());
                DisplayInitializationError("Failed to update heatmap: " + ex.Message);
            }
        }

        private Border CreateHeatmapCell(SectorMomentumModel data)
        {
            // Create a border for the cell
            var border = new Border
            {
                BorderThickness = new Thickness(1),
                Padding = new Thickness(5),
                Margin = new Thickness(1),
                Tag = data // Store the data for future reference/interaction
            };
            
            // Set background color based on momentum value
            border.Background = GetHeatmapColor(data.MomentumValue);
            border.BorderBrush = new SolidColorBrush(Color.FromArgb(100, 255, 255, 255));
            
            // Create a grid for the content
            var grid = new Grid();
            grid.RowDefinitions.Add(new RowDefinition()); // Symbol
            grid.RowDefinitions.Add(new RowDefinition()); // Momentum value
            
            // Add symbol
            var symbolText = new TextBlock
            {
                Text = data.Symbol,
                Foreground = Brushes.White,
                FontWeight = FontWeights.Bold,
                TextAlignment = TextAlignment.Center,
                HorizontalAlignment = HorizontalAlignment.Center
            };
            Grid.SetRow(symbolText, 0);
            grid.Children.Add(symbolText);
            
            // Add momentum value with percentage format
            var momentumText = new TextBlock
            {
                Text = $"{data.MomentumValue:P2}",
                Foreground = Brushes.White,
                FontSize = 11,
                TextAlignment = TextAlignment.Center,
                HorizontalAlignment = HorizontalAlignment.Center,
                Margin = new Thickness(0, 2, 0, 0)
            };
            Grid.SetRow(momentumText, 1);
            grid.Children.Add(momentumText);
            
            border.Child = grid;
            
            // Add tooltip
            border.ToolTip = new ToolTip
            {
                Content = $"{data.Name} ({data.Symbol})\nMomentum: {data.MomentumValue:P2}\nVolume: {data.Volume:N0}"
            };
            
            return border;
        }

        private SolidColorBrush GetHeatmapColor(double momentumValue)
        {
            // Convert -1.0 to 1.0 range to a color gradient (red to green)
            // Negative values: red gradient
            // Near zero: yellow
            // Positive values: green gradient
            
            if (momentumValue <= -0.1)
            {
                // Strong negative momentum: dark red to light red
                double factor = Math.Min(1.0, Math.Abs(momentumValue) * 5); // Scale for more visual distinction
                byte intensity = (byte)(183 + (factor * 72));
                return new SolidColorBrush(Color.FromRgb(intensity, 28, 28));
            }
            else if (momentumValue < 0)
            {
                // Slight negative momentum: orange to yellow
                double factor = Math.Abs(momentumValue) * 10; // Scale for more visual distinction
                byte red = 255;
                byte green = (byte)(157 + ((1 - factor) * 98));
                return new SolidColorBrush(Color.FromRgb(red, green, 0));
            }
            else if (momentumValue < 0.1)
            {
                // Slight positive momentum: yellow-green to light green
                double factor = momentumValue * 10; // Scale for more visual distinction
                byte red = (byte)(255 - (factor * 255));
                byte green = 215;
                return new SolidColorBrush(Color.FromRgb(red, green, 0));
            }
            else
            {
                // Strong positive momentum: light green to dark green
                double factor = Math.Min(1.0, momentumValue * 5); // Scale for more visual distinction
                byte intensity = (byte)(120 - (factor * 64));
                return new SolidColorBrush(Color.FromRgb(0, intensity, 0));
            }
        }

        private void CreatePlaceholderHeatmap()
        {
            try
            {
                // Create a placeholder heatmap with sample data
                var placeholderData = new Dictionary<string, List<SectorMomentumModel>>();
                
                // Sample sectors and subsectors with momentum values
                var sectors = new[] { "Technology", "Financial", "Energy", "Healthcare", "Industrial", "Materials", "Consumer" };
                
                var random = new Random();
                
                // Create placeholder data for each sector
                foreach (var sector in sectors)
                {
                    var subsectors = new List<SectorMomentumModel>();
                    
                    // Number of subsectors varies by sector
                    int subsectorCount = random.Next(3, 8);
                    
                    for (int i = 0; i < subsectorCount; i++)
                    {
                        // Generate random momentum values between -0.3 and 0.3
                        double momentum = (random.NextDouble() * 0.6) - 0.3;
                        
                        // Symbol is first letter of sector + number
                        string symbol = sector.Substring(0, 1) + i;
                        
                        subsectors.Add(new SectorMomentumModel
                        {
                            Name = $"{sector} Subsector {i+1}",
                            Symbol = symbol,
                            MomentumValue = momentum,
                            Volume = random.Next(100000, 10000000)
                        });
                    }
                    
                    placeholderData[sector] = subsectors;
                }
                
                // Update member variable and update the UI
                _sectorData = placeholderData;
                UpdateHeatmap();
                
                // Update status
                StatusText.Text = "No data available";
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error creating placeholder heatmap", ex.ToString());
            }
        }

        // Force the control to update its layout
        public void ForceLayoutUpdate()
        {
            try
            {
                // Re-run layout logic
                if (HeatmapGrid != null)
                {
                    UpdateHeatmap();
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error in ForceLayoutUpdate", ex.ToString());
            }
        }

        #region INotifyPropertyChanged implementation

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion

        #region IDisposable implementation

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    if (_autoRefreshTimer != null)
                    {
                        _autoRefreshTimer.Stop();
                        _autoRefreshTimer = null;
                    }
                }

                // Free unmanaged resources
                _disposed = true;
            }
        }

        #endregion
    }
}
