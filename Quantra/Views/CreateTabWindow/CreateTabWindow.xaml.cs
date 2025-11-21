using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Linq;
using Quantra.DAL.Services;

namespace Quantra
{
    public partial class CreateTabWindow : Window
    {
        public string NewTabName { get; private set; }
        public int GridRows { get; private set; } = 4;
        public int GridColumns { get; private set; } = 4;
        private readonly UserSettingsService _userSettingsService;

        // Parameterless constructor for XAML designer support
        public CreateTabWindow()
        {
            InitializeComponent();
            GridRows = 4;
            GridColumns = 4;
        }

        public CreateTabWindow(UserSettingsService userSettingsService)
        {
            InitializeComponent(); // Ensure this is called to initialize the UI components
            _userSettingsService = userSettingsService;
            // Set window properties
            this.Owner = Application.Current.MainWindow;
            this.WindowStartupLocation = WindowStartupLocation.CenterOwner;
            this.ShowInTaskbar = false;
            
            // Load default grid settings from user settings
            try
            {
                var settings = _userSettingsService.GetUserSettings();
                GridRows = Math.Max(1, settings.DefaultGridRows);
                GridColumns = Math.Max(1, settings.DefaultGridColumns);
                GridRowsTextBox.Text = GridRows.ToString();
                GridColumnsTextBox.Text = GridColumns.ToString();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to load default grid settings", ex.ToString());
                // Fall back to 4x4 grid
                GridRowsTextBox.Text = "4";
                GridColumnsTextBox.Text = "4";
            }
            
            // Set focus to tab name textbox
            this.Loaded += (s, e) => {
                TabNameTextBox.Focus();
                UpdateGridPreview(); // Initialize grid preview
                
                // Add text changed events for dynamic preview updates
                GridRowsTextBox.TextChanged += GridDimension_TextChanged;
                GridColumnsTextBox.TextChanged += GridDimension_TextChanged;
            };
            
            // Add keyboard shortcuts
            this.KeyDown += CreateTabWindow_KeyDown;
        }
        
        private void GridDimension_TextChanged(object sender, TextChangedEventArgs e)
        {
            UpdateGridPreview();
        }
        
        private void UpdateGridPreview()
        {
            try
            {
                // Parse dimensions
                if (int.TryParse(GridRowsTextBox.Text, out int rows) && 
                    int.TryParse(GridColumnsTextBox.Text, out int columns))
                {
                    // Validate dimensions
                    rows = Math.Max(1, Math.Min(rows, 20)); // Limit to reasonable size for preview
                    columns = Math.Max(1, Math.Min(columns, 20));
                    
                    // Update the grid preview
                    GridPreview.Children.Clear();
                    GridPreview.RowDefinitions.Clear();
                    GridPreview.ColumnDefinitions.Clear();
                    
                    // Add rows and columns
                    for (int i = 0; i < rows; i++)
                    {
                        GridPreview.RowDefinitions.Add(new RowDefinition());
                    }
                    
                    for (int i = 0; i < columns; i++)
                    {
                        GridPreview.ColumnDefinitions.Add(new ColumnDefinition());
                    }
                    
                    // Add subtle cell indicators
                    for (int r = 0; r < rows; r++)
                    {
                        for (int c = 0; c < columns; c++)
                        {
                            var cellBorder = new Border
                            {
                                BorderBrush = new SolidColorBrush(Color.FromArgb(80, 255, 255, 255)),
                                BorderThickness = new Thickness(1),
                                Background = new SolidColorBrush(Color.FromArgb(20, 75, 169, 248))
                            };
                            
                            Grid.SetRow(cellBorder, r);
                            Grid.SetColumn(cellBorder, c);
                            GridPreview.Children.Add(cellBorder);
                            
                            // Add cell coordinates (only for small grids to avoid clutter)
                            if (rows <= 8 && columns <= 8)
                            {
                                var textBlock = new TextBlock
                                {
                                    Text = $"{r+1},{c+1}",  // 1-based for user-friendly display
                                    Foreground = new SolidColorBrush(Color.FromArgb(120, 255, 255, 255)),
                                    FontSize = 10,
                                    HorizontalAlignment = HorizontalAlignment.Center,
                                    VerticalAlignment = VerticalAlignment.Center
                                };
                                
                                Grid.SetRow(textBlock, r);
                                Grid.SetColumn(textBlock, c);
                                GridPreview.Children.Add(textBlock);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", "Failed to update grid preview", ex.ToString());
                // Don't show error to user - preview isn't critical functionality
            }
        }
        
        private void CreateTabWindow_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                this.DialogResult = false;
                this.Close();
                e.Handled = true;
            }
            else if (e.Key == Key.Enter)
            {
                CreateButton_Click(sender, e);
                e.Handled = true;
            }
        }
        
        private void CreateButton_Click(object sender, RoutedEventArgs e)
        {
            // Validate tab name
            string tabName = TabNameTextBox.Text?.Trim();
            if (string.IsNullOrWhiteSpace(tabName))
            {
                ShowWarning("Please enter a tab name.");
                TabNameTextBox.Focus();
                return;
            }
            
            // Validate that tab name doesn't already exist
            var mainWindow = Application.Current.MainWindow as MainWindow;
            if (mainWindow != null)
            {
                var existingTabs = mainWindow.GetTabNames();
                if (existingTabs.Contains(tabName))
                {
                    ShowWarning($"A tab with the name '{tabName}' already exists. Please choose a different name.");
                    TabNameTextBox.Focus();
                    TabNameTextBox.SelectAll();
                    return;
                }
                
                // Validate that tab name isn't "+"
                if (tabName == "+")
                {
                    ShowWarning("Tab name cannot be '+'. Please choose a different name.");
                    TabNameTextBox.Focus();
                    TabNameTextBox.SelectAll();
                    return;
                }
            }
            
            // Parse and validate grid dimensions
            if (!int.TryParse(GridRowsTextBox.Text, out int rows) || rows < 1)
            {
                ShowWarning("Please enter a valid number of rows (minimum 1).");
                GridRowsTextBox.Focus();
                GridRowsTextBox.SelectAll();
                return;
            }
            
            if (!int.TryParse(GridColumnsTextBox.Text, out int columns) || columns < 1)
            {
                ShowWarning("Please enter a valid number of columns (minimum 1).");
                GridColumnsTextBox.Focus();
                GridColumnsTextBox.SelectAll();
                return;
            }
            
            // Save the values
            NewTabName = tabName;
            GridRows = rows;
            GridColumns = columns;
            
            // Return success
            this.DialogResult = true;
            this.Close();
        }
        
        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            this.DialogResult = false;
            this.Close();
        }
        
        private void ShowWarning(string message)
        {
            WarningTextBlock.Text = message;
            WarningTextBlock.Visibility = Visibility.Visible;
        }
    }
}
