using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Linq;
using Quantra.DAL.Services;
using Quantra.ViewModels;

namespace Quantra
{
    public partial class CreateTabWindow : Window
    {
        private readonly CreateTabWindowViewModel _viewModel;
        public string NewTabName { get; private set; }
        public int GridRows { get; private set; } = 4;
        public int GridColumns { get; private set; } = 4;

        // Parameterless constructor for XAML designer support
        public CreateTabWindow()
        {
            InitializeComponent();
            GridRows = 4;
            GridColumns = 4;
        }

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public CreateTabWindow(CreateTabWindowViewModel viewModel)
        {
            InitializeComponent();
            
            _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
            DataContext = _viewModel;
            
            // Set window properties
            this.Owner = Application.Current.MainWindow;
            this.WindowStartupLocation = WindowStartupLocation.CenterOwner;
            this.ShowInTaskbar = false;
            
            // Subscribe to ViewModel events
            _viewModel.TabCreated += OnTabCreated;
            _viewModel.CloseRequested += OnCloseRequested;
            
            // Initialize UI
            GridRowsTextBox.Text = _viewModel.GridRows.ToString();
            GridColumnsTextBox.Text = _viewModel.GridColumns.ToString();
            
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

        /// <summary>
        /// Legacy constructor for compatibility
        /// </summary>
        public CreateTabWindow(UserSettingsService userSettingsService)
            : this(new CreateTabWindowViewModel(userSettingsService))
        {
        }

        private void OnTabCreated(object sender, CreateTabEventArgs e)
        {
            NewTabName = e.TabName;
            GridRows = e.GridRows;
            GridColumns = e.GridColumns;
            DialogResult = true;
        }

        private void OnCloseRequested(object sender, bool accepted)
        {
            DialogResult = accepted;
            Close();
        }

        protected override void OnClosed(EventArgs e)
        {
            // Unsubscribe from events to prevent memory leaks
            if (_viewModel != null)
            {
                _viewModel.TabCreated -= OnTabCreated;
                _viewModel.CloseRequested -= OnCloseRequested;
            }
            base.OnClosed(e);
        }
        
        private void GridDimension_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Update ViewModel from UI
            if (_viewModel != null)
            {
                if (int.TryParse(GridRowsTextBox.Text, out int rows))
                {
                    _viewModel.GridRows = rows;
                }
                if (int.TryParse(GridColumnsTextBox.Text, out int columns))
                {
                    _viewModel.GridColumns = columns;
                }
            }
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
            
            // Update ViewModel and execute command
            if (_viewModel != null)
            {
                _viewModel.TabName = tabName;
                _viewModel.GridRows = rows;
                _viewModel.GridColumns = columns;
                
                if (_viewModel.CreateCommand.CanExecute(null))
                {
                    _viewModel.CreateCommand.Execute(null);
                }
            }
        }
        
        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            _viewModel?.CancelCommand?.Execute(null);
        }
        
        private void ShowWarning(string message)
        {
            WarningTextBlock.Text = message;
            WarningTextBlock.Visibility = Visibility.Visible;
        }
    }
}
