using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Threading.Tasks;
using Quantra.Models;
using Quantra.DAL.Services;

namespace Quantra.Views.Backtesting
{
    /// <summary>
    /// Interaction logic for CustomBenchmarkDialog.xaml
    /// </summary>
    public partial class CustomBenchmarkDialog : Window
    {
        private CustomBenchmark _benchmark;
        private bool _isEditMode;
        private readonly AlphaVantageService _alphaVantageService;
        private System.Windows.Threading.DispatcherTimer _searchTimer;
        private string _lastSearchText = "";
        
        /// <summary>
        /// Gets the custom benchmark that was created or edited
        /// </summary>
        public CustomBenchmark Benchmark => _benchmark;
        
        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public CustomBenchmarkDialog()
        {
            InitializeComponent();
            _benchmark = new CustomBenchmark();
            _isEditMode = false;
            
            // Attach resize behavior for borderless window
            this.SourceInitialized += (s, e) =>
            {
                WindowResizeBehavior.AttachResizeBehavior(this);
            };
            
            InitializeUI();
        }
        
        /// <summary>
        /// Constructor for creating a new custom benchmark
        /// </summary>
        public CustomBenchmarkDialog(AlphaVantageService alphaVantageService)
        {
            InitializeComponent();
            _benchmark = new CustomBenchmark();
            _isEditMode = false;
            _alphaVantageService = alphaVantageService;
            
            // Attach resize behavior for borderless window
            this.SourceInitialized += (s, e) =>
            {
                WindowResizeBehavior.AttachResizeBehavior(this);
            };
            
            InitializeUI();
            InitializeSymbolSearch();
        }
        
        /// <summary>
        /// Constructor for editing an existing custom benchmark
        /// </summary>
        /// <param name="benchmark">The benchmark to edit</param>
        public CustomBenchmarkDialog(CustomBenchmark benchmark, AlphaVantageService alphaVantageService)
        {
            InitializeComponent();
            _benchmark = benchmark ?? new CustomBenchmark();
            _isEditMode = benchmark != null;
            _alphaVantageService = alphaVantageService;
            
            // Attach resize behavior for borderless window
            this.SourceInitialized += (s, e) =>
            {
                WindowResizeBehavior.AttachResizeBehavior(this);
            };
            
            InitializeUI();
            InitializeSymbolSearch();
            LoadBenchmarkData();
        }
        
        /// <summary>
        /// Initialize symbol search functionality
        /// </summary>
        private void InitializeSymbolSearch()
        {
            if (_alphaVantageService == null)
                return;
                
            // Set up search timer for debouncing
            _searchTimer = new System.Windows.Threading.DispatcherTimer();
            _searchTimer.Interval = TimeSpan.FromMilliseconds(500);
            _searchTimer.Tick += SearchTimer_Tick;
            
            // Hook up text changed event for search
            SymbolComboBox.AddHandler(TextBox.TextChangedEvent, new TextChangedEventHandler(SymbolComboBox_TextChanged));
            
            // Hook up selection changed event
            SymbolComboBox.SelectionChanged += SymbolComboBox_SelectionChanged;
        }
        
        /// <summary>
        /// Handle text changes in symbol search box with debouncing
        /// </summary>
        private void SymbolComboBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Restart timer on each keystroke
            _searchTimer.Stop();
            _searchTimer.Start();
        }
        
        /// <summary>
        /// Execute search after debounce delay
        /// </summary>
        private async void SearchTimer_Tick(object sender, EventArgs e)
        {
            _searchTimer.Stop();
            
            var searchText = SymbolComboBox.Text?.Trim();
            
            // Only search if text has changed and is at least 1 character
            if (string.IsNullOrEmpty(searchText) || searchText == _lastSearchText || searchText.Length < 1)
                return;
                
            _lastSearchText = searchText;
            
            try
            {
                // Show loading indicator
                SymbolComboBox.IsEnabled = false;
                
                // Search for symbols
                var results = await _alphaVantageService.SearchSymbolsAsync(searchText);
                
                // Update ComboBox items
                SymbolComboBox.ItemsSource = results;
                
                // Open dropdown if results found
                if (results.Count > 0)
                {
                    SymbolComboBox.IsDropDownOpen = true;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error searching symbols: {ex.Message}", "Search Error", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
            }
            finally
            {
                SymbolComboBox.IsEnabled = true;
            }
        }
        
        /// <summary>
        /// Handle symbol selection from dropdown
        /// </summary>
        private void SymbolComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (SymbolComboBox.SelectedItem is SymbolSearchResult selected)
            {
                // Auto-fill the name field
                ComponentNameTextBox.Text = selected.Name;
            }
        }
        
        /// <summary>
        /// Initialize UI controls
        /// </summary>
        private void InitializeUI()
        {
            // Set window title based on mode
            this.Title = _isEditMode ? "Edit Custom Benchmark" : "Create Custom Benchmark";
            
            // Populate category combo box
            CategoryComboBox.ItemsSource = Enum.GetValues(typeof(BenchmarkCategory));
            CategoryComboBox.SelectedItem = _benchmark.Category;
            
            // Set initial components list
            UpdateComponentsUI();
            
            // Set default focus
            NameTextBox.Focus();
        }
        
        /// <summary>
        /// Load benchmark data into the UI
        /// </summary>
        private void LoadBenchmarkData()
        {
            // Set basic info
            NameTextBox.Text = _benchmark.Name;
            DescriptionTextBox.Text = _benchmark.Description;
            CategoryComboBox.SelectedItem = _benchmark.Category;
            
            // Update components list
            UpdateComponentsUI();
        }
        
        /// <summary>
        /// Update the components list in the UI
        /// </summary>
        private void UpdateComponentsUI()
        {
            ComponentsItemControl.ItemsSource = null;
            ComponentsItemControl.ItemsSource = _benchmark.Components;
            
            // Update total weight
            double totalWeight = _benchmark.Components.Sum(c => c.Weight);
            TotalWeightText.Text = $"Total: {totalWeight:P0}";
            
            // Update preview text
            PreviewText.Text = _benchmark.DisplaySymbol;
        }
        
        /// <summary>
        /// Event handler for Add Component button
        /// </summary>
        private void AddComponentButton_Click(object sender, RoutedEventArgs e)
        {
            string symbol = "";
            string name = ComponentNameTextBox.Text?.Trim();
            
            // Get symbol from ComboBox
            if (SymbolComboBox.SelectedItem is SymbolSearchResult selectedResult)
            {
                symbol = selectedResult.Symbol;
                if (string.IsNullOrEmpty(name))
                {
                    name = selectedResult.Name;
                }
            }
            else
            {
                symbol = SymbolComboBox.Text?.Trim().ToUpper();
            }
            
            string weightText = WeightTextBox.Text?.Trim();
            
            // Validate input
            if (string.IsNullOrEmpty(symbol))
            {
                MessageBox.Show("Please select or enter a symbol.", "Input Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                SymbolComboBox.Focus();
                return;
            }
            
            // Use symbol as name if name is empty
            if (string.IsNullOrEmpty(name))
            {
                name = symbol;
            }
            
            // Parse weight
            if (!double.TryParse(weightText.TrimEnd('%'), out double weight))
            {
                MessageBox.Show("Please enter a valid weight percentage.", "Input Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                WeightTextBox.Focus();
                return;
            }
            
            // Convert percentage to decimal (0-1)
            weight = weight / 100.0;
            
            // Check if component already exists
            if (_benchmark.Components.Any(c => c.Symbol == symbol))
            {
                // Update existing component
                var component = _benchmark.Components.First(c => c.Symbol == symbol);
                component.Name = name;
                component.Weight = weight;
            }
            else
            {
                // Add new component
                _benchmark.AddComponent(symbol, name, weight);
            }
            
            // Clear input fields
            SymbolComboBox.Text = "";
            SymbolComboBox.SelectedItem = null;
            ComponentNameTextBox.Text = "";
            WeightTextBox.Text = "100";
            
            // Update UI
            _benchmark.NormalizeWeights();
            UpdateComponentsUI();
            
            // Set focus to symbol field
            SymbolComboBox.Focus();
        }
        
        /// <summary>
        /// Event handler for Edit Component button
        /// </summary>
        private void EditComponentButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string symbol)
            {
                var component = _benchmark.Components.FirstOrDefault(c => c.Symbol == symbol);
                if (component != null)
                {
                    SymbolComboBox.Text = component.Symbol;
                    ComponentNameTextBox.Text = component.Name;
                    WeightTextBox.Text = (component.Weight * 100).ToString("F2");
                    
                    // Set focus to the weight field for easy editing
                    WeightTextBox.Focus();
                    WeightTextBox.SelectAll();
                }
            }
        }
        
        /// <summary>
        /// Event handler for Remove Component button
        /// </summary>
        private void RemoveComponentButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string symbol)
            {
                _benchmark.RemoveComponent(symbol);
                _benchmark.NormalizeWeights();
                UpdateComponentsUI();
            }
        }
        
        /// <summary>
        /// Event handler for Equal Weight button
        /// </summary>
        private void EqualWeightButton_Click(object sender, RoutedEventArgs e)
        {
            if (_benchmark.Components.Count == 0)
                return;
                
            // Set equal weights
            double equalWeight = 1.0 / _benchmark.Components.Count;
            foreach (var component in _benchmark.Components)
            {
                component.Weight = equalWeight;
            }
            
            // Update UI
            UpdateComponentsUI();
        }
        
        /// <summary>
        /// Event handler for Normalize Weights button
        /// </summary>
        private void NormalizeWeightsButton_Click(object sender, RoutedEventArgs e)
        {
            _benchmark.NormalizeWeights();
            UpdateComponentsUI();
        }
        
        /// <summary>
        /// Event handler for Save button
        /// </summary>
        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            // Validate input
            if (string.IsNullOrWhiteSpace(NameTextBox.Text))
            {
                MessageBox.Show("Please enter a benchmark name.", "Validation Error", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                NameTextBox.Focus();
                return;
            }
            
            if (_benchmark.Components.Count == 0)
            {
                MessageBox.Show("Please add at least one component to the benchmark.", 
                    "Validation Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                SymbolComboBox.Focus();
                return;
            }
            
            // Update benchmark from UI
            _benchmark.Name = NameTextBox.Text.Trim();
            _benchmark.Description = DescriptionTextBox.Text?.Trim();
            _benchmark.Category = (BenchmarkCategory)CategoryComboBox.SelectedItem;
            _benchmark.ModifiedDate = DateTime.Now;
            
            // Normalize weights
            _benchmark.NormalizeWeights();
            
            // Validate the benchmark
            if (!_benchmark.Validate(out string errorMessage))
            {
                MessageBox.Show(errorMessage, "Validation Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            
            DialogResult = true;
            Close();
        }
        
        /// <summary>
        /// Event handler for Cancel button
        /// </summary>
        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}