using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using Quantra.Models;

namespace Quantra.Views.Backtesting
{
    /// <summary>
    /// Interaction logic for CustomBenchmarkDialog.xaml
    /// </summary>
    public partial class CustomBenchmarkDialog : Window
    {
        private CustomBenchmark _benchmark;
        private bool _isEditMode;
        
        /// <summary>
        /// Gets the custom benchmark that was created or edited
        /// </summary>
        public CustomBenchmark Benchmark => _benchmark;
        
        /// <summary>
        /// Constructor for creating a new custom benchmark
        /// </summary>
        public CustomBenchmarkDialog()
        {
            InitializeComponent();
            _benchmark = new CustomBenchmark();
            _isEditMode = false;
            
            InitializeUI();
        }
        
        /// <summary>
        /// Constructor for editing an existing custom benchmark
        /// </summary>
        /// <param name="benchmark">The benchmark to edit</param>
        public CustomBenchmarkDialog(CustomBenchmark benchmark)
        {
            InitializeComponent();
            _benchmark = benchmark ?? new CustomBenchmark();
            _isEditMode = benchmark != null;
            
            InitializeUI();
            LoadBenchmarkData();
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
            string symbol = SymbolTextBox.Text?.Trim().ToUpper();
            string name = ComponentNameTextBox.Text?.Trim();
            string weightText = WeightTextBox.Text?.Trim();
            
            // Validate input
            if (string.IsNullOrEmpty(symbol))
            {
                MessageBox.Show("Please enter a symbol.", "Input Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                SymbolTextBox.Focus();
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
            SymbolTextBox.Text = "";
            ComponentNameTextBox.Text = "";
            WeightTextBox.Text = "100";
            
            // Update UI
            _benchmark.NormalizeWeights();
            UpdateComponentsUI();
            
            // Set focus to symbol field
            SymbolTextBox.Focus();
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
                    SymbolTextBox.Text = component.Symbol;
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
                SymbolTextBox.Focus();
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