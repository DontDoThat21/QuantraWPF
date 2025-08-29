using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using Quantra.Models;
using Quantra.Services;

namespace Quantra.Views.Backtesting
{
    /// <summary>
    /// Interaction logic for CustomBenchmarkManager.xaml
    /// </summary>
    public partial class CustomBenchmarkManager : Window
    {
        private readonly CustomBenchmarkService _benchmarkService;
        private List<CustomBenchmark> _benchmarks;
        private CustomBenchmark _selectedBenchmark;
        
        /// <summary>
        /// Gets the selected benchmark (if user clicked "Use" button)
        /// </summary>
        public CustomBenchmark SelectedBenchmark => _selectedBenchmark;
        
        /// <summary>
        /// Constructor
        /// </summary>
        public CustomBenchmarkManager()
        {
            InitializeComponent();
            _benchmarkService = new CustomBenchmarkService();
            
            // Load benchmarks
            LoadBenchmarks();
        }
        
        /// <summary>
        /// Load all custom benchmarks
        /// </summary>
        private void LoadBenchmarks()
        {
            _benchmarks = _benchmarkService.GetCustomBenchmarks();
            BenchmarksGrid.ItemsSource = null;
            BenchmarksGrid.ItemsSource = _benchmarks;
            
            // Clear selection
            ClearDetails();
        }
        
        /// <summary>
        /// Clear details panel
        /// </summary>
        private void ClearDetails()
        {
            BenchmarkNameText.Text = "";
            BenchmarkDescriptionText.Text = "";
            ComponentsListView.ItemsSource = null;
            UseBenchmarkButton.IsEnabled = false;
            _selectedBenchmark = null;
        }
        
        /// <summary>
        /// Update details panel with selected benchmark
        /// </summary>
        private void UpdateDetails(CustomBenchmark benchmark)
        {
            if (benchmark == null)
            {
                ClearDetails();
                return;
            }
            
            _selectedBenchmark = benchmark;
            
            BenchmarkNameText.Text = $"{benchmark.Name} ({benchmark.DisplaySymbol})";
            BenchmarkDescriptionText.Text = benchmark.Description;
            ComponentsListView.ItemsSource = benchmark.Components;
            UseBenchmarkButton.IsEnabled = true;
        }
        
        /// <summary>
        /// Event handler for Create New button
        /// </summary>
        private void CreateNewButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new CustomBenchmarkDialog();
            bool? result = dialog.ShowDialog();
            
            if (result == true && dialog.Benchmark != null)
            {
                // Save the new benchmark
                _benchmarkService.SaveCustomBenchmark(dialog.Benchmark);
                
                // Reload benchmarks
                LoadBenchmarks();
            }
        }
        
        /// <summary>
        /// Event handler for Edit button
        /// </summary>
        private void EditButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string id)
            {
                var benchmark = _benchmarks.FirstOrDefault(b => b.Id == id);
                if (benchmark != null)
                {
                    var dialog = new CustomBenchmarkDialog(benchmark);
                    bool? result = dialog.ShowDialog();
                    
                    if (result == true && dialog.Benchmark != null)
                    {
                        // Save the edited benchmark
                        _benchmarkService.SaveCustomBenchmark(dialog.Benchmark);
                        
                        // Reload benchmarks
                        LoadBenchmarks();
                    }
                }
            }
        }
        
        /// <summary>
        /// Event handler for Clone button
        /// </summary>
        private void CloneButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string id)
            {
                var benchmark = _benchmarks.FirstOrDefault(b => b.Id == id);
                if (benchmark != null)
                {
                    // Create a new benchmark with same properties but new ID
                    var clone = new CustomBenchmark
                    {
                        Name = $"{benchmark.Name} (Copy)",
                        Description = benchmark.Description,
                        Category = benchmark.Category
                    };
                    
                    // Copy components
                    foreach (var component in benchmark.Components)
                    {
                        clone.AddComponent(component.Symbol, component.Name, component.Weight);
                    }
                    
                    // Show dialog to allow editing the clone
                    var dialog = new CustomBenchmarkDialog(clone);
                    bool? result = dialog.ShowDialog();
                    
                    if (result == true && dialog.Benchmark != null)
                    {
                        // Save the cloned benchmark
                        _benchmarkService.SaveCustomBenchmark(dialog.Benchmark);
                        
                        // Reload benchmarks
                        LoadBenchmarks();
                    }
                }
            }
        }
        
        /// <summary>
        /// Event handler for Delete button
        /// </summary>
        private void DeleteButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string id)
            {
                var benchmark = _benchmarks.FirstOrDefault(b => b.Id == id);
                if (benchmark != null)
                {
                    // Confirm deletion
                    var result = MessageBox.Show(
                        $"Are you sure you want to delete the benchmark '{benchmark.Name}'?",
                        "Confirm Delete", MessageBoxButton.YesNo, MessageBoxImage.Question);
                        
                    if (result == MessageBoxResult.Yes)
                    {
                        // Delete the benchmark
                        _benchmarkService.DeleteCustomBenchmark(id);
                        
                        // Reload benchmarks
                        LoadBenchmarks();
                    }
                }
            }
        }
        
        /// <summary>
        /// Event handler for selection change in the benchmarks grid
        /// </summary>
        private void BenchmarksGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var benchmark = BenchmarksGrid.SelectedItem as CustomBenchmark;
            UpdateDetails(benchmark);
        }
        
        /// <summary>
        /// Event handler for Use Benchmark button
        /// </summary>
        private void UseBenchmarkButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedBenchmark != null)
            {
                // Set this as the active custom benchmark in user settings
                UserSettingsService.SetActiveCustomBenchmark(_selectedBenchmark.Id);
                
                DialogResult = true;
                Close();
            }
        }
    }
}