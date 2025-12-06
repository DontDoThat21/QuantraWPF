using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using Quantra.DAL.Models;
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using Microsoft.Win32;

namespace Quantra.Views.PredictionAnalysis
{
    public partial class TrainingConfigurationWindow : Window
    {
        private readonly LoggingService _loggingService;
        private readonly TrainingConfigurationService _configService;
        private TrainingConfiguration _currentConfiguration;
        private List<TrainingConfiguration> _allConfigurations;

        public TrainingConfiguration Configuration { get; private set; }

        public TrainingConfigurationWindow(TrainingConfiguration initialConfig, LoggingService loggingService)
        {
            InitializeComponent();

            _loggingService = loggingService;
            _configService = new TrainingConfigurationService(_loggingService);
            _currentConfiguration = initialConfig?.Clone() ?? TrainingConfiguration.CreateDefault();

            LoadAllConfigurations();
            PopulateUIFromConfiguration(_currentConfiguration);

            // Set up event handlers for model type changes
            ModelTypeBox.SelectionChanged += ModelTypeBox_SelectionChanged;
            ArchitectureBox.SelectionChanged += ArchitectureBox_SelectionChanged;
        }

        /// <summary>
        /// Load all available configurations into the preset dropdown
        /// </summary>
        private void LoadAllConfigurations()
        {
            try
            {
                _allConfigurations = _configService.GetAllConfigurations();

                PresetComboBox.ItemsSource = _allConfigurations;

                // Select the current configuration if it exists in the list
                var matchingConfig = _allConfigurations.FirstOrDefault(c =>
                    c.ConfigurationName == _currentConfiguration.ConfigurationName);

                if (matchingConfig != null)
                {
                    PresetComboBox.SelectedItem = matchingConfig;
                }
                else if (_allConfigurations.Any())
                {
                    PresetComboBox.SelectedIndex = 0;
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to load configurations");
                MessageBox.Show($"Error loading configurations: {ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// Populate UI controls from configuration object
        /// </summary>
        private void PopulateUIFromConfiguration(TrainingConfiguration config)
        {
            try
            {
                // Model Selection
                SelectComboBoxByTag(ModelTypeBox, config.ModelType);
                SelectComboBoxByTag(ArchitectureBox, config.ArchitectureType);
                SelectComboBoxByTag(FeatureTypeBox, config.FeatureType);
                LookbackPeriodBox.Text = config.LookbackPeriod.ToString();

                // Neural Network Hyperparameters
                EpochsBox.Text = config.Epochs.ToString();
                BatchSizeBox.Text = config.BatchSize.ToString();
                LearningRateBox.Text = config.LearningRate.ToString();
                DropoutBox.Text = config.Dropout.ToString();
                HiddenDimBox.Text = config.HiddenDim.ToString();
                NumLayersBox.Text = config.NumLayers.ToString();
                SelectComboBoxByTag(OptimizerBox, config.Optimizer);
                WeightDecayBox.Text = config.WeightDecay.ToString();

                // TFT-Specific
                NumHeadsBox.Text = config.NumHeads.ToString();
                NumAttentionLayersBox.Text = config.NumAttentionLayers.ToString();

                // Random Forest
                NumberOfTreesBox.Text = config.NumberOfTrees.ToString();
                MaxDepthBox.Text = config.MaxDepth.ToString();
                MinSamplesSplitBox.Text = config.MinSamplesSplit.ToString();

                // Training Optimization
                UseEarlyStoppingBox.IsChecked = config.UseEarlyStopping;
                UseLRSchedulerBox.IsChecked = config.UseLearningRateScheduler;
                EarlyStoppingPatienceBox.Text = config.EarlyStoppingPatience.ToString();
                LRSchedulerPatienceBox.Text = config.LRSchedulerPatience.ToString();
                LRSchedulerFactorBox.Text = config.LRSchedulerFactor.ToString();
                TrainTestSplitBox.Text = config.TrainTestSplit.ToString();

                // Advanced Options
                UseGPUBox.IsChecked = config.UseGPU;
                VerboseLoggingBox.IsChecked = config.VerboseLogging;
                SaveCheckpointsBox.IsChecked = config.SaveCheckpoints;
                UseFeatureEngineeringBox.IsChecked = config.UseFeatureEngineering;

                // Update parameter visibility
                UpdateParameterVisibility();
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to populate UI from configuration");
            }
        }

        /// <summary>
        /// Collect UI values into configuration object
        /// </summary>
        private TrainingConfiguration CollectConfigurationFromUI()
        {
            var config = new TrainingConfiguration();

            try
            {
                // Model Selection
                config.ModelType = GetComboBoxTag(ModelTypeBox);
                config.ArchitectureType = GetComboBoxTag(ArchitectureBox);
                config.FeatureType = GetComboBoxTag(FeatureTypeBox);
                config.LookbackPeriod = ParseInt(LookbackPeriodBox.Text, 60);

                // Neural Network Hyperparameters
                config.Epochs = ParseInt(EpochsBox.Text, 50);
                config.BatchSize = ParseInt(BatchSizeBox.Text, 32);
                config.LearningRate = ParseDouble(LearningRateBox.Text, 0.001);
                config.Dropout = ParseDouble(DropoutBox.Text, 0.1);
                config.HiddenDim = ParseInt(HiddenDimBox.Text, 128);
                config.NumLayers = ParseInt(NumLayersBox.Text, 2);
                config.Optimizer = GetComboBoxTag(OptimizerBox);
                config.WeightDecay = ParseDouble(WeightDecayBox.Text, 0.0001);

                // TFT-Specific
                config.NumHeads = ParseInt(NumHeadsBox.Text, 4);
                config.NumAttentionLayers = ParseInt(NumAttentionLayersBox.Text, 2);

                // Random Forest
                config.NumberOfTrees = ParseInt(NumberOfTreesBox.Text, 100);
                config.MaxDepth = ParseInt(MaxDepthBox.Text, 10);
                config.MinSamplesSplit = ParseInt(MinSamplesSplitBox.Text, 2);

                // Training Optimization
                config.UseEarlyStopping = UseEarlyStoppingBox.IsChecked ?? true;
                config.UseLearningRateScheduler = UseLRSchedulerBox.IsChecked ?? true;
                config.EarlyStoppingPatience = ParseInt(EarlyStoppingPatienceBox.Text, 10);
                config.LRSchedulerPatience = ParseInt(LRSchedulerPatienceBox.Text, 5);
                config.LRSchedulerFactor = ParseDouble(LRSchedulerFactorBox.Text, 0.5);
                config.TrainTestSplit = ParseDouble(TrainTestSplitBox.Text, 0.8);

                // Advanced Options
                config.UseGPU = UseGPUBox.IsChecked ?? true;
                config.VerboseLogging = VerboseLoggingBox.IsChecked ?? true;
                config.SaveCheckpoints = SaveCheckpointsBox.IsChecked ?? true;
                config.UseFeatureEngineering = UseFeatureEngineeringBox.IsChecked ?? true;

                // Preserve name and description from current config
                config.ConfigurationName = _currentConfiguration.ConfigurationName;
                config.Description = _currentConfiguration.Description;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to collect configuration from UI");
                throw;
            }

            return config;
        }

        /// <summary>
        /// Validate current configuration
        /// </summary>
        private bool ValidateConfiguration(out List<string> errors)
        {
            errors = new List<string>();

            try
            {
                var config = CollectConfigurationFromUI();
                errors = config.Validate();

                if (errors.Any())
                {
                    ValidationText.Text = string.Join("; ", errors);
                    return false;
                }

                ValidationText.Text = "";
                return true;
            }
            catch (Exception ex)
            {
                errors.Add($"Error validating configuration: {ex.Message}");
                ValidationText.Text = errors[0];
                return false;
            }
        }

        /// <summary>
        /// Update parameter section visibility based on model type
        /// </summary>
        private void UpdateParameterVisibility()
        {
            var modelType = GetComboBoxTag(ModelTypeBox);
            var architecture = GetComboBoxTag(ArchitectureBox);

            // Show TFT parameters if TFT architecture is selected
            TFTParametersPanel.Visibility = (architecture == "tft")
                ? Visibility.Visible
                : Visibility.Collapsed;

            // Show Random Forest parameters if RF model is selected
            RandomForestParametersPanel.Visibility = (modelType == "random_forest")
                ? Visibility.Visible
                : Visibility.Collapsed;
        }

        #region Event Handlers

        private void PresetComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (PresetComboBox.SelectedItem is TrainingConfiguration selectedConfig)
            {
                _currentConfiguration = selectedConfig.Clone();
                PopulateUIFromConfiguration(_currentConfiguration);
            }
        }

        private void ModelTypeBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            UpdateParameterVisibility();
        }

        private void ArchitectureBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            UpdateParameterVisibility();
        }

        private void SaveAsNewButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Validate first
                if (!ValidateConfiguration(out var errors))
                {
                    MessageBox.Show($"Cannot save invalid configuration:\n\n{string.Join("\n", errors)}",
                        "Validation Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                // Prompt for name
                var dialog = new InputDialog("Save Configuration", "Enter configuration name:");
                if (dialog.ShowDialog() == true && !string.IsNullOrWhiteSpace(dialog.InputText))
                {
                    var config = CollectConfigurationFromUI();
                    config.ConfigurationName = dialog.InputText;
                    config.Description = $"Custom configuration created {DateTime.Now:yyyy-MM-dd HH:mm}";
                    config.CreatedDate = DateTime.Now;
                    config.LastModifiedDate = DateTime.Now;

                    if (_configService.SaveConfiguration(config))
                    {
                        _currentConfiguration = config;
                        LoadAllConfigurations();
                        MessageBox.Show($"Configuration '{config.ConfigurationName}' saved successfully!",
                            "Success", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                    else
                    {
                        MessageBox.Show("Failed to save configuration. Check logs for details.",
                            "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to save configuration");
                MessageBox.Show($"Error saving configuration: {ex.Message}",
                    "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void DeleteButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (PresetComboBox.SelectedItem is TrainingConfiguration selectedConfig)
                {
                    var result = MessageBox.Show(
                        $"Are you sure you want to delete configuration '{selectedConfig.ConfigurationName}'?\n\n" +
                        "Note: Built-in configurations (Default, Fast Training, High Accuracy, TFT Optimized) cannot be deleted.",
                        "Confirm Delete",
                        MessageBoxButton.YesNo,
                        MessageBoxImage.Question);

                    if (result == MessageBoxResult.Yes)
                    {
                        if (_configService.DeleteConfiguration(selectedConfig.ConfigurationName))
                        {
                            LoadAllConfigurations();
                            MessageBox.Show("Configuration deleted successfully!",
                                "Success", MessageBoxButton.OK, MessageBoxImage.Information);
                        }
                        else
                        {
                            MessageBox.Show("Cannot delete built-in configuration.",
                                "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to delete configuration");
                MessageBox.Show($"Error deleting configuration: {ex.Message}",
                    "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            var result = MessageBox.Show(
                "Reset to Default configuration?",
                "Confirm Reset",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question);

            if (result == MessageBoxResult.Yes)
            {
                _currentConfiguration = TrainingConfiguration.CreateDefault();
                PopulateUIFromConfiguration(_currentConfiguration);
            }
        }

        private void OKButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (!ValidateConfiguration(out var errors))
                {
                    MessageBox.Show($"Please fix the following errors:\n\n{string.Join("\n", errors)}",
                        "Validation Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                Configuration = CollectConfigurationFromUI();
                DialogResult = true;
                Close();
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error applying configuration");
                MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }

        #endregion

        #region Helper Methods

        private void SelectComboBoxByTag(ComboBox comboBox, string tag)
        {
            if (comboBox == null || string.IsNullOrEmpty(tag)) return;

            foreach (ComboBoxItem item in comboBox.Items)
            {
                if (item.Tag?.ToString() == tag)
                {
                    comboBox.SelectedItem = item;
                    return;
                }
            }

            // If not found, select first item
            if (comboBox.Items.Count > 0)
            {
                comboBox.SelectedIndex = 0;
            }
        }

        private string GetComboBoxTag(ComboBox comboBox)
        {
            if (comboBox?.SelectedItem is ComboBoxItem item)
            {
                return item.Tag?.ToString() ?? "";
            }
            return "";
        }

        private int ParseInt(string text, int defaultValue)
        {
            return int.TryParse(text, out int value) ? value : defaultValue;
        }

        private double ParseDouble(string text, double defaultValue)
        {
            return double.TryParse(text, out double value) ? value : defaultValue;
        }

        #endregion
    }

    /// <summary>
    /// Simple input dialog for getting configuration name
    /// </summary>
    public class InputDialog : Window
    {
        private TextBox _inputTextBox;
        public string InputText { get; private set; }

        public InputDialog(string title, string prompt)
        {
            Title = title;
            Width = 400;
            Height = 150;
            WindowStartupLocation = WindowStartupLocation.CenterOwner;
            Background = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(30, 30, 48));

            var grid = new Grid { Margin = new Thickness(15) };
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

            var promptText = new TextBlock
            {
                Text = prompt,
                Foreground = System.Windows.Media.Brushes.White,
                Margin = new Thickness(0, 0, 0, 10)
            };
            Grid.SetRow(promptText, 0);

            _inputTextBox = new TextBox
            {
                Margin = new Thickness(0, 0, 0, 15),
                Padding = new Thickness(5)
            };
            Grid.SetRow(_inputTextBox, 1);

            var buttonPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right
            };
            Grid.SetRow(buttonPanel, 2);

            var okButton = new Button
            {
                Content = "OK",
                Width = 80,
                Height = 30,
                Margin = new Thickness(0, 0, 10, 0)
            };
            okButton.Click += (s, e) =>
            {
                InputText = _inputTextBox.Text;
                DialogResult = true;
                Close();
            };

            var cancelButton = new Button
            {
                Content = "Cancel",
                Width = 80,
                Height = 30
            };
            cancelButton.Click += (s, e) =>
            {
                DialogResult = false;
                Close();
            };

            buttonPanel.Children.Add(okButton);
            buttonPanel.Children.Add(cancelButton);

            grid.Children.Add(promptText);
            grid.Children.Add(_inputTextBox);
            grid.Children.Add(buttonPanel);

            Content = grid;

            _inputTextBox.Focus();
        }
    }
}
