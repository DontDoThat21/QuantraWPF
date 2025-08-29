using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Data;  // Add this for CollectionViewSource
using Quantra.Models;
using Quantra.Services;
using Quantra.Services.Interfaces;

namespace Quantra.Views.Components
{
    /// <summary>
    /// Interaction logic for IndicatorBuilder.xaml
    /// </summary>
    public partial class IndicatorBuilder : UserControl, INotifyPropertyChanged
    {
        private readonly ITechnicalIndicatorService _indicatorService;
        private CustomIndicatorDefinition _currentDefinition;

        // Observable collections for UI binding
        public ObservableCollection<IIndicator> AvailableIndicators { get; private set; }
        public ObservableCollection<string> SelectedIndicators { get; private set; }

        // Current selected indicators and parameters
        private IIndicator _selectedIndicator;
        public IIndicator SelectedIndicator
        {
            get => _selectedIndicator;
            set
            {
                if (_selectedIndicator != value)
                {
                    _selectedIndicator = value;
                    OnPropertyChanged(nameof(SelectedIndicator));
                    LoadIndicatorParameters();
                }
            }
        }

        public IndicatorBuilder()
        {
            InitializeComponent();
            
            // Get the indicator service via service locator
            _indicatorService = ServiceLocator.GetService<ITechnicalIndicatorService>();
            
            // Initialize observable collections
            AvailableIndicators = new ObservableCollection<IIndicator>();
            SelectedIndicators = new ObservableCollection<string>();
            
            // Set data context
            this.DataContext = this;
            
            // Load available indicators
            LoadAvailableIndicatorsAsync();
        }

        private async void LoadAvailableIndicatorsAsync()
        {
            try
            {
                // Clear current list
                AvailableIndicators.Clear();
                
                // Load indicators from service
                var indicators = await _indicatorService.GetAllIndicatorsAsync();
                
                // Add to collection
                foreach (var indicator in indicators)
                {
                    AvailableIndicators.Add(indicator);
                }
                
                // Group and sort indicators
                CollectionViewSource.GetDefaultView(IndicatorsList.ItemsSource).GroupDescriptions.Add(
                    new PropertyGroupDescription("Category"));
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to load indicators: {ex.Message}", "Error", 
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void LoadIndicatorParameters()
        {
            // Clear current parameters
            ParametersPanel.Children.Clear();
            
            if (_selectedIndicator == null)
            {
                ParametersPanel.Children.Add(new TextBlock
                {
                    Text = "No component selected",
                    Foreground = new SolidColorBrush(Color.FromRgb(0x55, 0x55, 0x55)),
                    FontStyle = FontStyles.Italic
                });
                return;
            }
            
            // Add parameters for the selected indicator
            foreach (var param in _selectedIndicator.Parameters.Values)
            {
                // Create parameter control based on type
                var paramControl = CreateParameterControl(param);
                if (paramControl != null)
                {
                    ParametersPanel.Children.Add(paramControl);
                }
            }
        }

        private UIElement CreateParameterControl(IndicatorParameter parameter)
        {
            // Create a stack panel for the parameter
            var panel = new StackPanel
            {
                Margin = new Thickness(0, 5, 0, 5)
            };
            
            // Add label
            panel.Children.Add(new TextBlock
            {
                Text = parameter.Name,
                Foreground = new SolidColorBrush(Colors.White),
                Margin = new Thickness(0, 0, 0, 2)
            });
            
            // Add description if available
            if (!string.IsNullOrWhiteSpace(parameter.Description))
            {
                panel.Children.Add(new TextBlock
                {
                    Text = parameter.Description,
                    Foreground = new SolidColorBrush(Color.FromRgb(0xC0, 0xC0, 0xC0)),
                    FontSize = 11,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = new Thickness(0, 0, 0, 5)
                });
            }
            
            // Create editor control based on parameter type
            UIElement editor = null;
            
            if (parameter.ParameterType == typeof(double) || parameter.ParameterType == typeof(float))
            {
                var slider = new Slider
                {
                    Minimum = parameter.MinValue != null ? Convert.ToDouble(parameter.MinValue) : 0,
                    Maximum = parameter.MaxValue != null ? Convert.ToDouble(parameter.MaxValue) : 100,
                    Value = Convert.ToDouble(parameter.Value),
                    IsSnapToTickEnabled = true,
                    TickFrequency = 1,
                    Margin = new Thickness(0, 5, 0, 0)
                };
                
                var valueText = new TextBlock
                {
                    Text = slider.Value.ToString("F2"),
                    Foreground = new SolidColorBrush(Colors.White),
                    HorizontalAlignment = HorizontalAlignment.Right
                };
                
                slider.ValueChanged += (s, e) =>
                {
                    parameter.Value = slider.Value;
                    valueText.Text = slider.Value.ToString("F2");
                    UpdatePreview();
                };
                
                panel.Children.Add(slider);
                panel.Children.Add(valueText);
                editor = panel;
            }
            else if (parameter.ParameterType == typeof(int))
            {
                var slider = new Slider
                {
                    Minimum = parameter.MinValue != null ? Convert.ToDouble(parameter.MinValue) : 0,
                    Maximum = parameter.MaxValue != null ? Convert.ToDouble(parameter.MaxValue) : 100,
                    Value = Convert.ToDouble(parameter.Value),
                    IsSnapToTickEnabled = true,
                    TickFrequency = 1,
                    Margin = new Thickness(0, 5, 0, 0)
                };
                
                var valueText = new TextBlock
                {
                    Text = ((int)slider.Value).ToString(),
                    Foreground = new SolidColorBrush(Colors.White),
                    HorizontalAlignment = HorizontalAlignment.Right
                };
                
                slider.ValueChanged += (s, e) =>
                {
                    parameter.Value = (int)slider.Value;
                    valueText.Text = ((int)slider.Value).ToString();
                    UpdatePreview();
                };
                
                panel.Children.Add(slider);
                panel.Children.Add(valueText);
                editor = panel;
            }
            else if (parameter.ParameterType == typeof(bool))
            {
                var checkBox = new CheckBox
                {
                    Content = "",
                    IsChecked = (bool)parameter.Value,
                    Margin = new Thickness(0, 5, 0, 0)
                };
                
                checkBox.Checked += (s, e) =>
                {
                    parameter.Value = true;
                    UpdatePreview();
                };
                
                checkBox.Unchecked += (s, e) =>
                {
                    parameter.Value = false;
                    UpdatePreview();
                };
                
                panel.Children.Add(checkBox);
                editor = panel;
            }
            else if (parameter.ParameterType == typeof(string) && parameter.Options != null)
            {
                var comboBox = new ComboBox
                {
                    ItemsSource = parameter.Options,
                    SelectedItem = parameter.Value,
                    Margin = new Thickness(0, 5, 0, 0),
                    Background = new SolidColorBrush(Color.FromRgb(0x33, 0x33, 0x33)),
                    Foreground = new SolidColorBrush(Colors.White)
                };
                
                comboBox.SelectionChanged += (s, e) =>
                {
                    parameter.Value = comboBox.SelectedItem;
                    UpdatePreview();
                };
                
                panel.Children.Add(comboBox);
                editor = panel;
            }
            else
            {
                var textBox = new TextBox
                {
                    Text = parameter.Value?.ToString() ?? "",
                    Margin = new Thickness(0, 5, 0, 0),
                    Background = new SolidColorBrush(Color.FromRgb(0x33, 0x33, 0x33)),
                    Foreground = new SolidColorBrush(Colors.White)
                };
                
                textBox.TextChanged += (s, e) =>
                {
                    try
                    {
                        if (parameter.ParameterType == typeof(string))
                        {
                            parameter.Value = textBox.Text;
                        }
                        else if (parameter.ParameterType == typeof(int))
                        {
                            parameter.Value = int.Parse(textBox.Text);
                        }
                        else if (parameter.ParameterType == typeof(double))
                        {
                            parameter.Value = double.Parse(textBox.Text);
                        }
                        else if (parameter.ParameterType == typeof(bool))
                        {
                            parameter.Value = bool.Parse(textBox.Text);
                        }
                        UpdatePreview();
                    }
                    catch
                    {
                        // Invalid value, ignore
                    }
                };
                
                panel.Children.Add(textBox);
                editor = panel;
            }
            
            return editor;
        }

        private void UpdatePreview()
        {
            // TODO: Update the preview chart with the current indicator values
        }

        #region INotifyPropertyChanged
        
        public event PropertyChangedEventHandler PropertyChanged;
        
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        
        #endregion
    }
}