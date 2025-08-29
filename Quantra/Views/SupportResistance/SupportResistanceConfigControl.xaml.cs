using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using Quantra.Models;

namespace Quantra.Views.SupportResistance
{
    /// <summary>
    /// Interaction logic for SupportResistanceConfigControl.xaml
    /// </summary>
    public partial class SupportResistanceConfigControl : UserControl, INotifyPropertyChanged
    {
        private SupportResistanceStrategy _strategy;
        private ObservableCollection<PriceLevelAnalyzer.PriceLevel> _detectedLevels;
        
        // Event handlers
        public event EventHandler SettingsApplied;
        
        public SupportResistanceConfigControl()
        {
            InitializeComponent();
            DataContext = this;
            _detectedLevels = new ObservableCollection<PriceLevelAnalyzer.PriceLevel>();
        }
        
        /// <summary>
        /// Initialize the control with a strategy to configure
        /// </summary>
        public void Initialize(SupportResistanceStrategy strategy)
        {
            _strategy = strategy;
            
            // Bind strategy properties to the view
            LookbackPeriods = _strategy.LookbackPeriods;
            MinTouchesToConfirm = _strategy.MinTouchesToConfirm;
            LevelTolerance = _strategy.LevelTolerance;
            BreakoutConfirmation = _strategy.BreakoutConfirmation;
            UseVolumeConfirmation = _strategy.UseVolumeConfirmation;
            VolumeThreshold = _strategy.VolumeThreshold;
            UsePriceAction = _strategy.UsePriceAction;
            UsePivotPoints = _strategy.UsePivotPoints;
            UseFibonacciLevels = _strategy.UseFibonacciLevels;
            UseVolumeProfile = _strategy.UseVolumeProfile;
            
            // Initialize detected levels
            UpdateDetectedLevels();
        }
        
        /// <summary>
        /// Update the detected levels collection from the strategy
        /// </summary>
        public void UpdateDetectedLevels()
        {
            if (_strategy == null)
                return;
                
            var levels = _strategy.GetDetectedLevels();
            
            _detectedLevels.Clear();
            
            if (levels != null)
            {
                foreach (var level in levels)
                {
                    _detectedLevels.Add(level);
                }
            }
            
            OnPropertyChanged(nameof(DetectedLevels));
        }
        
        #region Properties
        
        public int LookbackPeriods { get; set; }
        public int MinTouchesToConfirm { get; set; }
        public double LevelTolerance { get; set; }
        public int BreakoutConfirmation { get; set; }
        public bool UseVolumeConfirmation { get; set; }
        public double VolumeThreshold { get; set; }
        public bool UsePriceAction { get; set; }
        public bool UsePivotPoints { get; set; }
        public bool UseFibonacciLevels { get; set; }
        public bool UseVolumeProfile { get; set; }
        
        public ObservableCollection<PriceLevelAnalyzer.PriceLevel> DetectedLevels
        {
            get { return _detectedLevels; }
        }
        
        #endregion
        
        #region Event Handlers
        
        private void ApplySettings_Click(object sender, RoutedEventArgs e)
        {
            if (_strategy == null)
                return;
                
            // Apply settings to the strategy
            _strategy.LookbackPeriods = LookbackPeriods;
            _strategy.MinTouchesToConfirm = MinTouchesToConfirm;
            _strategy.LevelTolerance = LevelTolerance;
            _strategy.BreakoutConfirmation = BreakoutConfirmation;
            _strategy.UseVolumeConfirmation = UseVolumeConfirmation;
            _strategy.VolumeThreshold = VolumeThreshold;
            _strategy.UsePriceAction = UsePriceAction;
            _strategy.UsePivotPoints = UsePivotPoints;
            _strategy.UseFibonacciLevels = UseFibonacciLevels;
            _strategy.UseVolumeProfile = UseVolumeProfile;
            
            // Notify that settings were applied
            SettingsApplied?.Invoke(this, EventArgs.Empty);
        }
        
        private void ResetDefaults_Click(object sender, RoutedEventArgs e)
        {
            // Reset to default values
            LookbackPeriods = 100;
            MinTouchesToConfirm = 2;
            LevelTolerance = 0.5;
            BreakoutConfirmation = 2;
            UseVolumeConfirmation = true;
            VolumeThreshold = 1.5;
            UsePriceAction = true;
            UsePivotPoints = true;
            UseFibonacciLevels = true;
            UseVolumeProfile = true;
            
            // Update bindings
            OnPropertyChanged(nameof(LookbackPeriods));
            OnPropertyChanged(nameof(MinTouchesToConfirm));
            OnPropertyChanged(nameof(LevelTolerance));
            OnPropertyChanged(nameof(BreakoutConfirmation));
            OnPropertyChanged(nameof(UseVolumeConfirmation));
            OnPropertyChanged(nameof(VolumeThreshold));
            OnPropertyChanged(nameof(UsePriceAction));
            OnPropertyChanged(nameof(UsePivotPoints));
            OnPropertyChanged(nameof(UseFibonacciLevels));
            OnPropertyChanged(nameof(UseVolumeProfile));
        }
        
        #endregion
        
        #region INotifyPropertyChanged
        
        public event PropertyChangedEventHandler PropertyChanged;
        
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        
        #endregion
    }
}