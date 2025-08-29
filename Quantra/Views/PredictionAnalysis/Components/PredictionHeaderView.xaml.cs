using System;
using System.Windows;
using System.Windows.Controls;
using System.ComponentModel;

namespace Quantra.Controls.Components
{
    public partial class PredictionHeaderView : UserControl, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        
        // Event to bubble up refresh button click
        public event EventHandler RefreshRequested;
        
        // Event to bubble up auto mode changes
        public event EventHandler<bool> AutoModeChanged;
        
        private bool _isAutomatedMode;
        public bool IsAutomatedMode
        {
            get { return _isAutomatedMode; }
            set 
            { 
                if (_isAutomatedMode != value)
                {
                    _isAutomatedMode = value;
                    OnPropertyChanged("IsAutomatedMode");
                    AutoModeChanged?.Invoke(this, value);
                }
            }
        }
        
        public PredictionHeaderView()
        {
            InitializeComponent();
            this.DataContext = this;
        }
        
        private void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            // Bubble up the event to parent
            RefreshRequested?.Invoke(this, EventArgs.Empty);
        }
        
        protected void OnPropertyChanged(string name)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
