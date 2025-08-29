using System;
using System.Windows.Controls;

namespace Quantra.Controls.Components
{
    public partial class StatusBarView : UserControl
    {
        public StatusBarView()
        {
            InitializeComponent();
        }
        
        // Method to update the status text
        public void UpdateStatus(string status)
        {
            StatusText.Text = status;
        }
        
        // Method to update the last updated text
        public void UpdateLastUpdated(DateTime lastUpdated)
        {
            LastUpdatedText.Text = $"Last updated: {lastUpdated:MM/dd/yyyy HH:mm}";
        }
        
        // Method for initial state
        public void SetReady()
        {
            StatusText.Text = "Ready";
            LastUpdatedText.Text = "Last updated: Never";
        }
        
        // Method for analyzing state
        public void SetAnalyzing()
        {
            StatusText.Text = "Analyzing data...";
        }
        
        // Method for error state
        public void SetError(string message = "Error during analysis")
        {
            StatusText.Text = message;
        }
    }
}
