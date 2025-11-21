using System;
using System.Windows;
using Quantra.Models;
using Quantra.ViewModels;

namespace Quantra.Views
{
    /// <summary>
    /// Interaction logic for CustomizeLayoutDialog.xaml
    /// </summary>
    public partial class CustomizeLayoutDialog : Window
    {
        private CustomizeLayoutDialogViewModel _viewModel;

        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public CustomizeLayoutDialog()
        {
            InitializeComponent();
            _viewModel = new CustomizeLayoutDialogViewModel(null);
            DataContext = _viewModel;
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="currentLayout">Current layout configuration</param>
        public CustomizeLayoutDialog(LayoutConfig currentLayout = null)
        {
            InitializeComponent();
            
            _viewModel = new CustomizeLayoutDialogViewModel(currentLayout);
            DataContext = _viewModel;
            
            // Subscribe to ViewModel events
            _viewModel.CloseRequested += OnCloseRequested;
            _viewModel.LayoutApplied += OnLayoutApplied;
        }

        /// <summary>
        /// Gets the result layout configuration if dialog was accepted
        /// </summary>
        public LayoutConfig ResultLayout { get; private set; }

        /// <summary>
        /// Gets whether the dialog was accepted
        /// </summary>
        public bool DialogAccepted { get; private set; }

        private void OnCloseRequested(object sender, bool accepted)
        {
            DialogAccepted = accepted;
            if (accepted)
            {
                ResultLayout = _viewModel.CurrentLayout;
                DialogResult = true;
            }
            else
            {
                DialogResult = false;
            }
        }

        private void OnLayoutApplied(object sender, LayoutConfig layout)
        {
            ResultLayout = layout;
        }

        protected override void OnClosed(EventArgs e)
        {
            // Unsubscribe from events to prevent memory leaks
            if (_viewModel != null)
            {
                _viewModel.CloseRequested -= OnCloseRequested;
                _viewModel.LayoutApplied -= OnLayoutApplied;
            }
            base.OnClosed(e);
        }
    }
}