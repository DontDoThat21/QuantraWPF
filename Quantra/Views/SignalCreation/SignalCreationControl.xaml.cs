using System.Windows.Controls;
using Quantra.DAL.Services.Interfaces;
using Quantra.ViewModels;

namespace Quantra.Views.SignalCreation
{
    /// <summary>
    /// Interaction logic for SignalCreationControl.xaml
    /// </summary>
    public partial class SignalCreationControl : UserControl
    {
        private readonly SignalCreationViewModel _viewModel;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        /// <param name="tradingSignalService">Trading signal service</param>
        public SignalCreationControl(ITradingSignalService tradingSignalService)
        {
            InitializeComponent();

            _viewModel = new SignalCreationViewModel(tradingSignalService);
            DataContext = _viewModel;
        }

        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public SignalCreationControl()
        {
            InitializeComponent();

            // Only run during design time
            if (System.ComponentModel.DesignerProperties.GetIsInDesignMode(this))
            {
                return;
            }

            // At runtime, we expect the DI constructor to be used
            // This constructor is here for XAML designer support only
        }
    }
}
