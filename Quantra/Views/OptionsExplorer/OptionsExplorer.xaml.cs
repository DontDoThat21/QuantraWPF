using System.Windows;
using System.Windows.Controls;
using Microsoft.Extensions.DependencyInjection;
using Quantra.ViewModels;

namespace Quantra.Views.OptionsExplorer
{
    /// <summary>
    /// Interaction logic for OptionsExplorer.xaml
    /// Enhanced to support both legacy and new ViewModels
    /// </summary>
    public partial class OptionsExplorer : UserControl
    {
        public OptionsExplorer()
        {
            InitializeComponent();
            
            // Try to get the new enhanced ViewModel from DI container
            if (Application.Current is App && App.ServiceProvider != null)
            {
                try
                {
                    // Use new comprehensive OptionsViewModel if available
                    var newViewModel = App.ServiceProvider.GetService<OptionsViewModel>();
                    if (newViewModel != null)
                    {
                        DataContext = newViewModel;
                        return;
                    }
                }
                catch
                {
                    // Fall back to legacy ViewModel if new one not registered
                }
                
                try
                {
                    // Fall back to legacy OptionsExplorerViewModel
                    var legacyViewModel = App.ServiceProvider.GetService<OptionsExplorerViewModel>();
                    if (legacyViewModel != null)
                    {
                        DataContext = legacyViewModel;
                        return;
                    }
                }
                catch
                {
                    // If DI fails, DataContext will remain null
                    // XAML bindings will simply not work until ViewModel is set
                }
            }
        }

        /// <summary>
        /// Constructor for explicit ViewModel injection (legacy compatibility)
        /// </summary>
        public OptionsExplorer(OptionsExplorerViewModel viewModel) : this()
        {
            DataContext = viewModel;
        }

        /// <summary>
        /// Constructor for new ViewModel injection
        /// </summary>
        public OptionsExplorer(OptionsViewModel viewModel) : this()
        {
            DataContext = viewModel;
        }
    }
}
