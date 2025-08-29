// Handles UI helper methods for StockExplorer
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Quantra.Models;
using System.Threading.Tasks;
using System.Windows.Threading;
using Quantra.Views.Shared;

namespace Quantra.Controls
{
    public partial class StockExplorer
    {
        private string _lastSearchText = "";
        
        private async void SymbolComboBox_KeyUp(object sender, KeyEventArgs e)
        {
            if (sender is ComboBox comboBox)
            {
                // If Enter key is pressed and there's a selected/filtered symbol, and we're in Individual Asset mode, select it immediately
                if (e.Key == Key.Enter && !string.IsNullOrEmpty(comboBox.Text) && 
                    CurrentSelectionMode == Quantra.Enums.SymbolSelectionMode.IndividualAsset)
                {
                    var symbolToSelect = comboBox.Text.ToUpper();
                    if (_viewModel.FilteredSymbols.Contains(symbolToSelect))
                    {
                        try
                        {
                            // Use the modular symbol selection method
                            await HandleSymbolSelectionAsync(symbolToSelect, "KeyboardEntry");
                        }
                        catch (System.OperationCanceledException)
                        {
                            // Operation was cancelled - this is expected when user types quickly
                            DatabaseMonolith.Log("Info", $"Keyboard symbol selection for {symbolToSelect} was cancelled");
                        }
                        catch (Exception ex)
                        {
                            DatabaseMonolith.Log("Error", "Error selecting symbol via keyboard", ex.ToString());
                            CustomModal.ShowError($"Error selecting symbol: {ex.Message}", "Error", Window.GetWindow(this));
                        }
                    }
                }
                // Note: Validation is now handled through ViewModel SymbolSearchText property changes
            }
        }
    }
}
