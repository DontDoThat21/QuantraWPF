// Handles UI event handlers for StockExplorer
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using Quantra.Models;
using System.Threading.Tasks;
using System.Windows.Threading;
using Quantra.Views.Shared;
using Quantra.Enums;

namespace Quantra.Controls
{
    public partial class StockExplorer
    {
        private string _lastSearchText = "";

        private void StockExplorer_Loaded(object sender, RoutedEventArgs e)
        {
            // Load DataGrid settings after the control is fully loaded
            LoadDataGridSettings();
        }

        // Selection mode changed event handler
        private void SelectionModeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (sender is not ComboBox comboBox || comboBox.SelectedIndex < 0)
                return;

            var newMode = (SymbolSelectionMode)comboBox.SelectedIndex;
            CurrentSelectionMode = newMode;
        }

        // Called when selection mode changes to update UI accordingly
        private async void OnSelectionModeChanged()
        {
            try
            {
                // Reset symbol and price fields to their defaults when mode changes
                SymbolText = "";
                PriceText = "";
                UpdatedTimestampText = "";
                
                switch (CurrentSelectionMode)
                {
                    case SymbolSelectionMode.IndividualAsset:
                        // Show the symbol search combo box
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Visible;
                        if (ModeStatusPanel != null)
                            ModeStatusPanel.Visibility = Visibility.Collapsed;
                        // Hide all RSI buttons and Top P/E button
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        SetAllDatabaseButtonVisibility(Visibility.Collapsed);
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Disable search button initially - will be enabled when valid symbol is typed
                        DisableStockSearchButton();
                        break;

                    case SymbolSelectionMode.TopVolumeRsiDiscrepancies:
                        // Hide the individual symbol search for Top Volume RSI mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Top Volume RSI button, hide other buttons
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Top Volume RSI' to search for high volume RSI discrepancies";
                        }
                        break;

                    case SymbolSelectionMode.TopPE:
                        // Hide the individual symbol search for Top P/E mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Top P/E button, hide other buttons
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Top P/E' to search for stocks with highest P/E ratios";
                        }
                        break;

                    case SymbolSelectionMode.HighVolume:
                        // Hide the individual symbol search for High Volume mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show High Volume button, hide other buttons
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load High Volume' to search for high volume stocks";
                        }
                        break;

                    case SymbolSelectionMode.LowPE:
                        // Hide the individual symbol search for Low P/E mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Low P/E button, hide other buttons
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Low P/E' to search for stocks with lowest P/E ratios";
                        }
                        break;

                    case SymbolSelectionMode.RsiOversold:
                        // Hide the individual symbol search for RSI Oversold mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show RSI Oversold button, hide other buttons
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Visible;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load RSI Oversold' to search for oversold stocks";
                        }
                        break;

                    case SymbolSelectionMode.RsiOverbought:
                        // Hide the individual symbol search for RSI Overbought mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show RSI Overbought button, hide other buttons
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load RSI Overbought' to search for overbought stocks";
                        }
                        break;

                    case SymbolSelectionMode.AllDatabase:
                        // Hide the individual symbol search for All Database mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show All Database button, hide other buttons
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load All Database' to display all cached stock data";
                        }
                        break;

                    case SymbolSelectionMode.HighTheta:
                        // Hide the individual symbol search for High Theta mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show High Theta button, hide other buttons
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load High Theta' to find stocks with high time decay opportunities";
                        }
                        break;

                    case SymbolSelectionMode.HighBeta:
                        // Hide the individual symbol search for High Beta mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show High Beta button, hide other buttons
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load High Beta' to find stocks with high market correlation";
                        }
                        break;

                    case SymbolSelectionMode.HighAlpha:
                        // Hide the individual symbol search for High Alpha mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show High Alpha button, hide other buttons
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Visible;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load High Alpha' to find stocks generating excess returns";
                        }
                        break;

                    case SymbolSelectionMode.BullishCupAndHandle:
                        // Hide the individual symbol search for Bullish Cup and Handle mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Bullish Cup and Handle button, hide other buttons
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Visible;
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Cup and Handle' to find stocks with bullish cup and handle patterns";
                        }
                        break;

                    case SymbolSelectionMode.BearishCupAndHandle:
                        // Hide the individual symbol search for Bearish Cup and Handle mode
                        if (SymbolSearchPanel != null)
                            SymbolSearchPanel.Visibility = Visibility.Collapsed;
                        
                        // Show Bearish Cup and Handle button, hide other buttons
                        if (BearishCupAndHandleButton != null)
                            BearishCupAndHandleButton.Visibility = Visibility.Visible;
                        if (BullishCupAndHandleButton != null)
                            BullishCupAndHandleButton.Visibility = Visibility.Collapsed;
                        if (RsiOversoldButton != null)
                            RsiOversoldButton.Visibility = Visibility.Collapsed;
                        if (RsiOverboughtButton != null)
                            RsiOverboughtButton.Visibility = Visibility.Collapsed;
                        if (TopVolumeRsiButton != null)
                            TopVolumeRsiButton.Visibility = Visibility.Collapsed;
                        if (TopPEButton != null)
                            TopPEButton.Visibility = Visibility.Collapsed;
                        if (HighVolumeButton != null)
                            HighVolumeButton.Visibility = Visibility.Collapsed;
                        if (LowPEButton != null)
                            LowPEButton.Visibility = Visibility.Collapsed;
                        if (AllDatabaseButton != null)
                            AllDatabaseButton.Visibility = Visibility.Collapsed;
                        if (HighThetaButton != null)
                            HighThetaButton.Visibility = Visibility.Collapsed;
                        if (HighBetaButton != null)
                            HighBetaButton.Visibility = Visibility.Collapsed;
                        if (HighAlphaButton != null)
                            HighAlphaButton.Visibility = Visibility.Collapsed;
                        
                        // Show status but don't auto-load
                        if (ModeStatusPanel != null)
                        {
                            ModeStatusPanel.Visibility = Visibility.Visible;
                            if (ModeStatusText != null)
                                ModeStatusText.Text = "Click 'Load Bearish Cup and Handle' to find stocks with bearish cup and handle patterns";
                        }
                        break;
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error changing selection mode", ex.ToString());
                CustomModal.ShowError($"Error changing selection mode: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = $"Error loading {CurrentSelectionMode} data";
            }
        }

        // SymbolComboBox selection changed event handler
        // This handler ONLY enables the Search button when a valid symbol is selected
        // It does NOT trigger an automatic search - search is only triggered by:
        // 1. Enter key press (handled in SymbolComboBox_KeyUp)
        // 2. Clicking on a dropdown item (handled in SymbolComboBox_DropDownClosed)
        // 3. Clicking the Search button (handled in RefreshButton_Click)
        private void SymbolComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isHandlingSelectionChanged || sender is not ComboBox comboBox) 
                return;

            var selectedSymbol = comboBox.SelectedItem as string;
            if (!string.IsNullOrEmpty(selectedSymbol))
            {
                // Only enable the Search button - do NOT auto-search
                // Search is triggered by Enter key, dropdown click, or Search button click
                EnableStockSearchButton();
            }
        }

        // DropDownClosed event handler - triggers search when user explicitly clicks on a dropdown item
        private async void SymbolComboBox_DropDownClosed(object sender, EventArgs e)
        {
            if (sender is not ComboBox comboBox || CurrentSelectionMode != Quantra.Enums.SymbolSelectionMode.IndividualAsset)
                return;

            var selectedSymbol = comboBox.SelectedItem as string;
            if (!string.IsNullOrEmpty(selectedSymbol))
            {
                try
                {
                    // User explicitly clicked on a dropdown item, trigger the search
                    await HandleSymbolSelectionAsync(selectedSymbol, "DropdownSelection");
                }
                catch (System.OperationCanceledException)
                {
                    // Operation was cancelled - this is expected when user selects quickly
                }
                catch (Exception ex)
                {
                    CustomModal.ShowError($"Error selecting symbol: {ex.Message}", "Error", Window.GetWindow(this));
                }
            }
        }

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
                            //DatabaseMonolith.Log("Info", $"Keyboard symbol selection for {symbolToSelect} was cancelled");
                        }
                        catch (Exception ex)
                        {
                            //DatabaseMonolith.Log("Error", "Error selecting symbol via keyboard", ex.ToString());
                            CustomModal.ShowError($"Error selecting symbol: {ex.Message}", "Error", Window.GetWindow(this));
                        }
                    }
                }
                // Note: Validation is now handled through ViewModel SymbolSearchText property changes
            }
        }

        // RSI Oversold button click event handler
        private async void RsiOversoldButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Disable button to prevent multiple clicks
                if (RsiOversoldButton != null)
                    RsiOversoldButton.IsEnabled = false;

                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading RSI Oversold data...";
                }

                // Load RSI oversold stocks
                await LoadSymbolsForMode(SymbolSelectionMode.RsiOversold);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for RSI Oversold";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading RSI oversold stocks", ex.ToString());
                CustomModal.ShowError($"Error loading RSI oversold stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading RSI Oversold data";
            }
            finally
            {
                // Always re-enable button
                if (RsiOversoldButton != null)
                    RsiOversoldButton.IsEnabled = true;
            }
        }

        // RSI Overbought button click event handler
        private async void RsiOverboughtButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading RSI Overbought data...";
                }

                // Load RSI overbought stocks
                await LoadSymbolsForMode(SymbolSelectionMode.RsiOverbought);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for RSI Overbought";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading RSI overbought stocks", ex.ToString());
                CustomModal.ShowError($"Error loading RSI overbought stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading RSI Overbought data";
            }
        }

        // Top Volume RSI button click event handler
        private async void TopVolumeRsiButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Top Volume RSI data...";
                }

                // Load top volume RSI discrepancies stocks
                await LoadSymbolsForMode(SymbolSelectionMode.TopVolumeRsiDiscrepancies);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for Top Volume RSI Discrepancies";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading top volume RSI discrepancies", ex.ToString());
                CustomModal.ShowError($"Error loading top volume RSI discrepancies: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Top Volume RSI data";
            }
        }

        // Top P/E button click event handler
        private async void TopPEButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Top P/E data...";
                }

                // Load top P/E stocks
                await LoadSymbolsForMode(SymbolSelectionMode.TopPE);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for Top P/E";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading top P/E stocks", ex.ToString());
                CustomModal.ShowError($"Error loading top P/E stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Top P/E data";
            }
        }

        // High Volume button click event handler
        private async void HighVolumeButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading High Volume data...";
                }

                // Load high volume stocks
                await LoadSymbolsForMode(SymbolSelectionMode.HighVolume);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for High Volume";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading high volume stocks", ex.ToString());
                CustomModal.ShowError($"Error loading high volume stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading High Volume data";
            }
        }

        // Low P/E button click event handler
        private async void LowPEButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Low P/E data...";
                }

                // Load low P/E stocks
                await LoadSymbolsForMode(SymbolSelectionMode.LowPE);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for Low P/E";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading low P/E stocks", ex.ToString());
                CustomModal.ShowError($"Error loading low P/E stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Low P/E data";
            }
        }

        // All Database button click event handler
        private async void AllDatabaseButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading all cached database data...";
                }

                // Load all cached stocks directly from the cache service
                await LoadSymbolsForMode(SymbolSelectionMode.AllDatabase);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks from database cache";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading all database stocks", ex.ToString());
                CustomModal.ShowError($"Error loading all database stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading All Database data";
            }
        }

        private async void HighThetaButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading High Theta data...";
                }

                // Load high theta stocks
                await LoadSymbolsForMode(SymbolSelectionMode.HighTheta);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for High Theta";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading high theta stocks", ex.ToString());
                CustomModal.ShowError($"Error loading high theta stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading High Theta data";
            }
        }

        private async void HighBetaButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading High Beta data...";
                }

                // Load high beta stocks
                await LoadSymbolsForMode(SymbolSelectionMode.HighBeta);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for High Beta";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading high beta stocks", ex.ToString());
                CustomModal.ShowError($"Error loading high beta stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading High Beta data";
            }
        }

        private async void HighAlphaButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading High Alpha data...";
                }

                // Load high alpha stocks
                await LoadSymbolsForMode(SymbolSelectionMode.HighAlpha);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks for High Alpha";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading high alpha stocks", ex.ToString());
                CustomModal.ShowError($"Error loading high alpha stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading High Alpha data";
            }
        }

        private async void BullishCupAndHandleButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Bullish Cup and Handle data...";
                }

                // Load stocks with bullish cup and handle patterns
                await LoadSymbolsForMode(SymbolSelectionMode.BullishCupAndHandle);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks with Cup and Handle patterns";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading bullish cup and handle stocks", ex.ToString());
                CustomModal.ShowError($"Error loading bullish cup and handle stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Cup and Handle data";
            }
        }

        private async void BearishCupAndHandleButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show status that we're loading
                if (ModeStatusPanel != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Loading Bearish Cup and Handle data...";
                }

                // Load stocks with bearish cup and handle patterns
                await LoadSymbolsForMode(SymbolSelectionMode.BearishCupAndHandle);

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    var count = _viewModel?.CachedStocks?.Count ?? 0;
                    ModeStatusText.Text = $"Loaded {count} stocks with Bearish Cup and Handle patterns";
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading bearish cup and handle stocks", ex.ToString());
                CustomModal.ShowError($"Error loading bearish cup and handle stocks: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading Bearish Cup and Handle data";
            }
        }

        // Time Range Button Click Handler
        private async void TimeRangeButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string timeRange)
            {
                try
                {
                    // Update the current time range in the ViewModel
                    _viewModel.CurrentTimeRange = timeRange;
                    
                    // Update button styles
                    UpdateTimeRangeButtonStyles(timeRange);
                    
                    // If we have a selected symbol, reload its chart data with the new time range
                    if (!string.IsNullOrEmpty(_viewModel.SelectedSymbol))
                    {
                        await LoadChartDataForTimeRange(_viewModel.SelectedSymbol, timeRange);
                    }
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", $"Error changing time range to {timeRange}", ex.ToString());
                    CustomModal.ShowError($"Error changing time range: {ex.Message}", "Error", Window.GetWindow(this));
                }
            }
        }

        // Update time range button styles to show active selection
        private void UpdateTimeRangeButtonStyles(string activeTimeRange)
        {
            // Reset all buttons to default style
            var buttons = new[] { TimeRange1D, TimeRange5D, TimeRange1M, TimeRange6M, TimeRange1Y, TimeRange5Y, TimeRangeAll };
            
            foreach (var button in buttons)
            {
                if (button != null)
                {
                    button.Background = new SolidColorBrush(Color.FromRgb(0x3A, 0x6E, 0xA5)); // Default blue
                }
            }
            
            // Highlight the active button
            Button activeButton = activeTimeRange switch
            {
                "1day" => TimeRange1D,
                "5day" => TimeRange5D,
                "1mo" => TimeRange1M,
                "6mo" => TimeRange6M,
                "1y" => TimeRange1Y,
                "5y" => TimeRange5Y,
                "all" => TimeRangeAll,
                _ => TimeRange1D // Default to 1D instead of 1M
            };
            
            if (activeButton != null)
            {
                activeButton.Background = new SolidColorBrush(Color.FromRgb(0x6A, 0x5A, 0xCD)); // Purple for active
            }
        }

        // StockDataGrid selection changed event handler
        private async void StockDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isHandlingSelectionChanged || sender is not DataGrid dataGrid) 
                return;

            if (dataGrid.SelectedItem is QuoteData selectedQuote)
            {
                await HandleSymbolSelectionAsync(selectedQuote.Symbol, "DataGrid");
            }
        }

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            // In Individual Asset mode, search for the typed symbol
            if (CurrentSelectionMode == Quantra.Enums.SymbolSelectionMode.IndividualAsset)
            {
                var searchText = _viewModel.SymbolSearchText?.ToUpper().Trim();
                if (!string.IsNullOrEmpty(searchText) && _viewModel.FilteredSymbols.Contains(searchText))
                {
                    try
                    {
                        // Load data for the typed symbol
                        await HandleSymbolSelectionAsync(searchText, "SearchButton");
                    }
                    catch (System.OperationCanceledException)
                    {
                        // Operation was cancelled - handle gracefully for search
                        //DatabaseMonolith.Log("Info", "Stock symbol search was cancelled");
                        CustomModal.ShowWarning("Symbol search was cancelled.", "Operation Cancelled", Window.GetWindow(this));
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Error", "Error searching for stock data", ex.ToString());
                        CustomModal.ShowError($"Error searching for data: {ex.Message}", "Error", Window.GetWindow(this));
                    }
                }
                else
                {
                    CustomModal.ShowWarning("Please enter a valid stock symbol to search.", "Invalid Symbol", Window.GetWindow(this));
                }
            }
            else if (!string.IsNullOrEmpty(_viewModel.SelectedSymbol))
            {
                // Create a new cancellation token source for the refresh operation
                using var refreshCancellation = new System.Threading.CancellationTokenSource();
                var refreshToken = refreshCancellation.Token;
                
                try
                {
                    // Refresh the data for the currently selected symbol from API (for other modes)
                    await RefreshSymbolDataFromAPI(_viewModel.SelectedSymbol, refreshToken);
                    
                    // Update the price and RSI labels
                    await UpdatePriceAndRsiLabels(_viewModel.SelectedSymbol);
                    
                    // Reload indicator data asynchronously to avoid blocking UI
                    await LoadIndicatorDataAsync(_viewModel.SelectedSymbol, refreshToken);
                }
                catch (System.OperationCanceledException)
                {
                    // Operation was cancelled - this shouldn't normally happen for refresh but handle gracefully
                    //DatabaseMonolith.Log("Info", "Stock data refresh was cancelled");
                    CustomModal.ShowWarning("Data refresh was cancelled.", "Operation Cancelled", Window.GetWindow(this));
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", "Error refreshing stock data", ex.ToString());
                    CustomModal.ShowError($"Error refreshing data: {ex.Message}", "Error", Window.GetWindow(this));
                }
            }
            else
            {
                CustomModal.ShowWarning("Please select a stock symbol to refresh.", "No Symbol Selected", Window.GetWindow(this));
            }
        }

        // Sentiment Analysis Event Handler
        private async void RunSentimentAnalysisButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Check if we have a selected symbol
                var selectedStock = StockDataGrid?.SelectedItem as QuoteData;
                if (selectedStock == null || string.IsNullOrEmpty(selectedStock.Symbol))
                {
                    SentimentError = "Please select a stock symbol first.";
                    return;
                }

                // Clear previous errors and results
                SentimentError = "";
                HasSentimentResults = false;
                IsSentimentLoading = true;

                // Set busy cursor during sentiment analysis
                Mouse.OverrideCursor = Cursors.Wait;

                string symbol = selectedStock.Symbol;

                //DatabaseMonolith.Log("Info", $"Starting sentiment analysis for {symbol}");

                // Run sentiment analysis for different sources in parallel
                var sentimentTasks = new List<Task<double>>();
                
                // News sentiment
                var newsTask = GetNewsSentimentAsync(symbol);
                sentimentTasks.Add(newsTask);
                
                // Social media sentiment (using OpenAI for now)
                var socialMediaTask = GetSocialMediaSentimentAsync(symbol);
                sentimentTasks.Add(socialMediaTask);
                
                // Analyst sentiment
                var analystTask = GetAnalystSentimentAsync(symbol);
                sentimentTasks.Add(analystTask);

                // Wait for all sentiment analysis tasks to complete
                var results = await Task.WhenAll(sentimentTasks);

                // Update UI with results
                await Dispatcher.InvokeAsync(() =>
                {
                    NewsSentimentScore = results[0];
                    SocialMediaSentimentScore = results[1];
                    AnalystSentimentScore = results[2];

                    // Calculate overall sentiment (weighted average)
                    OverallSentimentScore = (NewsSentimentScore + SocialMediaSentimentScore + AnalystSentimentScore) / 3.0;

                    // Generate summary
                    SentimentSummary = GenerateSentimentSummary(symbol);

                    HasSentimentResults = true;
                    IsSentimentLoading = false;

                    // Reset cursor when sentiment analysis completes
                    Mouse.OverrideCursor = null;
                });

                //DatabaseMonolith.Log("Info", $"Sentiment analysis completed for {symbol}. Overall: {OverallSentimentScore:F2}");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error during sentiment analysis", ex.ToString());
                
                await Dispatcher.InvokeAsync(() =>
                {
                    SentimentError = $"Error running sentiment analysis: {ex.Message}";
                    IsSentimentLoading = false;
                    
                    // Reset cursor on error
                    Mouse.OverrideCursor = null;
                });
            }
        }

        private void SymbolSearchTimer_Tick(object sender, EventArgs e)
        {
            _symbolSearchTimer?.Stop();
            
            // Handle delayed symbol search logic
            if (CurrentSelectionMode == SymbolSelectionMode.IndividualAsset && _viewModel != null)
            {
                ValidateSearchButtonState();
            }
        }
    }
}
