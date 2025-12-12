// Handles UI event handlers for StockExplorer
using System;
using System.Linq;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using Quantra.Models;
using System.Threading.Tasks;
using System.Windows.Threading;
using Quantra.Views.Shared;
using Quantra.Enums;
using Quantra.Views.StockExplorer;
using Quantra.DAL.Enums;
using Quantra.DAL.Services;

namespace Quantra.Controls
{
    public partial class StockExplorer
    {
        private string _lastSearchText = "";
        
        // Tracks whether a dropdown selection was made (vs just closing without selection)
        private bool _dropdownSelectionMade = false;
        private string _selectedSymbolBeforeDropdown = null;
        
        // Debounce timer for scroll-based pagination
        private DispatcherTimer _scrollDebounceTimer;
        private bool _scrollLoadPending = false;

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

        // Load All Historicals button click event handler
        private async void LoadAllHistoricalsButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Show mode selection dialog
                var modeResult = LoadHistoricalsModeWindow.Show(
                    _stockConfigurationService,
                    GetLoadedSymbolCount(),
                    GetTotalSymbolCount(),
                    Window.GetWindow(this));

                if (modeResult == null)
                {
                    // User cancelled
                    return;
                }

                var (selectedMode, selectedSymbols) = modeResult.Value;

                // Disable button to prevent multiple clicks
                if (LoadAllHistoricalsButton != null)
                    LoadAllHistoricalsButton.IsEnabled = false;

                // Show loading status
                if (ModeStatusPanel != null && ModeStatusText != null)
                {
                    ModeStatusPanel.Visibility = Visibility.Visible;
                    ModeStatusText.Text = "Initializing historical data load...";
                }

                // Set busy cursor during loading
                Mouse.OverrideCursor = Cursors.Wait;
                
                // Get the list of tickers based on selected mode
                List<string> tickers = await GetTickersForMode(selectedMode, selectedSymbols);
                
                if (tickers == null || !tickers.Any())
                {
                    CustomModal.ShowWarning("No symbols to load based on selected mode.", "No Symbols", Window.GetWindow(this));
                    return;
                }
                
                if (ModeStatusText != null)
                {
                    ModeStatusText.Text = $"Loading historical data for {tickers.Count} symbols...";
                }
                
                // Show the counter in SharedTitleBar
                SharedTitleBar.SetLoadAllHistoricalsActive(true, $"Loading: 0/{tickers.Count}");

                int successCount = 0;
                int errorCount = 0;
                int totalCount = tickers.Count;

                // Batch processing constants for API rate limiting
                const int BATCH_SIZE = 5;
                const int DELAY_BETWEEN_BATCHES_MS = 2000;
                const int COMPLETION_MESSAGE_DELAY_MS = 3000;

                for (int i = 0; i < tickers.Count; i += BATCH_SIZE)
                {
                    var batch = tickers.Skip(i).Take(BATCH_SIZE).ToList();
                    var batchTasks = new List<Task<bool>>();

                    foreach (var ticker in batch)
                    {
                        batchTasks.Add(Task.Run(async () =>
                        {
                            try
                            {
                                // Use TIME_SERIES_DAILY_ADJUSTED via GetExtendedHistoricalData
                                var historicalData = await _alphaVantageService.GetExtendedHistoricalData(ticker, "daily", "full");
                                
                                if (historicalData != null && historicalData.Count > 0)
                                {
                                    // Store the historical data in the cache using consistent parameters
                                    await _cacheService.CacheHistoricalDataAsync(ticker, "daily", "daily", historicalData);
                                    return true;
                                }
                                return false;
                            }
                            catch (Exception ex)
                            {
                                //DatabaseMonolith.Log("Warning", $"Failed to load historical data for {ticker}", ex.ToString());
                                return false;
                            }
                        }));
                    }

                    // Wait for batch to complete
                    var batchResults = await Task.WhenAll(batchTasks);
                    
                    successCount += batchResults.Count(r => r);
                    errorCount += batchResults.Count(r => !r);

                    // Update counter in SharedTitleBar
                    int processedCount = Math.Min(i + BATCH_SIZE, totalCount);
                    int remainingCount = totalCount - processedCount;
                    SharedTitleBar.UpdateLoadAllHistoricalsCounter(remainingCount, totalCount);
                    
                    // Update status text
                    await Dispatcher.InvokeAsync(() =>
                    {
                        if (ModeStatusText != null)
                            ModeStatusText.Text = $"Loading historical data... {processedCount}/{totalCount} processed";
                    });

                    // Add delay between batches to respect API limits
                    if (i + BATCH_SIZE < tickers.Count)
                    {
                        await Task.Delay(DELAY_BETWEEN_BATCHES_MS);
                    }
                }

                // Update status to show completion
                if (ModeStatusText != null)
                {
                    ModeStatusText.Text = $"Loaded historical data for {successCount} stocks. {errorCount} errors.";
                }

                // Hide the counter in SharedTitleBar
                SharedTitleBar.SetLoadAllHistoricalsActive(false, $"Complete: {successCount}/{totalCount}");
                
                // Show completion message after a delay then hide counter
                await Task.Delay(COMPLETION_MESSAGE_DELAY_MS);
                SharedTitleBar.SetLoadAllHistoricalsActive(false);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error loading all historical data", ex.ToString());
                CustomModal.ShowError($"Error loading all historical data: {ex.Message}", "Error", Window.GetWindow(this));
                
                // Show error status
                if (ModeStatusText != null)
                    ModeStatusText.Text = "Error loading all historical data";
                    
                // Hide the counter in SharedTitleBar
                SharedTitleBar.SetLoadAllHistoricalsActive(false);
            }
            finally
            {
                // Always re-enable button and reset cursor
                if (LoadAllHistoricalsButton != null)
                    LoadAllHistoricalsButton.IsEnabled = true;
                    
                Mouse.OverrideCursor = null;
            }
        }

        /// <summary>
        /// Gets tickers based on the selected load mode
        /// </summary>
        private async Task<List<string>> GetTickersForMode(HistoricalsLoadMode mode, List<string> configSymbols)
        {
            switch (mode)
            {
                case HistoricalsLoadMode.AllSymbols:
                    return await GetAllSymbolsFromApi();
                    
                case HistoricalsLoadMode.NonLoadedOnly:
                    return await GetNonLoadedSymbols();
                    
                case HistoricalsLoadMode.StockConfiguration:
                    return configSymbols ?? new List<string>();
                    
                default:
                    return new List<string>();
            }
        }

        /// <summary>
        /// Gets all symbols from AlphaVantage API and caches them
        /// </summary>
        private async Task<List<string>> GetAllSymbolsFromApi()
        {
            if (ModeStatusText != null)
            {
                ModeStatusText.Text = "Fetching publicly traded tickers from AlphaVantage...";
            }

            SharedTitleBar.SetLoadAllHistoricalsActive(true, "Caching publicly traded tickers...");

            try
            {
                var allSymbols = await _alphaVantageService.GetAllStockSymbols();
                
                if (allSymbols != null && allSymbols.Any())
                {
                    // Cache the symbols in the database using StockSymbolCacheService
                    var stockSymbols = allSymbols.Select(symbol => new Models.StockSymbol
                    {
                        Symbol = symbol,
                        Name = string.Empty,
                        Sector = string.Empty,
                        Industry = string.Empty,
                        LastUpdated = DateTime.Now
                    }).ToList();
                    
                    _stockSymbolCacheService.CacheStockSymbols(stockSymbols);
                    
                    if (ModeStatusText != null)
                    {
                        ModeStatusText.Text = $"Cached {allSymbols.Count} publicly traded tickers from AlphaVantage";
                    }
                    
                    await Task.Delay(1500);
                    return allSymbols;
                }
            }
            catch (Exception ex)
            {
                if (ModeStatusText != null)
                {
                    ModeStatusText.Text = "Failed to fetch symbols from AlphaVantage, using database cache...";
                }
                await Task.Delay(1500);
            }

            // Fallback to database cache
            return _stockSymbolCacheService.GetAllSymbolsAsList();
        }

        /// <summary>
        /// Gets symbols that don't have cached historical data
        /// </summary>
        private async Task<List<string>> GetNonLoadedSymbols()
        {
            if (ModeStatusText != null)
            {
                ModeStatusText.Text = "Identifying symbols without cached historical data...";
            }

            try
            {
                // First ensure we have all symbols cached
                var allSymbols = await GetAllSymbolsFromApi();
                
                if (allSymbols == null || !allSymbols.Any())
                {
                    return new List<string>();
                }

                // Get symbols that already have cached data
                var loadedSymbols = _cacheService.GetAllCachedSymbols();
                var loadedSet = new HashSet<string>(loadedSymbols, StringComparer.OrdinalIgnoreCase);

                // Return only symbols without cached data
                var nonLoadedSymbols = allSymbols.Where(s => !loadedSet.Contains(s)).ToList();

                if (ModeStatusText != null)
                {
                    ModeStatusText.Text = $"Found {nonLoadedSymbols.Count} symbols without cached data (out of {allSymbols.Count} total)";
                }

                await Task.Delay(1000);
                return nonLoadedSymbols;
            }
            catch (Exception ex)
            {
                if (ModeStatusText != null)
                {
                    ModeStatusText.Text = $"Error identifying non-loaded symbols: {ex.Message}";
                }
                return new List<string>();
            }
        }

        /// <summary>
        /// Gets the count of loaded symbols for display
        /// </summary>
        private int GetLoadedSymbolCount()
        {
            try
            {
                var loadedSymbols = _cacheService.GetAllCachedSymbols();
                return loadedSymbols?.Count ?? 0;
            }
            catch
            {
                return 0;
            }
        }

        /// <summary>
        /// Gets the total symbol count for display
        /// </summary>
        private int GetTotalSymbolCount()
        {
            try
            {
                var allSymbols = _stockSymbolCacheService.GetAllSymbolsAsList();
                return allSymbols?.Count > 0 ? allSymbols.Count : LoadHistoricalsModeWindow.DEFAULT_TOTAL_SYMBOL_COUNT;
            }
            catch
            {
                return LoadHistoricalsModeWindow.DEFAULT_TOTAL_SYMBOL_COUNT;
            }
        }

        // Time Range Button Click Handler
        private async void TimeRangeButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is string timeRange)
            {
                // Set waiting cursor immediately on UI thread
                Mouse.OverrideCursor = Cursors.Wait;
                
                try
                {
                    // Update the current time range in the ViewModel
                    _viewModel.CurrentTimeRange = timeRange;
                                        
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
                finally
                {
                    // Always reset cursor back to normal on UI thread
                    Mouse.OverrideCursor = null;
                }
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
                        CustomModal.ShowWarning("Symbol search was cancelled.", "Operation Cancelled", Window.GetWindow(this));
                    }
                    catch (Exception ex)
                    {
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
                    CustomModal.ShowWarning("Data refresh was cancelled.", "Operation Cancelled", Window.GetWindow(this));
                }
                catch (Exception ex)
                {
                    CustomModal.ShowError($"Error refreshing data: {ex.Message}", "Error", Window.GetWindow(this));
                }
            }
            else
            {
                CustomModal.ShowWarning("Please select a stock symbol to refresh.", "No Symbol Selected", Window.GetWindow(this));
            }
        }

        /// <summary>
        /// Handles the refresh button click for the selected symbol - fetches latest data from API
        /// </summary>
        private async void RefreshSymbolDataButton_Click(object sender, RoutedEventArgs e)
        {
            var selectedSymbol = _viewModel?.SelectedSymbol;
            if (string.IsNullOrEmpty(selectedSymbol))
            {
                CustomModal.ShowWarning("Please select a stock symbol to refresh.", "No Symbol Selected", Window.GetWindow(this));
                return;
            }

            // Disable the button during refresh
            if (RefreshSymbolDataButton != null)
                RefreshSymbolDataButton.IsEnabled = false;

            // Create a new cancellation token source for the refresh operation
            using var refreshCancellation = new System.Threading.CancellationTokenSource();
            var refreshToken = refreshCancellation.Token;

            try
            {
                // Refresh the data for the currently selected symbol from API
                // Note: RefreshSymbolDataFromAPI handles cursor state internally
                await RefreshSymbolDataFromAPI(selectedSymbol, refreshToken);

                // Update the cache timestamp display
                UpdateCacheTimestampDisplay(selectedSymbol);

                // Update the price and RSI labels
                await UpdatePriceAndRsiLabels(selectedSymbol);

                // Reload indicator data asynchronously to avoid blocking UI
                await LoadIndicatorDataAsync(selectedSymbol, refreshToken);
            }
            catch (System.OperationCanceledException)
            {
                CustomModal.ShowWarning("Data refresh was cancelled.", "Operation Cancelled", Window.GetWindow(this));
            }
            catch (Exception ex)
            {
                CustomModal.ShowError($"Error refreshing data: {ex.Message}", "Error", Window.GetWindow(this));
            }
            finally
            {
                // Re-enable the button
                if (RefreshSymbolDataButton != null)
                    RefreshSymbolDataButton.IsEnabled = true;
            }
        }

        /// <summary>
        /// Updates the cache timestamp display for the selected symbol
        /// </summary>
        private void UpdateCacheTimestampDisplay(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                CacheTimestampText = "--";
                CacheTimestampColor = System.Windows.Media.Brushes.White;
                SelectedSymbolDisplay = "--";
                OnPropertyChanged(nameof(CanRefreshSymbol));
                return;
            }

            SelectedSymbolDisplay = symbol;

            // Get cache timestamp from the selected stock
            var selectedStock = _viewModel?.SelectedStock;
            if (selectedStock?.CacheTime.HasValue == true)
            {
                var cacheTime = selectedStock.CacheTime.Value;
                var age = DateTime.Now - cacheTime;
                
                // Format the timestamp
                CacheTimestampText = cacheTime.ToString("MM/dd/yyyy HH:mm");

                // Color based on age: Green < 1 hour, Yellow < 24 hours, Orange < 7 days, Red >= 7 days
                if (age.TotalHours < 1)
                {
                    CacheTimestampColor = System.Windows.Media.Brushes.LightGreen;
                }
                else if (age.TotalHours < 24)
                {
                    CacheTimestampColor = System.Windows.Media.Brushes.Yellow;
                }
                else if (age.TotalDays < 7)
                {
                    CacheTimestampColor = System.Windows.Media.Brushes.Orange;
                }
                else
                {
                    CacheTimestampColor = System.Windows.Media.Brushes.OrangeRed;
                }
            }
            else
            {
                CacheTimestampText = "Not cached";
                CacheTimestampColor = System.Windows.Media.Brushes.Gray;
            }

            OnPropertyChanged(nameof(CanRefreshSymbol));
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

        private async void SymbolSearchTimer_Tick(object sender, EventArgs e)
        {
            _symbolSearchTimer?.Stop();
            
            var searchText = SymbolSearchTextBox?.Text?.Trim();
            if (string.IsNullOrEmpty(searchText) || searchText == _lastSearchText || searchText.Length < 1)
                return;
                
            _lastSearchText = searchText;
            
            try
            {
                if (SearchLoadingText != null)
                    SearchLoadingText.Visibility = Visibility.Visible;
                    
                var results = await _alphaVantageService.SearchSymbolsAsync(searchText);
                
                if (SearchResultsListBox != null)
                    SearchResultsListBox.ItemsSource = results;
                    
                if (SearchLoadingText != null)
                    SearchLoadingText.Visibility = Visibility.Collapsed;
                
                if (results.Count > 0 && SearchResultsPopup != null)
                {
                    SearchResultsPopup.IsOpen = true;
                }
            }
            catch (Exception ex)
            {
                if (SearchLoadingText != null)
                    SearchLoadingText.Visibility = Visibility.Collapsed;
                CustomModal.ShowError($"Error searching symbols: {ex.Message}", "Search Error", Window.GetWindow(this));
            }
        }
        
        private async void AutoRefreshTimer_Tick(object sender, EventArgs e)
        {
            if (!_isAutoRefreshEnabled)
                return;
                
            try
            {
                await PerformAutoRefresh();
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Error during auto-refresh", ex.ToString());
            }
        }
        
        private void AutoRefreshToggleButton_Checked(object sender, RoutedEventArgs e)
        {
            _isAutoRefreshEnabled = true;
            _autoRefreshTimer?.Start();

            // Update status text with interval
            if (AutoRefreshStatusText != null)
            {
                AutoRefreshStatusText.Visibility = Visibility.Visible;
                UpdateAutoRefreshStatusText();
                AutoRefreshStatusText.Foreground = System.Windows.Media.Brushes.LimeGreen;
            }

            _loggingService?.Log("Info", "Auto-refresh enabled for StockExplorer");

            // Save setting to user settings
            SaveAutoRefreshState(true);

            // Perform immediate refresh
            _ = PerformAutoRefresh();
        }
        
        private void AutoRefreshToggleButton_Unchecked(object sender, RoutedEventArgs e)
        {
            _isAutoRefreshEnabled = false;
            _autoRefreshTimer?.Stop();

            // Update status text
            if (AutoRefreshStatusText != null)
            {
                AutoRefreshStatusText.Visibility = Visibility.Collapsed;
                AutoRefreshStatusText.Text = "";
            }

            _loggingService?.Log("Info", "Auto-refresh disabled for StockExplorer");

            // Save setting to user settings
            SaveAutoRefreshState(false);
        }
        
        private async System.Threading.Tasks.Task PerformAutoRefresh()
        {
            if (_viewModel == null || _alphaVantageService == null)
                return;
                
            try
            {
                // Update status text
                await Dispatcher.InvokeAsync(() =>
                {
                    if (AutoRefreshStatusText != null)
                    {
                        AutoRefreshStatusText.Text = "Refreshing...";
                        AutoRefreshStatusText.Foreground = System.Windows.Media.Brushes.Yellow;
                    }
                });
                
                // Get visible stocks in the grid - ONLY the currently displayed page in pagination
                var visibleStocks = _viewModel.CachedStocks.ToList();
                
                if (visibleStocks.Count == 0)
                {
                    await Dispatcher.InvokeAsync(() =>
                    {
                        if (AutoRefreshStatusText != null)
                        {
                            AutoRefreshStatusText.Text = "No stocks to refresh";
                            AutoRefreshStatusText.Foreground = System.Windows.Media.Brushes.Orange;
                        }
                    });
                    return;
                }
                
                _loggingService?.Log("Info", $"Auto-refreshing {visibleStocks.Count} stocks on page {_viewModel.CurrentPage}");
                
                int successCount = 0;
                int errorCount = 0;
                
                // Process stocks in batches to avoid API rate limits
                const int batchSize = 5;
                for (int i = 0; i < visibleStocks.Count; i += batchSize)
                {
                    var batch = visibleStocks.Skip(i).Take(batchSize).ToList();
                    var batchTasks = new System.Collections.Generic.List<System.Threading.Tasks.Task>();
                    
                    foreach (var stock in batch)
                    {
                        batchTasks.Add(System.Threading.Tasks.Task.Run(async () =>
                        {
                            try
                            {
                                // Get quote data to recalculate indicators from OHLCV
                                var quoteData = await _alphaVantageService.GetQuoteDataAsync(stock.Symbol);
                                
                                if (quoteData != null)
                                {
                                    // Update core quote data
                                    stock.Price = quoteData.Price;
                                    stock.Volume = quoteData.Volume;
                                    stock.DayHigh = quoteData.DayHigh;
                                    stock.DayLow = quoteData.DayLow;
                                    stock.ChangePercent = quoteData.ChangePercent;
                                    stock.Change = quoteData.Change;
                                    
                                    // Reload calculated indicators for grid display
                                    var rsi = await _alphaVantageService.GetRSI(stock.Symbol);
                                    stock.RSI = rsi;
                                    
                                    // Reload VWAP
                                    var vwap = await _alphaVantageService.GetVWAP(stock.Symbol);
                                    stock.VWAP = vwap;
                                    
                                    // Reload P/E Ratio (if available)
                                    var peRatio = await _alphaVantageService.GetPERatioAsync(stock.Symbol);
                                    stock.PERatio = peRatio ?? stock.PERatio;
                                    
                                    // Update timestamp to reflect when data was refreshed
                                    stock.LastAccessed = DateTime.Now;
                                    stock.CacheTime = DateTime.Now;
                                    
                                    // Cache the updated quote data
                                    await _cacheService.CacheQuoteDataAsync(quoteData);
                                }
                                
                                System.Threading.Interlocked.Increment(ref successCount);
                            }
                            catch (Exception ex)
                            {
                                _loggingService?.Log("Warning", $"Failed to refresh {stock.Symbol}", ex.ToString());
                                System.Threading.Interlocked.Increment(ref errorCount);
                            }
                        }));
                    }
                    
                    // Wait for batch to complete
                    await System.Threading.Tasks.Task.WhenAll(batchTasks);
                    
                    // Add delay between batches to respect API rate limits
                    if (i + batchSize < visibleStocks.Count)
                    {
                        await System.Threading.Tasks.Task.Delay(1000); // 1 second delay
                    }
                }
                
                // Update last refresh time
                _lastAutoRefreshTime = DateTime.Now;
                
                // Refresh the DataGrid to show updated values
                await Dispatcher.InvokeAsync(() =>
                {
                    StockDataGrid?.Items.Refresh();
                    
                    // Update status text
                    if (AutoRefreshStatusText != null)
                    {
                        AutoRefreshStatusText.Text = $"Last refresh: {_lastAutoRefreshTime:HH:mm:ss} ({successCount} OK, {errorCount} errors)";
                        AutoRefreshStatusText.Foreground = errorCount > 0 ? System.Windows.Media.Brushes.Orange : System.Windows.Media.Brushes.LimeGreen;
                    }
                });
                
                _loggingService?.Log("Info", $"Auto-refresh completed: {successCount} successful, {errorCount} errors");
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Error performing auto-refresh", ex.ToString());
                
                await Dispatcher.InvokeAsync(() =>
                {
                    if (AutoRefreshStatusText != null)
                    {
                        AutoRefreshStatusText.Text = "Refresh failed";
                        AutoRefreshStatusText.Foreground = System.Windows.Media.Brushes.Red;
                    }
                });
            }
        }

        /// <summary>
        /// Handles scroll events on the StockDataGrid to implement auto-pagination on scroll
        /// </summary>
        private void StockDataGrid_ScrollChanged(object sender, ScrollChangedEventArgs e)
        {
            // Only trigger load more when scrolling vertically and near the bottom
            if (e.VerticalChange <= 0)
                return;
                
            // Check if we're at or near the bottom of the scroll area
            var scrollViewer = e.OriginalSource as ScrollViewer;
            if (scrollViewer == null)
                return;
                
            // Calculate if we're within 90% of the bottom
            var verticalOffset = scrollViewer.VerticalOffset;
            var scrollableHeight = scrollViewer.ScrollableHeight;
            
            if (scrollableHeight <= 0)
                return;
                
            var scrollPercentage = verticalOffset / scrollableHeight;
            
            // If scrolled past 90% and we have more pages, schedule a debounced load
            // This will automatically load the next page when user scrolls to bottom
            if (scrollPercentage >= 0.9 && _viewModel != null && _viewModel.HasMorePages && !_viewModel.IsLoading && !_scrollLoadPending)
            {
                _scrollLoadPending = true;
                
                // Initialize debounce timer if needed
                if (_scrollDebounceTimer == null)
                {
                    _scrollDebounceTimer = new DispatcherTimer
                    {
                        Interval = TimeSpan.FromMilliseconds(300)
                    };
                    _scrollDebounceTimer.Tick += async (s, args) =>
                    {
                        _scrollDebounceTimer.Stop();
                        _scrollLoadPending = false;
                        
                        if (_viewModel != null && _viewModel.HasMorePages && !_viewModel.IsLoading)
                        {
                            await _viewModel.LoadMoreCachedStocksAsync();
                        }
                    };
                }
                
                // Reset and start the debounce timer
                _scrollDebounceTimer.Stop();
                _scrollDebounceTimer.Start();
            }
        }

        /// <summary>
        /// Handles double-click event on DataGrid row to open candlestick chart modal
        /// </summary>
        private void StockDataGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            try
            {
                // Get the clicked row
                if (StockDataGrid?.SelectedItem is QuoteData selectedStock && !string.IsNullOrEmpty(selectedStock.Symbol))
                {
                    _loggingService?.Log("Info", $"Opening candlestick chart modal for {selectedStock.Symbol}");
                    
                    // Get AlphaVantageService from DI container to ensure it's not null
                    var alphaVantageService = _alphaVantageService ?? App.ServiceProvider?.GetService(typeof(AlphaVantageService)) as AlphaVantageService;
                    
                    if (alphaVantageService == null)
                    {
                        _loggingService?.Log("Error", "AlphaVantageService is not available - cannot open candlestick chart");
                        CustomModal.ShowError("AlphaVantage service is not initialized. Cannot load candlestick data.", "Service Error", Window.GetWindow(this));
                        return;
                    }
                    
                    // Create and show the modal
                    var modal = new CandlestickChartModal(
                        selectedStock.Symbol,
                        alphaVantageService,
                        _loggingService
                    );
                    
                    modal.Owner = Window.GetWindow(this);
                    modal.ShowDialog();
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to open candlestick chart modal");
                CustomModal.ShowError($"Failed to open candlestick chart: {ex.Message}", "Error", Window.GetWindow(this));
            }
        }
    }
}
