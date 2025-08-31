using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using Quantra.Models;
using Quantra.ViewModels;
using Quantra.DAL.Services.Interfaces;
using Quantra.Enums;

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for SpreadsExplorer.xaml
    /// </summary>
    public partial class SpreadsExplorer : UserControl
    {
        private SpreadsExplorerViewModel _viewModel;
        private AlphaVantageService _alphaVantageService;

        public SpreadsExplorer()
        {
            InitializeComponent();
            
            _alphaVantageService = new AlphaVantageService();
            _viewModel = new SpreadsExplorerViewModel();
            
            DataContext = _viewModel;
            
            // Initialize with default values
            InitializeDefaultSpread();
        }

        private void InitializeDefaultSpread()
        {
            try
            {
                // Set default symbol
                _viewModel.UnderlyingSymbol = "AAPL";
                _viewModel.UnderlyingPrice = 150.0; // Default price, will be updated when loaded
                
                // Add initial legs for a simple spread
                AddDefaultLegs();
                
                DatabaseMonolith.Log("Info", "SpreadsExplorer initialized with default spread");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error initializing SpreadsExplorer", ex.ToString());
            }
        }

        private void AddDefaultLegs()
        {
            // Add two legs for a basic vertical spread
            var leg1 = new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 5.0,
                Option = new OptionData
                {
                    UnderlyingSymbol = _viewModel.UnderlyingSymbol,
                    OptionType = "CALL",
                    StrikePrice = 145.0,
                    ExpirationDate = DateTime.Now.AddDays(30),
                    ImpliedVolatility = 0.25,
                    Delta = 0.60,
                    Gamma = 0.05,
                    Theta = -0.15,
                    Vega = 0.20,
                    Rho = 0.10
                }
            };

            var leg2 = new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 2.0,
                Option = new OptionData
                {
                    UnderlyingSymbol = _viewModel.UnderlyingSymbol,
                    OptionType = "CALL",
                    StrikePrice = 155.0,
                    ExpirationDate = DateTime.Now.AddDays(30),
                    ImpliedVolatility = 0.23,
                    Delta = 0.35,
                    Gamma = 0.04,
                    Theta = -0.10,
                    Vega = 0.15,
                    Rho = 0.08
                }
            };

            _viewModel.OptionLegs.Add(leg1);
            _viewModel.OptionLegs.Add(leg2);
        }

        private void SpreadTypeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            try
            {
                if (SpreadTypeComboBox.SelectedItem is ComboBoxItem selectedItem)
                {
                    var spreadType = selectedItem.Content.ToString();
                    var strategyType = selectedItem.Tag.ToString();
                    
                    _viewModel.SpreadType = spreadType;
                    _viewModel.StrategyType = ParseStrategyType(strategyType);
                    
                    // Auto-configure legs based on spread type
                    ConfigureLegsForSpreadType(spreadType);
                    
                    DatabaseMonolith.Log("Info", $"Spread type changed to: {spreadType}");
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error changing spread type", ex.ToString());
            }
        }

        private MultiLegStrategyType ParseStrategyType(string strategyType)
        {
            return strategyType switch
            {
                "VerticalSpread" => MultiLegStrategyType.VerticalSpread,
                "Straddle" => MultiLegStrategyType.Straddle,
                "Strangle" => MultiLegStrategyType.Strangle,
                "IronCondor" => MultiLegStrategyType.IronCondor,
                "ButterflySpread" => MultiLegStrategyType.ButterflySpread,
                "CalendarSpread" => MultiLegStrategyType.CalendarSpread,
                "CoveredCall" => MultiLegStrategyType.CoveredCall,
                _ => MultiLegStrategyType.Custom
            };
        }

        private void ConfigureLegsForSpreadType(string spreadType)
        {
            try
            {
                // Clear existing legs
                _viewModel.OptionLegs.Clear();
                
                var currentPrice = _viewModel.UnderlyingPrice;
                var expiryDate = DateTime.Now.AddDays(30);
                
                switch (spreadType)
                {
                    case "Bull Call Spread":
                        ConfigureBullCallSpread(currentPrice, expiryDate);
                        break;
                    case "Bear Put Spread":
                        ConfigureBearPutSpread(currentPrice, expiryDate);
                        break;
                    case "Long Straddle":
                        ConfigureLongStraddle(currentPrice, expiryDate);
                        break;
                    case "Short Straddle":
                        ConfigureShortStraddle(currentPrice, expiryDate);
                        break;
                    case "Long Strangle":
                        ConfigureLongStrangle(currentPrice, expiryDate);
                        break;
                    case "Short Strangle":
                        ConfigureShortStrangle(currentPrice, expiryDate);
                        break;
                    case "Iron Condor":
                        ConfigureIronCondor(currentPrice, expiryDate);
                        break;
                    case "Butterfly Spread":
                        ConfigureButterflySpread(currentPrice, expiryDate);
                        break;
                    case "Covered Call":
                        ConfigureCoveredCall(currentPrice, expiryDate);
                        break;
                    default:
                        // Keep existing legs for custom spreads
                        break;
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error configuring legs for spread type", ex.ToString());
            }
        }

        private void ConfigureBullCallSpread(double currentPrice, DateTime expiryDate)
        {
            var lowerStrike = Math.Round(currentPrice * 0.95, 2);
            var upperStrike = Math.Round(currentPrice * 1.05, 2);
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 5.0,
                Option = CreateOptionData("CALL", lowerStrike, expiryDate, 0.60, 0.25)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 2.0,
                Option = CreateOptionData("CALL", upperStrike, expiryDate, 0.35, 0.23)
            });
        }

        private void ConfigureBearPutSpread(double currentPrice, DateTime expiryDate)
        {
            var lowerStrike = Math.Round(currentPrice * 0.95, 2);
            var upperStrike = Math.Round(currentPrice * 1.05, 2);
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 5.0,
                Option = CreateOptionData("PUT", upperStrike, expiryDate, -0.60, 0.25)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 2.0,
                Option = CreateOptionData("PUT", lowerStrike, expiryDate, -0.35, 0.23)
            });
        }

        private void ConfigureLongStraddle(double currentPrice, DateTime expiryDate)
        {
            var strike = Math.Round(currentPrice, 2);
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 5.0,
                Option = CreateOptionData("CALL", strike, expiryDate, 0.50, 0.25)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 5.0,
                Option = CreateOptionData("PUT", strike, expiryDate, -0.50, 0.25)
            });
        }

        private void ConfigureShortStraddle(double currentPrice, DateTime expiryDate)
        {
            var strike = Math.Round(currentPrice, 2);
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 5.0,
                Option = CreateOptionData("CALL", strike, expiryDate, 0.50, 0.25)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 5.0,
                Option = CreateOptionData("PUT", strike, expiryDate, -0.50, 0.25)
            });
        }

        private void ConfigureLongStrangle(double currentPrice, DateTime expiryDate)
        {
            var callStrike = Math.Round(currentPrice * 1.05, 2);
            var putStrike = Math.Round(currentPrice * 0.95, 2);
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 3.0,
                Option = CreateOptionData("CALL", callStrike, expiryDate, 0.35, 0.23)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 3.0,
                Option = CreateOptionData("PUT", putStrike, expiryDate, -0.35, 0.23)
            });
        }

        private void ConfigureShortStrangle(double currentPrice, DateTime expiryDate)
        {
            var callStrike = Math.Round(currentPrice * 1.05, 2);
            var putStrike = Math.Round(currentPrice * 0.95, 2);
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 3.0,
                Option = CreateOptionData("CALL", callStrike, expiryDate, 0.35, 0.23)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 3.0,
                Option = CreateOptionData("PUT", putStrike, expiryDate, -0.35, 0.23)
            });
        }

        private void ConfigureIronCondor(double currentPrice, DateTime expiryDate)
        {
            var putStrike1 = Math.Round(currentPrice * 0.90, 2);
            var putStrike2 = Math.Round(currentPrice * 0.95, 2);
            var callStrike1 = Math.Round(currentPrice * 1.05, 2);
            var callStrike2 = Math.Round(currentPrice * 1.10, 2);
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 1.0,
                Option = CreateOptionData("PUT", putStrike1, expiryDate, -0.20, 0.20)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 2.5,
                Option = CreateOptionData("PUT", putStrike2, expiryDate, -0.35, 0.23)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 2.5,
                Option = CreateOptionData("CALL", callStrike1, expiryDate, 0.35, 0.23)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 1.0,
                Option = CreateOptionData("CALL", callStrike2, expiryDate, 0.20, 0.20)
            });
        }

        private void ConfigureButterflySpread(double currentPrice, DateTime expiryDate)
        {
            var lowerStrike = Math.Round(currentPrice * 0.95, 2);
            var middleStrike = Math.Round(currentPrice, 2);
            var upperStrike = Math.Round(currentPrice * 1.05, 2);
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 5.0,
                Option = CreateOptionData("CALL", lowerStrike, expiryDate, 0.60, 0.25)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 2,
                Price = 3.0,
                Option = CreateOptionData("CALL", middleStrike, expiryDate, 0.50, 0.24)
            });
            
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "BUY",
                Quantity = 1,
                Price = 2.0,
                Option = CreateOptionData("CALL", upperStrike, expiryDate, 0.35, 0.23)
            });
        }

        private void ConfigureCoveredCall(double currentPrice, DateTime expiryDate)
        {
            var callStrike = Math.Round(currentPrice * 1.05, 2);
            
            // Note: In a real covered call, you'd also have long stock position
            // For simplicity, we're just showing the call option leg
            _viewModel.OptionLegs.Add(new OptionLeg
            {
                Action = "SELL",
                Quantity = 1,
                Price = 3.0,
                Option = CreateOptionData("CALL", callStrike, expiryDate, 0.35, 0.23)
            });
        }

        private OptionData CreateOptionData(string optionType, double strike, DateTime expiry, double delta, double iv)
        {
            return new OptionData
            {
                UnderlyingSymbol = _viewModel.UnderlyingSymbol,
                OptionType = optionType,
                StrikePrice = strike,
                ExpirationDate = expiry,
                ImpliedVolatility = iv,
                Delta = delta,
                Gamma = 0.05,
                Theta = -0.15,
                Vega = 0.20,
                Rho = 0.10,
                Bid = 0,
                Ask = 0,
                LastPrice = 0,
                Volume = 0,
                OpenInterest = 0
            };
        }

        private async void LoadSymbolButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var symbol = SymbolTextBox.Text?.Trim().ToUpper();
                if (string.IsNullOrEmpty(symbol))
                {
                    MessageBox.Show("Please enter a symbol.", "Invalid Symbol", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                LoadSymbolButton.IsEnabled = false;
                LoadSymbolButton.Content = "LOADING...";

                // Get current price from Alpha Vantage
                var quoteData = await _alphaVantageService.GetQuoteDataAsync(symbol);
                if (quoteData != null)
                {
                    _viewModel.UnderlyingSymbol = symbol;
                    _viewModel.UnderlyingPrice = quoteData.Price;
                    
                    // Update existing legs with new symbol
                    foreach (var leg in _viewModel.OptionLegs)
                    {
                        leg.Option.UnderlyingSymbol = symbol;
                    }
                    
                    DatabaseMonolith.Log("Info", $"Loaded symbol {symbol} with price {quoteData.Price:C}");
                }
                else
                {
                    MessageBox.Show($"Could not load data for symbol {symbol}.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error loading symbol", ex.ToString());
                MessageBox.Show("Error loading symbol data. Please try again.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                LoadSymbolButton.IsEnabled = true;
                LoadSymbolButton.Content = "LOAD";
            }
        }

        private void AddLegButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var newLeg = new OptionLeg
                {
                    Action = "BUY",
                    Quantity = 1,
                    Price = 0.0,
                    Option = new OptionData
                    {
                        UnderlyingSymbol = _viewModel.UnderlyingSymbol,
                        OptionType = "CALL",
                        StrikePrice = _viewModel.UnderlyingPrice,
                        ExpirationDate = DateTime.Now.AddDays(30),
                        ImpliedVolatility = 0.25,
                        Delta = 0.50,
                        Gamma = 0.05,
                        Theta = -0.15,
                        Vega = 0.20,
                        Rho = 0.10
                    }
                };
                
                _viewModel.OptionLegs.Add(newLeg);
                DatabaseMonolith.Log("Info", "Added new option leg");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error adding new leg", ex.ToString());
            }
        }

        private void CalculateButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                _viewModel.CalculateSpreadMetrics();
                _viewModel.GeneratePayoffChart();
                DatabaseMonolith.Log("Info", "Calculated spread metrics and generated payoff chart");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error calculating spread metrics", ex.ToString());
                MessageBox.Show("Error calculating spread metrics. Please check your inputs.", "Calculation Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }
}