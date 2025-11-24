using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using LiveCharts;
using LiveCharts.Wpf;
using Quantra.Enums;
using Quantra.Models;
using System.Collections.Generic;

namespace Quantra.ViewModels
{
    public class SpreadsExplorerViewModel : INotifyPropertyChanged
    {
        private string _underlyingSymbol;
        private double _underlyingPrice;
        private string _spreadType;
        private MultiLegStrategyType _strategyType;
        private double _netCost;
        private double _maxProfit;
        private double _maxLoss;
        private double _probabilityOfProfit;
        private SeriesCollection _payoffSeries;

        public SpreadsExplorerViewModel()
        {
            OptionLegs = new ObservableCollection<OptionLeg>();
            BreakevenPrices = new ObservableCollection<double>();
            PayoffSeries = new SeriesCollection();

            // Initialize with default values
            UnderlyingSymbol = "AAPL";
            UnderlyingPrice = 150.0;
            SpreadType = "Bull Call Spread";
            StrategyType = MultiLegStrategyType.VerticalSpread;
        }

        #region Properties

        public string UnderlyingSymbol
        {
            get => _underlyingSymbol;
            set
            {
                _underlyingSymbol = value;
                OnPropertyChanged(nameof(UnderlyingSymbol));
            }
        }

        public double UnderlyingPrice
        {
            get => _underlyingPrice;
            set
            {
                _underlyingPrice = value;
                OnPropertyChanged(nameof(UnderlyingPrice));
                OnPropertyChanged(nameof(UnderlyingPriceDisplay));
            }
        }

        public string UnderlyingPriceDisplay => $"Current Price: {UnderlyingPrice:C}";

        public string SpreadType
        {
            get => _spreadType;
            set
            {
                _spreadType = value;
                OnPropertyChanged(nameof(SpreadType));
            }
        }

        public MultiLegStrategyType StrategyType
        {
            get => _strategyType;
            set
            {
                _strategyType = value;
                OnPropertyChanged(nameof(StrategyType));
            }
        }

        public ObservableCollection<OptionLeg> OptionLegs { get; set; }

        public ObservableCollection<double> BreakevenPrices { get; set; }

        public double NetCost
        {
            get => _netCost;
            set
            {
                _netCost = value;
                OnPropertyChanged(nameof(NetCost));
                OnPropertyChanged(nameof(NetCostDisplay));
            }
        }

        public string NetCostDisplay => NetCost >= 0 ? $"{NetCost:C}" : $"({Math.Abs(NetCost):C}) Credit";

        public double MaxProfit
        {
            get => _maxProfit;
            set
            {
                _maxProfit = value;
                OnPropertyChanged(nameof(MaxProfit));
                OnPropertyChanged(nameof(MaxProfitDisplay));
            }
        }

        public string MaxProfitDisplay => MaxProfit == double.PositiveInfinity ? "Unlimited" : $"{MaxProfit:C}";

        public double MaxLoss
        {
            get => _maxLoss;
            set
            {
                _maxLoss = value;
                OnPropertyChanged(nameof(MaxLoss));
                OnPropertyChanged(nameof(MaxLossDisplay));
            }
        }

        public string MaxLossDisplay => MaxLoss == double.NegativeInfinity ? "Unlimited" : $"{Math.Abs(MaxLoss):C}";

        public double ProbabilityOfProfit
        {
            get => _probabilityOfProfit;
            set
            {
                _probabilityOfProfit = value;
                OnPropertyChanged(nameof(ProbabilityOfProfit));
                OnPropertyChanged(nameof(ProbabilityOfProfitDisplay));
            }
        }

        public string ProbabilityOfProfitDisplay => $"{ProbabilityOfProfit:P1}";

        public string BreakevenDisplay => BreakevenPrices.Count == 0 ? "N/A" :
            BreakevenPrices.Count == 1 ? $"{BreakevenPrices[0]:C}" :
            $"{BreakevenPrices.Min():C} - {BreakevenPrices.Max():C}";

        public SeriesCollection PayoffSeries
        {
            get => _payoffSeries;
            set
            {
                _payoffSeries = value;
                OnPropertyChanged(nameof(PayoffSeries));
            }
        }

        public string TotalDelta => CalculateTotalGreek(leg => leg.Option.Delta * leg.Quantity * (leg.Action == "BUY" ? 1 : -1)).ToString("F3");

        public string TotalGamma => CalculateTotalGreek(leg => leg.Option.Gamma * leg.Quantity * (leg.Action == "BUY" ? 1 : -1)).ToString("F3");

        public string TotalTheta => CalculateTotalGreek(leg => leg.Option.Theta * leg.Quantity * (leg.Action == "BUY" ? 1 : -1)).ToString("F3");

        public string TotalVega => CalculateTotalGreek(leg => leg.Option.Vega * leg.Quantity * (leg.Action == "BUY" ? 1 : -1)).ToString("F3");

        public string AverageImpliedVolatility => OptionLegs.Count > 0 ?
            $"{OptionLegs.Average(leg => leg.Option.ImpliedVolatility):P1}" : "N/A";

        #endregion

        #region Methods

        private double CalculateTotalGreek(Func<OptionLeg, double> greekSelector)
        {
            if (OptionLegs == null || OptionLegs.Count == 0)
                return 0.0;

            return OptionLegs.Sum(greekSelector);
        }

        public void CalculateSpreadMetrics()
        {
            try
            {
                if (OptionLegs == null || OptionLegs.Count == 0)
                {
                    ResetMetrics();
                    return;
                }

                // Calculate net cost/credit
                NetCost = OptionLegs.Sum(leg => leg.TotalCost * (leg.Action == "BUY" ? 1 : -1));

                // Calculate breakeven points and max profit/loss based on spread type
                CalculateSpreadSpecificMetrics();

                // Calculate probability of profit (simplified estimation)
                CalculateProbabilityOfProfit();

                // Notify UI of Greek changes
                OnPropertyChanged(nameof(TotalDelta));
                OnPropertyChanged(nameof(TotalGamma));
                OnPropertyChanged(nameof(TotalTheta));
                OnPropertyChanged(nameof(TotalVega));
                OnPropertyChanged(nameof(AverageImpliedVolatility));

                //DatabaseMonolith.Log("Info", $"Calculated metrics for {SpreadType}: NetCost={NetCost:C}, MaxProfit={MaxProfit:C}, MaxLoss={MaxLoss:C}");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error calculating spread metrics", ex.ToString());
                ResetMetrics();
            }
        }

        private void CalculateSpreadSpecificMetrics()
        {
            BreakevenPrices.Clear();

            switch (StrategyType)
            {
                case MultiLegStrategyType.VerticalSpread:
                    CalculateVerticalSpreadMetrics();
                    break;
                case MultiLegStrategyType.Straddle:
                    CalculateStraddleMetrics();
                    break;
                case MultiLegStrategyType.Strangle:
                    CalculateStrangleMetrics();
                    break;
                case MultiLegStrategyType.IronCondor:
                    CalculateIronCondorMetrics();
                    break;
                case MultiLegStrategyType.ButterflySpread:
                    CalculateButterflyMetrics();
                    break;
                case MultiLegStrategyType.CoveredCall:
                    CalculateCoveredCallMetrics();
                    break;
                default:
                    CalculateGenericMetrics();
                    break;
            }
        }

        private void CalculateVerticalSpreadMetrics()
        {
            if (OptionLegs.Count < 2) return;

            var strikes = OptionLegs.Select(leg => leg.Option.StrikePrice).OrderBy(s => s).ToList();
            var isCallSpread = OptionLegs.Any(leg => leg.Option.OptionType == "CALL");

            if (isCallSpread)
            {
                // Bull Call Spread or Bear Call Spread
                MaxLoss = Math.Abs(NetCost);
                MaxProfit = (strikes.Max() - strikes.Min()) * 100 - MaxLoss;
                BreakevenPrices.Add(strikes.Min() + NetCost / 100);
            }
            else
            {
                // Bull Put Spread or Bear Put Spread
                MaxProfit = Math.Abs(NetCost);
                MaxLoss = (strikes.Max() - strikes.Min()) * 100 - MaxProfit;
                BreakevenPrices.Add(strikes.Max() - NetCost / 100);
            }
        }

        private void CalculateStraddleMetrics()
        {
            if (OptionLegs.Count < 2) return;

            var strike = OptionLegs.First().Option.StrikePrice;
            var isLong = OptionLegs.First().Action == "BUY";

            if (isLong)
            {
                // Long Straddle
                MaxLoss = Math.Abs(NetCost);
                MaxProfit = double.PositiveInfinity;
                BreakevenPrices.Add(strike - NetCost / 100);
                BreakevenPrices.Add(strike + NetCost / 100);
            }
            else
            {
                // Short Straddle
                MaxProfit = Math.Abs(NetCost);
                MaxLoss = double.NegativeInfinity;
                BreakevenPrices.Add(strike - NetCost / 100);
                BreakevenPrices.Add(strike + NetCost / 100);
            }
        }

        private void CalculateStrangleMetrics()
        {
            if (OptionLegs.Count < 2) return;

            var strikes = OptionLegs.Select(leg => leg.Option.StrikePrice).OrderBy(s => s).ToList();
            var isLong = OptionLegs.First().Action == "BUY";

            if (isLong)
            {
                // Long Strangle
                MaxLoss = Math.Abs(NetCost);
                MaxProfit = double.PositiveInfinity;
                BreakevenPrices.Add(strikes.Min() - NetCost / 100);
                BreakevenPrices.Add(strikes.Max() + NetCost / 100);
            }
            else
            {
                // Short Strangle
                MaxProfit = Math.Abs(NetCost);
                MaxLoss = double.NegativeInfinity;
                BreakevenPrices.Add(strikes.Min() - NetCost / 100);
                BreakevenPrices.Add(strikes.Max() + NetCost / 100);
            }
        }

        private void CalculateIronCondorMetrics()
        {
            if (OptionLegs.Count < 4) return;

            var strikes = OptionLegs.Select(leg => leg.Option.StrikePrice).OrderBy(s => s).ToList();
            MaxProfit = Math.Abs(NetCost);
            MaxLoss = (strikes[2] - strikes[1]) * 100 - MaxProfit;
            BreakevenPrices.Add(strikes[1] + NetCost / 100);
            BreakevenPrices.Add(strikes[2] - NetCost / 100);
        }

        private void CalculateButterflyMetrics()
        {
            if (OptionLegs.Count < 3) return;

            var strikes = OptionLegs.Select(leg => leg.Option.StrikePrice).OrderBy(s => s).ToList();
            MaxLoss = Math.Abs(NetCost);
            MaxProfit = (strikes.Max() - strikes.Min()) / 2 * 100 - MaxLoss;

            var middleStrike = strikes[1];
            BreakevenPrices.Add(middleStrike - NetCost / 100);
            BreakevenPrices.Add(middleStrike + NetCost / 100);
        }

        private void CalculateCoveredCallMetrics()
        {
            if (OptionLegs.Count < 1) return;

            var callLeg = OptionLegs.FirstOrDefault(leg => leg.Option.OptionType == "CALL");
            if (callLeg != null)
            {
                MaxProfit = Math.Abs(NetCost); // Premium collected
                MaxLoss = double.NegativeInfinity; // Unlimited if stock goes to zero
                BreakevenPrices.Add(UnderlyingPrice + NetCost / 100);
            }
        }

        private void CalculateGenericMetrics()
        {
            // For custom spreads, calculate basic metrics
            MaxLoss = Math.Abs(NetCost);
            MaxProfit = double.PositiveInfinity;

            // Simple breakeven estimation
            var avgStrike = OptionLegs.Average(leg => leg.Option.StrikePrice);
            BreakevenPrices.Add(avgStrike);
        }

        private void CalculateProbabilityOfProfit()
        {
            try
            {
                if (BreakevenPrices.Count == 0)
                {
                    ProbabilityOfProfit = 0.5; // Default 50% if no breakevens calculated
                    return;
                }

                // Simplified POP calculation based on distance from current price to breakevens
                var avgIV = OptionLegs.Average(leg => leg.Option.ImpliedVolatility);
                var daysToExpiry = OptionLegs.Average(leg => (leg.Option.ExpirationDate - DateTime.Now).TotalDays);

                if (daysToExpiry <= 0)
                {
                    ProbabilityOfProfit = 0.0;
                    return;
                }

                // Calculate expected move
                var expectedMove = UnderlyingPrice * avgIV * Math.Sqrt(daysToExpiry / 365.0);

                // Simplified probability calculation
                if (BreakevenPrices.Count == 1)
                {
                    var distance = Math.Abs(BreakevenPrices[0] - UnderlyingPrice);
                    ProbabilityOfProfit = Math.Max(0.1, Math.Min(0.9, 1.0 - (distance / expectedMove) * 0.4));
                }
                else if (BreakevenPrices.Count >= 2)
                {
                    var lowerBreakeven = BreakevenPrices.Min();
                    var upperBreakeven = BreakevenPrices.Max();
                    var profitableRange = upperBreakeven - lowerBreakeven;
                    ProbabilityOfProfit = Math.Max(0.1, Math.Min(0.9, profitableRange / (expectedMove * 2)));
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", "Error calculating probability of profit", ex.ToString());
                ProbabilityOfProfit = 0.5; // Default fallback
            }
        }

        public void GeneratePayoffChart()
        {
            try
            {
                if (OptionLegs == null || OptionLegs.Count == 0)
                {
                    PayoffSeries.Clear();
                    return;
                }

                var chartValues = new ChartValues<double>();
                var priceRange = GeneratePriceRange();

                foreach (var price in priceRange)
                {
                    var totalPnL = CalculatePnLAtPrice(price);
                    chartValues.Add(totalPnL);
                }

                PayoffSeries.Clear();
                PayoffSeries.Add(new LineSeries
                {
                    Title = $"{SpreadType} P&L",
                    Values = chartValues,
                    Stroke = System.Windows.Media.Brushes.Cyan,
                    StrokeThickness = 2,
                    Fill = System.Windows.Media.Brushes.Transparent,
                    PointGeometry = null
                });

                // Add breakeven lines
                foreach (var breakeven in BreakevenPrices)
                {
                    var breakevenIndex = priceRange.ToList().IndexOf(priceRange.OrderBy(p => Math.Abs(p - breakeven)).First());
                    if (breakevenIndex >= 0)
                    {
                        var breakevenValues = new ChartValues<double>();
                        for (int i = 0; i < priceRange.Count(); i++)
                        {
                            breakevenValues.Add(i == breakevenIndex ? 0 : double.NaN);
                        }

                        PayoffSeries.Add(new LineSeries
                        {
                            Title = $"Breakeven ${breakeven:F2}",
                            Values = breakevenValues,
                            Stroke = System.Windows.Media.Brushes.Yellow,
                            StrokeThickness = 1,
                            Fill = System.Windows.Media.Brushes.Transparent,
                            PointGeometry = DefaultGeometries.Circle,
                            PointGeometrySize = 8
                        });
                    }
                }

                //DatabaseMonolith.Log("Info", $"Generated payoff chart with {chartValues.Count} data points");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error generating payoff chart", ex.ToString());
                PayoffSeries.Clear();
            }
        }

        private double[] GeneratePriceRange()
        {
            var strikes = OptionLegs.Select(leg => leg.Option.StrikePrice).ToList();
            var minStrike = strikes.Min();
            var maxStrike = strikes.Max();

            var rangeStart = Math.Max(0, Math.Min(minStrike * 0.8, UnderlyingPrice * 0.8));
            var rangeEnd = Math.Max(maxStrike * 1.2, UnderlyingPrice * 1.2);

            var step = (rangeEnd - rangeStart) / 100.0;

            return Enumerable.Range(0, 101)
                .Select(i => rangeStart + (i * step))
                .ToArray();
        }

        private double CalculatePnLAtPrice(double price)
        {
            double totalPnL = 0;

            foreach (var leg in OptionLegs)
            {
                var intrinsicValue = CalculateIntrinsicValue(leg.Option, price);
                var legPnL = (intrinsicValue - leg.Price) * leg.Quantity * 100;

                if (leg.Action == "SELL")
                    legPnL = -legPnL;

                totalPnL += legPnL;
            }

            return totalPnL;
        }

        private double CalculateIntrinsicValue(OptionData option, double underlyingPrice)
        {
            if (option.OptionType == "CALL")
            {
                return Math.Max(0, underlyingPrice - option.StrikePrice);
            }
            else // PUT
            {
                return Math.Max(0, option.StrikePrice - underlyingPrice);
            }
        }

        private void ResetMetrics()
        {
            NetCost = 0;
            MaxProfit = 0;
            MaxLoss = 0;
            ProbabilityOfProfit = 0;
            BreakevenPrices.Clear();
        }

        #endregion

        #region Spread Strategy Definitions

        /// <summary>
        /// Gets the definition/description for a given spread strategy type
        /// </summary>
        /// <param name="spreadType">The spread strategy name</param>
        /// <returns>Description of the spread strategy</returns>
        public static string GetSpreadStrategyDefinition(string spreadType)
        {
            var definitions = new Dictionary<string, string>
            {
                ["Bull Call Spread"] = "A bullish strategy using two call options with different strike prices. " +
                                     "Buy a lower strike call and sell a higher strike call. " +
                                     "Limited profit potential with limited risk.",

                ["Bear Put Spread"] = "A bearish strategy using two put options with different strike prices. " +
                                    "Buy a higher strike put and sell a lower strike put. " +
                                    "Limited profit potential with limited risk.",

                ["Long Straddle"] = "A volatility strategy involving buying both a call and put at the same strike price. " +
                                  "Profits from large price movements in either direction. " +
                                  "Unlimited profit potential with limited risk.",

                ["Short Straddle"] = "A low volatility strategy involving selling both a call and put at the same strike price. " +
                                   "Profits when the stock price remains near the strike price. " +
                                   "Limited profit with unlimited risk.",

                ["Long Strangle"] = "A volatility strategy using calls and puts at different strike prices. " +
                                  "Buy an out-of-the-money call and an out-of-the-money put. " +
                                  "Profits from large price movements with lower cost than straddle.",

                ["Short Strangle"] = "A low volatility strategy selling calls and puts at different strike prices. " +
                                    "Sell an out-of-the-money call and an out-of-the-money put. " +
                                    "Profits when stock price stays between strike prices.",

                ["Iron Condor"] = "A neutral strategy combining a bull put spread and bear call spread. " +
                                "Profits when the stock price remains within a specific range. " +
                                "Limited profit and limited risk with four option legs.",

                ["Butterfly Spread"] = "A neutral strategy using three strike prices in a 1:2:1 ratio. " +
                                     "Profits when the stock price stays near the middle strike price. " +
                                     "Limited profit and limited risk with low cost.",

                ["Calendar Spread"] = "A time decay strategy using options with the same strike but different expirations. " +
                                    "Sell a near-term option and buy a longer-term option. " +
                                    "Profits from time decay and volatility differences.",

                ["Covered Call"] = "A conservative income strategy involving owning stock and selling call options. " +
                                 "Generates additional income from option premiums. " +
                                 "Limited upside potential but provides downside protection."
            };

            return definitions.TryGetValue(spreadType, out var definition)
                ? definition
                : "Custom spread strategy with user-defined option legs.";
        }

        #endregion

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}