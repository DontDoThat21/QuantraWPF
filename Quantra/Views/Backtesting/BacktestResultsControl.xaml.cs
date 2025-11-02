using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Services;

namespace Quantra.Views.Backtesting
{
    public partial class BacktestResultsControl : UserControl
    {
        private BacktestingEngine.BacktestResult _currentResult;
        private List<Models.HistoricalPrice> _historicalData;
        private List<BenchmarkComparisonData> _benchmarkData = new List<BenchmarkComparisonData>();
        private double _strategyEquityVolatility; // Store calculated equity volatility
        private bool _showRelativeReturns = false; // Track if showing relative or absolute returns
        private readonly Dictionary<string, Brush> _benchmarkColors = new Dictionary<string, Brush>
        {
            { "SPY", Brushes.DarkGreen },
            { "QQQ", Brushes.DarkBlue },
            { "IWM", Brushes.DarkOrange },
            { "DIA", Brushes.Purple },
            // Default color for custom benchmarks
            { "CUSTOM", Brushes.Magenta }
        };
        
        
        private readonly HistoricalDataService _historicalDataService;
        private readonly CustomBenchmarkService _customBenchmarkService;
        private ObservableCollection<CustomBenchmark> _customBenchmarks;
        
        public BacktestResultsControl(HistoricalDataService historicalDataService,
            CustomBenchmarkService customBenchmarkService
            )
        {
            InitializeComponent();
            _historicalDataService = historicalDataService;
            _customBenchmarkService = customBenchmarkService;
            _customBenchmarks = new ObservableCollection<CustomBenchmark>();
            
            // Initialize UI elements if needed
            LoadCustomBenchmarks();
            
            // Apply active benchmark selection from user settings
            ApplyActiveBenchmarkSelection();
            
            // Add debugging event handler
            CustomBenchmarkComboBox.SelectionChanged += (s, e) => {
                if (CustomBenchmarkComboBox.SelectedItem != null)
                {
                    //DatabaseMonolith.Log("Debug", $"CustomBenchmarkComboBox Selection: {CustomBenchmarkComboBox.SelectedItem.GetType().Name} - {CustomBenchmarkComboBox.SelectedItem}");
                }
            };
        }

        private void BacktestResultsControl_Loaded(object sender, RoutedEventArgs e)
        {
            // Initialize UI elements if needed
        }

        public async void LoadResults(BacktestingEngine.BacktestResult result, List<Models.HistoricalPrice> historical)
        {
            _currentResult = result;
            _historicalData = historical;
            
            // Calculate equity volatility for later use
            CalculateStrategyVolatility();
            
            // Update metrics display
            TotalReturnText.Text = result.TotalReturn.ToString("P2");
            MaxDrawdownText.Text = result.MaxDrawdown.ToString("P2");
            WinRateText.Text = result.WinRate.ToString("P2");
            CAGRText.Text = result.CAGR.ToString("P2");
            
            // Advanced metrics
            SharpeRatioText.Text = result.SharpeRatio.ToString("F2");
            SortinoRatioText.Text = result.SortinoRatio.ToString("F2");
            CalmarRatioText.Text = result.CalmarRatio.ToString("F2");
            ProfitFactorText.Text = result.ProfitFactor.ToString("F2");
            InformationRatioText.Text = result.InformationRatio.ToString("F2");
            
            // Price chart with trades
            var priceSeries = new LineSeries
            {
                Title = "Price",
                Values = new ChartValues<double>(historical.Select(h => h.Close)),
                Stroke = Brushes.DodgerBlue,
                Fill = Brushes.Transparent,
                PointGeometry = null
            };
            var buyPoints = result.Trades.Where(t => t.Action == "BUY").Select(t => new { t.EntryDate, t.EntryPrice }).ToList();
            var sellPoints = result.Trades.Where(t => t.Action == "SELL").Select(t => new { t.EntryDate, t.EntryPrice }).ToList();
            var buyMarkers = new ScatterSeries
            {
                Title = "Buy",
                Values = new ChartValues<ObservablePoint>(
                    buyPoints.Select(b => new ObservablePoint(
                        historical.FindIndex(h => h.Date == b.EntryDate), b.EntryPrice)
                    )
                ),
                Fill = Brushes.LimeGreen,
                MinPointShapeDiameter = 10,
                MaxPointShapeDiameter = 10
            };
            var sellMarkers = new ScatterSeries
            {
                Title = "Sell",
                Values = new ChartValues<ObservablePoint>(
                    sellPoints.Select(s => new ObservablePoint(
                        historical.FindIndex(h => h.Date == s.EntryDate), s.EntryPrice)
                    )
                ),
                Fill = Brushes.Red,
                MinPointShapeDiameter = 10,
                MaxPointShapeDiameter = 10
            };
            PriceChart.Series = new SeriesCollection { priceSeries, buyMarkers, sellMarkers };
            PriceChart.AxisX.Clear();
            PriceChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Date",
                Labels = historical.Select(h => h.Date.ToShortDateString()).ToArray(),
                LabelsRotation = 45,
                Separator = new LiveCharts.Wpf.Separator { Step = Math.Max(1, historical.Count / 10) }
            });
            PriceChart.AxisY.Clear();
            PriceChart.AxisY.Add(new LiveCharts.Wpf.Axis { Title = "Price" });

            // Equity curve
            var equitySeries = new LineSeries
            {
                Title = "Equity",
                Values = new ChartValues<double>(result.EquityCurve.Select(e => e.Equity)),
                Stroke = Brushes.DarkOrange,
                Fill = Brushes.Transparent,
                PointGeometry = null
            };
            EquityChart.Series = new SeriesCollection { equitySeries };
            EquityChart.AxisX.Clear();
            EquityChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Date",
                Labels = result.EquityCurve.Select(e => e.Date.ToShortDateString()).ToArray(),
                LabelsRotation = 45,
                Separator = new LiveCharts.Wpf.Separator { Step = Math.Max(1, result.EquityCurve.Count / 10) }
            });
            EquityChart.AxisY.Clear();
            EquityChart.AxisY.Add(new LiveCharts.Wpf.Axis { Title = "Equity" });

            // Drawdown curve
            var ddSeries = new LineSeries
            {
                Title = "Drawdown",
                Values = new ChartValues<double>(result.DrawdownCurve.Select(d => d.Drawdown)),
                Stroke = Brushes.Crimson,
                Fill = Brushes.LightPink,
                PointGeometry = null
            };
            DrawdownChart.Series = new SeriesCollection { ddSeries };
            DrawdownChart.AxisX.Clear();
            DrawdownChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Date",
                Labels = result.DrawdownCurve.Select(d => d.Date.ToShortDateString()).ToArray(),
                LabelsRotation = 45,
                Separator = new LiveCharts.Wpf.Separator { Step = Math.Max(1, result.DrawdownCurve.Count / 10) }
            });
            DrawdownChart.AxisY.Clear();
            DrawdownChart.AxisY.Add(new LiveCharts.Wpf.Axis { 
                Title = "Drawdown", 
                LabelFormatter = v => (v * 100).ToString("F1") + "%" 
            });
            
            // Setup Monte Carlo UI
            MonteCarloStatusText.Text = "Ready for simulation";
            
            // Update Monte Carlo UI if results are available
            if (result.HasMonteCarloResults)
            {
                UpdateMonteCarloVisualization();
            }
            
            // Load benchmark data
            await LoadBenchmarkData();
        }
        
        private async Task LoadBenchmarkData()
        {
            try
            {
                _benchmarkData.Clear();
                
                // Get all selected benchmark symbols
                var selectedBenchmarks = GetSelectedBenchmarks();
                
                if (selectedBenchmarks.Count == 0 || _currentResult == null || _historicalData == null || _historicalData.Count == 0)
                {
                    return;
                }
                
                // Get the date range from the backtest
                DateTime startDate = _currentResult.StartDate;
                DateTime endDate = _currentResult.EndDate;
                
                foreach (var benchmarkInfo in selectedBenchmarks)
                {
                    // Check if this is a custom benchmark
                    var customBenchmark = _customBenchmarks.FirstOrDefault(b => b.DisplaySymbol == benchmarkInfo.symbol);
                    
                    BenchmarkComparisonData benchmarkData;
                    
                    if (customBenchmark != null)
                    {
                        // Load custom benchmark data using the service
                        benchmarkData = await _customBenchmarkService.CalculateCustomBenchmarkData(
                            customBenchmark, startDate, endDate);
                    }
                    else
                    {
                        // Load standard benchmark data from historical service
                        benchmarkData = await LoadBenchmarkHistoricalData(benchmarkInfo.symbol, benchmarkInfo.name, startDate, endDate);
                    }
                    
                    if (benchmarkData != null)
                    {
                        _benchmarkData.Add(benchmarkData);
                    }
                }
                
                // Update benchmark comparison charts
                UpdateBenchmarkComparisonCharts();
                
            }
            catch (Exception ex)
            {
                // Log error
                //DatabaseMonolith.Log("Error", "Failed to load benchmark data", ex.ToString());
            }
        }
        
        private List<(string symbol, string name)> GetSelectedBenchmarks()
        {
            var benchmarks = new List<(string symbol, string name)>();
            var (activeBenchmarkType, activeBenchmarkId) = UserSettingsService.GetActiveBenchmark();
            
            // Add active benchmark first (this will be the primary baseline)
            bool activeBenchmarkAdded = false;
            
            if (activeBenchmarkType == "CUSTOM" && !string.IsNullOrEmpty(activeBenchmarkId))
            {
                var activeCustomBenchmark = _customBenchmarks.FirstOrDefault(b => b.Id == activeBenchmarkId);
                if (activeCustomBenchmark != null)
                {
                    benchmarks.Add((activeCustomBenchmark.DisplaySymbol, activeCustomBenchmark.Name));
                    activeBenchmarkAdded = true;
                }
            }
            else
            {
                switch (activeBenchmarkType)
                {
                    case "SPY":
                        if (SPYCheckBox.IsChecked == true)
                        {
                            benchmarks.Add(("SPY", "S&P 500"));
                            activeBenchmarkAdded = true;
                        }
                        break;
                    case "QQQ":
                        if (QQQCheckBox.IsChecked == true)
                        {
                            benchmarks.Add(("QQQ", "NASDAQ"));
                            activeBenchmarkAdded = true;
                        }
                        break;
                    case "IWM":
                        if (IWMCheckBox.IsChecked == true)
                        {
                            benchmarks.Add(("IWM", "Russell 2000"));
                            activeBenchmarkAdded = true;
                        }
                        break;
                    case "DIA":
                        if (DIACheckBox.IsChecked == true)
                        {
                            benchmarks.Add(("DIA", "Dow Jones"));
                            activeBenchmarkAdded = true;
                        }
                        break;
                }
            }
            
            // Add other selected benchmarks (excluding the active one if already added)
            if (SPYCheckBox.IsChecked == true && (activeBenchmarkType != "SPY" || !activeBenchmarkAdded))
                benchmarks.Add(("SPY", "S&P 500"));
                
            if (QQQCheckBox.IsChecked == true && (activeBenchmarkType != "QQQ" || !activeBenchmarkAdded))
                benchmarks.Add(("QQQ", "NASDAQ"));
                
            if (IWMCheckBox.IsChecked == true && (activeBenchmarkType != "IWM" || !activeBenchmarkAdded))
                benchmarks.Add(("IWM", "Russell 2000"));
                
            if (DIACheckBox.IsChecked == true && (activeBenchmarkType != "DIA" || !activeBenchmarkAdded))
                benchmarks.Add(("DIA", "Dow Jones"));

            // Add any selected custom benchmark (excluding the active one if already added)
            var selectedCustomBenchmark = CustomBenchmarkComboBox.SelectedItem as CustomBenchmark;
            if (selectedCustomBenchmark != null && 
                (activeBenchmarkType != "CUSTOM" || activeBenchmarkId != selectedCustomBenchmark.Id || !activeBenchmarkAdded))
            {
                benchmarks.Add((selectedCustomBenchmark.DisplaySymbol, selectedCustomBenchmark.Name));
            }
                
            return benchmarks;
        }
        
        private async Task<BenchmarkComparisonData> LoadBenchmarkHistoricalData(string symbol, string name, DateTime startDate, DateTime endDate)
        {
            try
            {
                // Load historical data for the benchmark
                var historicalData = await _historicalDataService.GetComprehensiveHistoricalData(symbol);
                
                // Filter to match the date range of the backtest
                var filteredData = historicalData
                    .Where(h => h.Date >= startDate && h.Date <= endDate)
                    .OrderBy(h => h.Date)
                    .ToList();
                
                if (filteredData.Count == 0)
                {
                    //DatabaseMonolith.Log("Warning", $"No benchmark data for {symbol} in date range {startDate} to {endDate}");
                    return null;
                }
                
                // Create benchmark comparison data
                var benchmarkData = new BenchmarkComparisonData
                {
                    Symbol = symbol,
                    Name = name,
                    HistoricalData = filteredData,
                    Dates = filteredData.Select(h => h.Date).ToList()
                };
                
                // Calculate normalized returns (starting at 1.0)
                double initialPrice = filteredData.First().Close;
                benchmarkData.NormalizedReturns = filteredData
                    .Select(h => h.Close / initialPrice)
                    .ToList();
                
                // Calculate performance metrics
                benchmarkData.TotalReturn = (filteredData.Last().Close / filteredData.First().Close) - 1;
                
                // Calculate drawdown and other metrics
                double peak = filteredData.First().Close;
                double maxDrawdown = 0;
                List<double> dailyReturns = new List<double>();
                List<double> downsideReturns = new List<double>();
                
                for (int i = 1; i < filteredData.Count; i++)
                {
                    if (filteredData[i].Close > peak)
                    {
                        peak = filteredData[i].Close;
                    }
                    
                    double drawdown = (peak - filteredData[i].Close) / peak;
                    if (drawdown > maxDrawdown)
                    {
                        maxDrawdown = drawdown;
                    }
                    
                    // Calculate daily returns
                    double dailyReturn = (filteredData[i].Close - filteredData[i - 1].Close) / filteredData[i - 1].Close;
                    dailyReturns.Add(dailyReturn);
                    
                    if (dailyReturn < 0)
                    {
                        downsideReturns.Add(dailyReturn);
                    }
                }
                
                benchmarkData.MaxDrawdown = maxDrawdown;
                benchmarkData.Volatility = CalculateStandardDeviation(dailyReturns);
                
                // Calculate CAGR
                double totalDays = (endDate - startDate).TotalDays;
                if (totalDays > 0 && filteredData.Count > 0)
                {
                    benchmarkData.TotalReturn = (filteredData.Last().Close / filteredData.First().Close) - 1;
                    benchmarkData.CAGR = Math.Pow(1 + benchmarkData.TotalReturn, 365.0 / totalDays) - 1;
                }
                
                // Calculate Sharpe, Sortino, and Calmar ratios
                double averageReturn = dailyReturns.Count > 0 ? dailyReturns.Average() : 0;
                double riskFreeRate = 0.0; // Simplified assumption
                
                benchmarkData.SharpeRatio = benchmarkData.Volatility > 0 ? 
                    (averageReturn - riskFreeRate) / benchmarkData.Volatility * Math.Sqrt(252) : 0;
                
                double downsideDeviation = CalculateStandardDeviation(downsideReturns);
                benchmarkData.SortinoRatio = downsideDeviation > 0 ? 
                    (averageReturn - riskFreeRate) / downsideDeviation * Math.Sqrt(252) : 0;
                
                benchmarkData.CalmarRatio = benchmarkData.MaxDrawdown > 0 ? 
                    benchmarkData.CAGR / benchmarkData.MaxDrawdown : 0;
                    
                benchmarkData.InformationRatio = benchmarkData.Volatility > 0 ?
                    (averageReturn - riskFreeRate) / benchmarkData.Volatility * Math.Sqrt(252) : 0;
                
                // Calculate Beta and Alpha if strategy data exists
                if (_currentResult != null && _currentResult.EquityCurve.Count > 0)
                {
                    // Align the dates between strategy and benchmark
                    var alignedReturns = AlignReturnsForComparison(_currentResult, filteredData);
                    if (alignedReturns.strategyReturns.Count > 0 && alignedReturns.benchmarkReturns.Count > 0)
                    {
                        benchmarkData.Beta = CalculateBeta(alignedReturns.strategyReturns, alignedReturns.benchmarkReturns);
                        benchmarkData.Alpha = CalculateAlpha(alignedReturns.strategyReturns, alignedReturns.benchmarkReturns, benchmarkData.Beta, riskFreeRate);
                        benchmarkData.Correlation = CalculateCorrelation(alignedReturns.strategyReturns, alignedReturns.benchmarkReturns);
                    }
                }
                
                return benchmarkData;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to load benchmark data for {symbol}", ex.ToString());
                return null;
            }
        }
        
        private (List<double> strategyReturns, List<double> benchmarkReturns) AlignReturnsForComparison(
            BacktestingEngine.BacktestResult strategyResult, 
            List<Models.HistoricalPrice> benchmarkData)
        {
            List<double> strategyReturns = new List<double>();
            List<double> benchmarkReturns = new List<double>();
            
            // Create dictionary of benchmark data for quick lookup
            var benchmarkByDate = benchmarkData.ToDictionary(h => h.Date.Date, h => h);
            
            // For each day in the strategy result, find the corresponding benchmark data
            for (int i = 1; i < strategyResult.EquityCurve.Count; i++)
            {
                DateTime currentDate = strategyResult.EquityCurve[i].Date.Date;
                DateTime previousDate = strategyResult.EquityCurve[i - 1].Date.Date;
                
                if (benchmarkByDate.ContainsKey(currentDate) && benchmarkByDate.ContainsKey(previousDate))
                {
                    double strategyReturn = (strategyResult.EquityCurve[i].Equity - strategyResult.EquityCurve[i - 1].Equity) / 
                                          strategyResult.EquityCurve[i - 1].Equity;
                    
                    double benchmarkReturn = (benchmarkByDate[currentDate].Close - benchmarkByDate[previousDate].Close) / 
                                          benchmarkByDate[previousDate].Close;
                    
                    strategyReturns.Add(strategyReturn);
                    benchmarkReturns.Add(benchmarkReturn);
                }
            }
            
            return (strategyReturns, benchmarkReturns);
        }
        
        private double CalculateStandardDeviation(List<double> values)
        {
            if (values == null || values.Count <= 1)
                return 0;
                
            double avg = values.Average();
            double sumOfSquaresOfDifferences = values.Sum(val => Math.Pow(val - avg, 2));
            return Math.Sqrt(sumOfSquaresOfDifferences / (values.Count - 1));
        }
        
        private double CalculateBeta(List<double> strategyReturns, List<double> benchmarkReturns)
        {
            if (strategyReturns.Count != benchmarkReturns.Count || strategyReturns.Count < 2)
                return 1;
                
            double covariance = CalculateCovariance(strategyReturns, benchmarkReturns);
            double benchmarkVariance = CalculateVariance(benchmarkReturns);
            
            return benchmarkVariance != 0 ? covariance / benchmarkVariance : 1;
        }
        
        private double CalculateAlpha(List<double> strategyReturns, List<double> benchmarkReturns, double beta, double riskFreeRate)
        {
            if (strategyReturns.Count != benchmarkReturns.Count || strategyReturns.Count < 2)
                return 0;
                
            double avgStrategyReturn = strategyReturns.Average();
            double avgBenchmarkReturn = benchmarkReturns.Average();
            
            // Annualizing the alpha (assuming daily returns)
            double annualFactor = 252; // Trading days in a year
            return (avgStrategyReturn - riskFreeRate) - beta * (avgBenchmarkReturn - riskFreeRate) * annualFactor;
        }
        
        private double CalculateCovariance(List<double> x, List<double> y)
        {
            if (x.Count != y.Count || x.Count < 2)
                return 0;
                
            double xMean = x.Average();
            double yMean = y.Average();
            double sum = 0;
            
            for (int i = 0; i < x.Count; i++)
            {
                sum += (x[i] - xMean) * (y[i] - yMean);
            }
            
            return sum / (x.Count - 1);
        }
        
        private double CalculateVariance(List<double> values)
        {
            if (values.Count < 2)
                return 0;
                
            double mean = values.Average();
            double sum = 0;
            
            foreach (double val in values)
            {
                sum += Math.Pow(val - mean, 2);
            }
            
            return sum / (values.Count - 1);
        }
        
        private double CalculateCorrelation(List<double> x, List<double> y)
        {
            if (x.Count != y.Count || x.Count < 2)
                return 0;
                
            double xStdDev = CalculateStandardDeviation(x);
            double yStdDev = CalculateStandardDeviation(y);
            double covariance = CalculateCovariance(x, y);
            
            return xStdDev > 0 && yStdDev > 0 ? covariance / (xStdDev * yStdDev) : 0;
        }
        
        private void CalculateStrategyVolatility()
        {
            if (_currentResult == null || _currentResult.EquityCurve == null || _currentResult.EquityCurve.Count <= 1)
            {
                _strategyEquityVolatility = 0;
                return;
            }
            
            // Convert equity curve to daily returns
            var equityValues = _currentResult.EquityCurve.Select(e => e.Equity).ToList();
            var dailyReturns = new List<double>();
            
            for (int i = 1; i < equityValues.Count; i++)
            {
                double dailyReturn = (equityValues[i] - equityValues[i - 1]) / equityValues[i - 1];
                dailyReturns.Add(dailyReturn);
            }
            
            // Calculate volatility as standard deviation of daily returns
            _strategyEquityVolatility = CalculateStandardDeviation(dailyReturns);
        }
        
        private void UpdateBenchmarkComparisonCharts()
        {
            if (_benchmarkData == null || _benchmarkData.Count == 0 || _currentResult == null)
            {
                return;
            }
            
            // 1. Update Combined Performance Summary
            UpdateCombinedPerformanceChart();
            
            // 2. Update Cumulative Returns Chart
            UpdateBenchmarkReturnsChart();
            
            // 3. Update Drawdown Comparison Chart
            UpdateBenchmarkDrawdownChart();
            
            // 4. Update Volatility Comparison Chart
            UpdateVolatilityComparisonChart();
            
            // 5. Update Risk Metrics Chart and Grid
            UpdateRiskMetricsComparison();
            
            // 6. Update Performance Attribution Chart
            UpdatePerformanceAttributionChart();
            
            // Set default tab selections
            TimeRangeSelector.SelectedIndex = 0; // "All Data"
            ShowRelativeReturnsToggle.IsChecked = false;
        }
        
        private void UpdateCombinedPerformanceChart()
        {
            // Create combined performance chart showing returns and key events
            CombinedPerformanceChart.Series = new SeriesCollection();
            
            // Create strategy returns series
            double initialEquity = _currentResult.EquityCurve.First().Equity;
            var strategyValues = _currentResult.EquityCurve
                .Select(e => e.Equity / initialEquity)
                .ToList();
                
            var strategySeries = new LineSeries
            {
                Title = "Strategy",
                Values = new ChartValues<double>(strategyValues),
                Stroke = Brushes.DarkRed,
                Fill = Brushes.Transparent,
                StrokeThickness = 3,
                PointGeometry = DefaultGeometries.Diamond,
                PointGeometrySize = 0,
                DataLabels = false,
                LabelPoint = point => $"Strategy: {point.Y.ToString("F2")}x ({((point.Y - 1) * 100).ToString("F2")}%)"
            };
            
            CombinedPerformanceChart.Series.Add(strategySeries);
            
            // Only add the first benchmark for clarity in the summary view
            if (_benchmarkData.Count > 0)
            {
                var benchmark = _benchmarkData[0]; // Use first benchmark
                
                var benchmarkSeries = new LineSeries
                {
                    Title = benchmark.Name,
                    Values = new ChartValues<double>(benchmark.NormalizedReturns),
                    Stroke = GetBenchmarkColor(benchmark.Symbol),
                    Fill = Brushes.Transparent,
                    PointGeometry = DefaultGeometries.Circle,
                    PointGeometrySize = 0,
                    DataLabels = false,
                    LabelPoint = point => $"{benchmark.Name}: {point.Y.ToString("F2")}x ({((point.Y - 1) * 100).ToString("F2")}%)"
                };
                
                CombinedPerformanceChart.Series.Add(benchmarkSeries);
                
                // Create performance comparison metrics
                UpdatePerformanceComparisonGrid(benchmark);
            }
            
            // Add point markers for significant events (big gains/losses)
            var significantTrades = _currentResult.Trades
                .Where(t => Math.Abs(t.ProfitLoss / initialEquity) > 0.02) // Over 2% impact on equity
                .Take(5); // Limit to 5 most significant
            
            foreach (var trade in significantTrades)
            {
                int tradeIndex = _currentResult.EquityCurve.FindIndex(e => e.Date >= trade.ExitDate);
                if (tradeIndex >= 0)
                {
                    // Add marker for significant trade
                    var marker = new ScatterSeries
                    {
                        Title = trade.ProfitLoss > 0 ? "Significant Gain" : "Significant Loss",
                        Values = new ChartValues<ScatterPoint> {
                            new ScatterPoint(tradeIndex, strategyValues[tradeIndex])
                        },
                        Fill = trade.ProfitLoss > 0 ? Brushes.Green : Brushes.Red,
                        MinPointShapeDiameter = 10,
                        MaxPointShapeDiameter = 10
                    };
                    
                    CombinedPerformanceChart.Series.Add(marker);
                }
            }
            
            // Setup axes
            CombinedPerformanceChart.AxisX.Clear();
            CombinedPerformanceChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Date",
                Labels = _currentResult.EquityCurve.Select(e => e.Date.ToShortDateString()).ToArray(),
                LabelsRotation = 45,
                Separator = new LiveCharts.Wpf.Separator { Step = Math.Max(1, _currentResult.EquityCurve.Count / 10) }
            });
            
            CombinedPerformanceChart.AxisY.Clear();
            CombinedPerformanceChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Normalized Return",
                LabelFormatter = value => value.ToString("F2") + "x"
            });
            
            // Configure chart for better interactivity
            CombinedPerformanceChart.LegendLocation = LegendLocation.Top;
            CombinedPerformanceChart.Zoom = ZoomingOptions.X;
            CombinedPerformanceChart.DisableAnimations = true;
        }
        
        private void UpdatePerformanceComparisonGrid(BenchmarkComparisonData benchmark)
        {
            // Create comparison metrics between strategy and benchmark
            var comparisonMetrics = new List<PerformanceComparisonMetric>();
            
            // Total Return
            comparisonMetrics.Add(new PerformanceComparisonMetric
            {
                Metric = "Total Return",
                Strategy = _currentResult.TotalReturn.ToString("P2"),
                Benchmark = benchmark.TotalReturn.ToString("P2"),
                Difference = (_currentResult.TotalReturn - benchmark.TotalReturn).ToString("P2"),
                IsOutperforming = _currentResult.TotalReturn > benchmark.TotalReturn,
                NumericDifference = _currentResult.TotalReturn - benchmark.TotalReturn
            });
            
            // CAGR
            comparisonMetrics.Add(new PerformanceComparisonMetric
            {
                Metric = "CAGR",
                Strategy = _currentResult.CAGR.ToString("P2"),
                Benchmark = benchmark.CAGR.ToString("P2"),
                Difference = (_currentResult.CAGR - benchmark.CAGR).ToString("P2"),
                IsOutperforming = _currentResult.CAGR > benchmark.CAGR,
                NumericDifference = _currentResult.CAGR - benchmark.CAGR
            });
            
            // Max Drawdown
            comparisonMetrics.Add(new PerformanceComparisonMetric
            {
                Metric = "Max Drawdown",
                Strategy = _currentResult.MaxDrawdown.ToString("P2"),
                Benchmark = benchmark.MaxDrawdown.ToString("P2"),
                Difference = (benchmark.MaxDrawdown - _currentResult.MaxDrawdown).ToString("P2"),
                IsOutperforming = _currentResult.MaxDrawdown < benchmark.MaxDrawdown,
                NumericDifference = benchmark.MaxDrawdown - _currentResult.MaxDrawdown
            });
            
            // Volatility
            comparisonMetrics.Add(new PerformanceComparisonMetric
            {
                Metric = "Volatility (Ann.)",
                Strategy = (_strategyEquityVolatility * Math.Sqrt(252)).ToString("P2"),
                Benchmark = (benchmark.Volatility * Math.Sqrt(252)).ToString("P2"),
                Difference = ((benchmark.Volatility - _strategyEquityVolatility) * Math.Sqrt(252)).ToString("P2"),
                IsOutperforming = _strategyEquityVolatility < benchmark.Volatility,
                NumericDifference = (benchmark.Volatility - _strategyEquityVolatility) * Math.Sqrt(252)
            });
            
            // Sharpe Ratio
            comparisonMetrics.Add(new PerformanceComparisonMetric
            {
                Metric = "Sharpe Ratio",
                Strategy = _currentResult.SharpeRatio.ToString("F2"),
                Benchmark = benchmark.SharpeRatio.ToString("F2"),
                Difference = (_currentResult.SharpeRatio - benchmark.SharpeRatio).ToString("F2"),
                IsOutperforming = _currentResult.SharpeRatio > benchmark.SharpeRatio,
                NumericDifference = _currentResult.SharpeRatio - benchmark.SharpeRatio
            });
            
            // Sortino Ratio
            comparisonMetrics.Add(new PerformanceComparisonMetric
            {
                Metric = "Sortino Ratio",
                Strategy = _currentResult.SortinoRatio.ToString("F2"),
                Benchmark = benchmark.SortinoRatio.ToString("F2"),
                Difference = (_currentResult.SortinoRatio - benchmark.SortinoRatio).ToString("F2"),
                IsOutperforming = _currentResult.SortinoRatio > benchmark.SortinoRatio,
                NumericDifference = _currentResult.SortinoRatio - benchmark.SortinoRatio
            });
            
            // Information Ratio
            comparisonMetrics.Add(new PerformanceComparisonMetric
            {
                Metric = "Information Ratio",
                Strategy = _currentResult.InformationRatio.ToString("F2"),
                Benchmark = benchmark.InformationRatio.ToString("F2"),
                Difference = (_currentResult.InformationRatio - benchmark.InformationRatio).ToString("F2"),
                IsOutperforming = _currentResult.InformationRatio > benchmark.InformationRatio,
                NumericDifference = _currentResult.InformationRatio - benchmark.InformationRatio
            });
            
            // Alpha
            comparisonMetrics.Add(new PerformanceComparisonMetric
            {
                Metric = "Alpha",
                Strategy = "N/A",
                Benchmark = "0.00%",
                Difference = benchmark.Alpha.ToString("P2"),
                IsOutperforming = benchmark.Alpha > 0,
                NumericDifference = benchmark.Alpha
            });
            
            // Beta
            comparisonMetrics.Add(new PerformanceComparisonMetric
            {
                Metric = "Beta",
                Strategy = "1.00",
                Benchmark = benchmark.Beta.ToString("F2"),
                Difference = (1 - benchmark.Beta).ToString("F2"),
                IsOutperforming = benchmark.Beta < 1,
                NumericDifference = 1 - benchmark.Beta
            });
            
            PerformanceComparisonGrid.ItemsSource = comparisonMetrics;
        }
        
        private void UpdateBenchmarkReturnsChart()
        {
            // Clear existing series
            BenchmarkReturnsChart.Series = new SeriesCollection();
            
            // Add strategy return series - normalize from initial equity
            double initialEquity = _currentResult.EquityCurve.First().Equity;
            var strategyValues = _currentResult.EquityCurve
                .Select(e => e.Equity / initialEquity)
                .ToList();
            
            // Create a copy for use in relative mode
            var strategyAbsoluteValues = new List<double>(strategyValues);
                
            var strategySeries = new LineSeries
            {
                Title = "Strategy",
                Values = new ChartValues<double>(strategyValues),
                Stroke = Brushes.DarkRed,
                Fill = Brushes.Transparent,
                StrokeThickness = 3,
                PointGeometry = DefaultGeometries.Diamond,
                PointGeometrySize = 0,
                DataLabels = false,
                LabelPoint = point => $"Strategy: {point.Y.ToString("F2")}x ({((point.Y - 1) * 100).ToString("F2")}%)"
            };
            
            BenchmarkReturnsChart.Series.Add(strategySeries);
            
            // Add benchmark series
            foreach (var benchmark in _benchmarkData)
            {
                var benchmarkValues = new List<double>(benchmark.NormalizedReturns);
                
                // If showing relative performance, normalize benchmark against strategy
                if (_showRelativeReturns && benchmarkValues.Count > 0)
                {
                    for (int i = 0; i < benchmarkValues.Count && i < strategyAbsoluteValues.Count; i++)
                    {
                        benchmarkValues[i] = benchmarkValues[i] / strategyAbsoluteValues[i];
                    }
                    
                    // Also adjust strategy values to be 1.0 (flat line as baseline)
                    if (benchmark == _benchmarkData.First())
                    {
                        strategyValues = Enumerable.Repeat(1.0, strategyValues.Count).ToList();
                        strategySeries.Values = new ChartValues<double>(strategyValues);
                        strategySeries.Title = "Strategy (Baseline)";
                        
                        // Add a dashed horizontal line at 1.0 to represent the strategy baseline
                        var baselineSeries = new LineSeries
                        {
                            Title = "Baseline",
                            Values = new ChartValues<double>(strategyValues),
                            Stroke = Brushes.Gray,
                            StrokeDashArray = new DoubleCollection { 4, 2 },
                            Fill = Brushes.Transparent,
                            PointGeometry = null,
                            LineSmoothness = 0
                        };
                        BenchmarkReturnsChart.Series.Add(baselineSeries);
                    }
                }
                
                var benchmarkSeries = new LineSeries
                {
                    Title = _showRelativeReturns ? $"{benchmark.Name} (Relative to Strategy)" : benchmark.Name,
                    Values = new ChartValues<double>(benchmarkValues),
                    Stroke = GetBenchmarkColor(benchmark.Symbol),
                    Fill = Brushes.Transparent,
                    PointGeometry = DefaultGeometries.Circle,
                    PointGeometrySize = 0,
                    DataLabels = false,
                    LabelPoint = point => _showRelativeReturns ?
                        $"{benchmark.Name}: {point.Y.ToString("F2")}x Strategy" :
                        $"{benchmark.Name}: {point.Y.ToString("F2")}x ({((point.Y - 1) * 100).ToString("F2")}%)"
                };
                
                BenchmarkReturnsChart.Series.Add(benchmarkSeries);
            }
            
            // Add areas to highlight outperformance/underperformance when in relative mode
            if (_showRelativeReturns && _benchmarkData.Count > 0)
            {
                AddOutperformanceHighlights();
            }
            
            // Setup axes
            BenchmarkReturnsChart.AxisX.Clear();
            BenchmarkReturnsChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Date",
                Labels = _currentResult.EquityCurve.Select(e => e.Date.ToShortDateString()).ToArray(),
                LabelsRotation = 45,
                Separator = new LiveCharts.Wpf.Separator { Step = Math.Max(1, _currentResult.EquityCurve.Count / 10) }
            });
            
            BenchmarkReturnsChart.AxisY.Clear();
            BenchmarkReturnsChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = _showRelativeReturns ? "Relative Performance" : "Normalized Return",
                LabelFormatter = value => _showRelativeReturns ? 
                    value.ToString("F2") + "x" : 
                    value.ToString("F2") + "x"
            });
            
            // Configure chart for better interactivity
            BenchmarkReturnsChart.LegendLocation = LegendLocation.Right;
            BenchmarkReturnsChart.Zoom = ZoomingOptions.X;
            BenchmarkReturnsChart.Pan = PanningOptions.X;
            BenchmarkReturnsChart.DisableAnimations = true;
        }
        
        private void AddOutperformanceHighlights()
        {
            if (_benchmarkData.Count == 0)
                return;
                
            var benchmark = _benchmarkData[0]; // Use the first benchmark (usually S&P 500)
            var benchmarkValues = benchmark.NormalizedReturns;
            var strategyValues = _currentResult.EquityCurve
                .Select(e => e.Equity / _currentResult.EquityCurve.First().Equity)
                .ToList();
                
            // Create relative performance values
            var relativeValues = new List<double>();
            for (int i = 0; i < Math.Min(benchmarkValues.Count, strategyValues.Count); i++)
            {
                relativeValues.Add(benchmarkValues[i] / strategyValues[i]);
            }
            
            // Create series for outperformance highlights (values below 1 mean benchmark underperformance = strategy outperformance)
            var outperformanceSeries = new LineSeries
            {
                Title = "Strategy Outperformance",
                Values = new ChartValues<double>(relativeValues.Select(v => v < 1 ? v : double.NaN)),
                Stroke = Brushes.Transparent,
                Fill = new SolidColorBrush(Color.FromArgb(50, 0, 255, 0)), // Light green area
                PointGeometry = null
            };
            
            // Create series for underperformance highlights (values above 1 mean benchmark outperformance = strategy underperformance)
            var underperformanceSeries = new LineSeries
            {
                Title = "Strategy Underperformance",
                Values = new ChartValues<double>(relativeValues.Select(v => v > 1 ? v : double.NaN)),
                Stroke = Brushes.Transparent,
                Fill = new SolidColorBrush(Color.FromArgb(50, 255, 0, 0)), // Light red area
                PointGeometry = null
            };
            
            // Add the series
            BenchmarkReturnsChart.Series.Add(outperformanceSeries);
            BenchmarkReturnsChart.Series.Add(underperformanceSeries);
        }
        
        private void UpdateBenchmarkDrawdownChart()
        {
            // Clear existing series
            BenchmarkDrawdownChart.Series = new SeriesCollection();
            
            // Add strategy drawdown series
            var strategyDrawdownSeries = new LineSeries
            {
                Title = "Strategy",
                Values = new ChartValues<double>(_currentResult.DrawdownCurve.Select(d => d.Drawdown)),
                Stroke = Brushes.DarkRed,
                Fill = Brushes.Transparent,
                StrokeThickness = 3,
                PointGeometry = DefaultGeometries.Diamond,
                PointGeometrySize = 0,
                DataLabels = false,
                LabelPoint = point => $"Strategy Drawdown: {(point.Y * 100).ToString("F2")}%"
            };
            
            BenchmarkDrawdownChart.Series.Add(strategyDrawdownSeries);
            
            // Create benchmark drawdown series
            foreach (var benchmark in _benchmarkData)
            {
                // Calculate drawdown for the benchmark
                var drawdownValues = new List<double>();
                double peak = benchmark.HistoricalData[0].Close;
                
                foreach (var price in benchmark.HistoricalData)
                {
                    if (price.Close > peak)
                    {
                        peak = price.Close;
                    }
                    
                    double drawdown = (peak - price.Close) / peak;
                    drawdownValues.Add(drawdown);
                }
                
                var benchmarkSeries = new LineSeries
                {
                    Title = benchmark.Name,
                    Values = new ChartValues<double>(drawdownValues),
                    Stroke = GetBenchmarkColor(benchmark.Symbol),
                    Fill = Brushes.Transparent,
                    PointGeometry = DefaultGeometries.Circle,
                    PointGeometrySize = 0,
                    DataLabels = false,
                    LabelPoint = point => $"{benchmark.Name} Drawdown: {(point.Y * 100).ToString("F2")}%"
                };
                
                BenchmarkDrawdownChart.Series.Add(benchmarkSeries);
            }
            
            // Show maximum drawdown lines for better visualization
            foreach (var benchmark in _benchmarkData)
            {
                var maxDdLine = new LineSeries
                {
                    Title = $"{benchmark.Name} Max DD",
                    Values = new ChartValues<double>(Enumerable.Repeat(benchmark.MaxDrawdown, benchmark.HistoricalData.Count)),
                    Stroke = GetBenchmarkColor(benchmark.Symbol),
                    StrokeDashArray = new DoubleCollection { 4, 2 },
                    Fill = Brushes.Transparent,
                    PointGeometry = null,
                    LineSmoothness = 0
                };
                
                BenchmarkDrawdownChart.Series.Add(maxDdLine);
            }
            
            // Add strategy maximum drawdown line
            var strategyMaxDdLine = new LineSeries
            {
                Title = "Strategy Max DD",
                Values = new ChartValues<double>(Enumerable.Repeat(_currentResult.MaxDrawdown, _currentResult.DrawdownCurve.Count)),
                Stroke = Brushes.DarkRed,
                StrokeDashArray = new DoubleCollection { 4, 2 },
                Fill = Brushes.Transparent,
                PointGeometry = null,
                LineSmoothness = 0
            };
            
            BenchmarkDrawdownChart.Series.Add(strategyMaxDdLine);
            
            // Setup axes
            BenchmarkDrawdownChart.AxisX.Clear();
            BenchmarkDrawdownChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Date",
                Labels = _currentResult.DrawdownCurve.Select(d => d.Date.ToShortDateString()).ToArray(),
                LabelsRotation = 45,
                Separator = new LiveCharts.Wpf.Separator { Step = Math.Max(1, _currentResult.DrawdownCurve.Count / 10) }
            });
            
            BenchmarkDrawdownChart.AxisY.Clear();
            BenchmarkDrawdownChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Drawdown",
                LabelFormatter = value => (value * 100).ToString("F1") + "%"
            });
            
            // Configure chart for better interactivity
            BenchmarkDrawdownChart.LegendLocation = LegendLocation.Right;
            BenchmarkDrawdownChart.Zoom = ZoomingOptions.X;
            BenchmarkDrawdownChart.DisableAnimations = true;
        }
        
        private void UpdateVolatilityComparisonChart()
        {
            // Clear existing series
            VolatilityComparisonChart.Series = new SeriesCollection();
            
            // Create volatility comparison data
            var labels = new List<string> { "Strategy" };
            var volatilityValues = new List<double> { _strategyEquityVolatility * 100 }; // Daily volatility as percentage
            var annualizedVolValues = new List<double> { _strategyEquityVolatility * Math.Sqrt(252) * 100 }; // Annualized volatility
            
            // Add risk-reward metric (return-to-volatility ratio)
            var returnToVolValues = new List<double> { _currentResult.TotalReturn / (_strategyEquityVolatility * Math.Sqrt(252)) * 100 };
            
            foreach (var benchmark in _benchmarkData)
            {
                labels.Add(benchmark.Name);
                volatilityValues.Add(benchmark.Volatility * 100); // Daily volatility as percentage
                annualizedVolValues.Add(benchmark.Volatility * Math.Sqrt(252) * 100); // Annualized volatility
                returnToVolValues.Add(benchmark.TotalReturn / (benchmark.Volatility * Math.Sqrt(252)) * 100); // Return-to-volatility ratio
            }
            
            // Daily volatility column series
            var volatilitySeries = new ColumnSeries
            {
                Title = "Daily Volatility",
                Values = new ChartValues<double>(volatilityValues),
                Fill = Brushes.SteelBlue,
                MaxColumnWidth = 30,
                DataLabels = true,
                LabelPoint = point => $"{point.Y.ToString("F2")}%"
            };
            
            // Annualized volatility column series
            var annualizedVolSeries = new ColumnSeries
            {
                Title = "Annualized Volatility",
                Values = new ChartValues<double>(annualizedVolValues),
                Fill = Brushes.IndianRed,
                MaxColumnWidth = 30,
                DataLabels = true,
                LabelPoint = point => $"{point.Y.ToString("F2")}%"
            };
            
            // Return-to-volatility ratio column series
            var returnToVolSeries = new ColumnSeries
            {
                Title = "Return/Risk Ratio",
                Values = new ChartValues<double>(returnToVolValues),
                Fill = Brushes.ForestGreen,
                MaxColumnWidth = 30,
                DataLabels = true,
                LabelPoint = point => $"{point.Y.ToString("F2")}"
            };
            
            VolatilityComparisonChart.Series.Add(volatilitySeries);
            VolatilityComparisonChart.Series.Add(annualizedVolSeries);
            VolatilityComparisonChart.Series.Add(returnToVolSeries);
            
            // Setup axes
            VolatilityComparisonChart.AxisX.Clear();
            VolatilityComparisonChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "",
                Labels = labels.ToArray(),
                LabelsRotation = 0
            });
            
            VolatilityComparisonChart.AxisY.Clear();
            VolatilityComparisonChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Value",
                LabelFormatter = value => value.ToString("F1") + "%"
            });
            
            // Configure chart for better display
            VolatilityComparisonChart.LegendLocation = LegendLocation.Top;
            VolatilityComparisonChart.DisableAnimations = true;
        }
        
        private void UpdateRiskMetricsComparison()
        {
            // Create comparison data for risk metrics chart
            var riskMetricsData = new ObservableCollection<BenchmarkComparisonData>();
            
            // Add strategy as first item for comparison
            var strategyComparison = new BenchmarkComparisonData
            {
                Name = "Strategy",
                SharpeRatio = _currentResult.SharpeRatio,
                SortinoRatio = _currentResult.SortinoRatio,
                CalmarRatio = _currentResult.CalmarRatio,
                InformationRatio = _currentResult.InformationRatio,
                Beta = 1.0, // By definition, strategy's beta against itself is 1.0
                Alpha = 0.0, // Alpha against itself is 0
                TotalReturn = _currentResult.TotalReturn,
                MaxDrawdown = _currentResult.MaxDrawdown,
                Volatility = _strategyEquityVolatility,
                CAGR = _currentResult.CAGR
            };
            
            riskMetricsData.Add(strategyComparison);
            
            foreach (var benchmark in _benchmarkData)
            {
                riskMetricsData.Add(benchmark);
            }
            
            // Display risk metrics in the grid - make it more comprehensive
            RiskMetricsGrid.ItemsSource = riskMetricsData;
            
            // Create a column chart comparing Sharpe, Sortino, and Calmar ratios
            RiskMetricsComparisonChart.Series = new SeriesCollection();
            
            // Sharpe Ratio Series
            var sharpeRatioSeries = new ColumnSeries
            {
                Title = "Sharpe Ratio",
                Values = new ChartValues<double>(riskMetricsData.Select(b => b.SharpeRatio)),
                Fill = Brushes.Green,
                MaxColumnWidth = 20,
                DataLabels = true,
                LabelPoint = point => $"{point.Y.ToString("F2")}"
            };
            
            // Sortino Ratio Series
            var sortinoRatioSeries = new ColumnSeries
            {
                Title = "Sortino Ratio",
                Values = new ChartValues<double>(riskMetricsData.Select(b => b.SortinoRatio)),
                Fill = Brushes.Blue,
                MaxColumnWidth = 20,
                DataLabels = true,
                LabelPoint = point => $"{point.Y.ToString("F2")}"
            };
            
            // Calmar Ratio Series
            var calmarRatioSeries = new ColumnSeries
            {
                Title = "Calmar Ratio",
                Values = new ChartValues<double>(riskMetricsData.Select(b => b.CalmarRatio)),
                Fill = Brushes.Orange,
                MaxColumnWidth = 20,
                DataLabels = true,
                LabelPoint = point => $"{point.Y.ToString("F2")}"
            };
            
            // Information Ratio Series
            var informationRatioSeries = new ColumnSeries
            {
                Title = "Information Ratio",
                Values = new ChartValues<double>(riskMetricsData.Select(b => b.InformationRatio)),
                Fill = Brushes.Purple,
                MaxColumnWidth = 20,
                DataLabels = true,
                LabelPoint = point => $"{point.Y.ToString("F2")}"
            };
            
            RiskMetricsComparisonChart.Series.Add(sharpeRatioSeries);
            RiskMetricsComparisonChart.Series.Add(sortinoRatioSeries);
            RiskMetricsComparisonChart.Series.Add(calmarRatioSeries);
            RiskMetricsComparisonChart.Series.Add(informationRatioSeries);
            
            // Setup axes
            RiskMetricsComparisonChart.AxisX.Clear();
            RiskMetricsComparisonChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "",
                Labels = riskMetricsData.Select(b => b.Name).ToArray(),
                LabelsRotation = 0
            });
            
            RiskMetricsComparisonChart.AxisY.Clear();
            RiskMetricsComparisonChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Ratio Value",
                LabelFormatter = value => value.ToString("F2")
            });
            
            // Configure chart for better display
            RiskMetricsComparisonChart.LegendLocation = LegendLocation.Top;
            RiskMetricsComparisonChart.DisableAnimations = true;
            
            // Create tooltip for explanation
            RiskMetricsComparisonChart.ToolTip = new ToolTip
            {
                Content = "Sharpe Ratio: Return per unit of risk\n" +
                         "Sortino Ratio: Return per unit of downside risk\n" +
                         "Calmar Ratio: Return per unit of maximum drawdown\n" +
                         "Information Ratio: Excess return per unit of risk relative to benchmark"
            };
            
            // Create Risk vs Return scatter plot
            UpdateRiskReturnScatterPlot(riskMetricsData);
        }
        
        private void UpdateRiskReturnScatterPlot(ObservableCollection<BenchmarkComparisonData> riskMetricsData)
        {
            // Clear existing series
            RiskReturnScatterChart.Series = new SeriesCollection();
            
            // Create scatter series for risk/return comparison
            var scatterSeries = new ScatterSeries
            {
                Title = "Risk vs Return",
                Values = new ChartValues<ScatterPoint>(),
                MinPointShapeDiameter = 15,
                MaxPointShapeDiameter = 45,
                DataLabels = true
            };
            
            var colors = new Dictionary<string, Brush>
            {
                { "Strategy", Brushes.DarkRed }
            };
            
            // Add benchmark colors
            foreach (var kvp in _benchmarkColors)
            {
                colors.Add(kvp.Key, kvp.Value);
            }
            
            // Create custom points with size based on annualized volatility
            for (int i = 0; i < riskMetricsData.Count; i++)
            {
                var data = riskMetricsData[i];
                
                // X-axis: Max Drawdown (%)
                // Y-axis: Total Return (%)
                // Size: Proportional to volatility
                double x = data.MaxDrawdown * 100;
                double y = data.TotalReturn * 100;
                
                // Scale size based on volatility (annualized)
                double volatility = data.Volatility * Math.Sqrt(252) * 100;
                double size = Math.Max(15, Math.Min(45, volatility * 2));
                
                var point = new ScatterPoint(x, y);
                scatterSeries.Values.Add(point);
                
                // Create individual series with custom colors
                var individualSeries = new ScatterSeries
                {
                    Title = data.Name,
                    Values = new ChartValues<ScatterPoint> { point },
                    MinPointShapeDiameter = (int)size,
                    MaxPointShapeDiameter = (int)size,
                    Fill = data.Symbol != null && colors.ContainsKey(data.Symbol) ? 
                           colors[data.Symbol] : 
                           (data.Name == "Strategy" ? Brushes.DarkRed : Brushes.Gray),
                    DataLabels = true,
                    LabelPoint = p => data.Name
                };
                
                RiskReturnScatterChart.Series.Add(individualSeries);
            }
            
            // Setup axes
            RiskReturnScatterChart.AxisX.Clear();
            RiskReturnScatterChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Risk (Maximum Drawdown %)",
                LabelFormatter = value => value.ToString("F1") + "%",
                MinValue = 0
            });
            
            RiskReturnScatterChart.AxisY.Clear();
            RiskReturnScatterChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Return (%)",
                LabelFormatter = value => value.ToString("F1") + "%"
            });
            
            // Configure chart for better display
            RiskReturnScatterChart.LegendLocation = LegendLocation.Right;
            RiskReturnScatterChart.DisableAnimations = true;
            
            // Add dotted quadrant lines (0% return horizontal and average drawdown vertical)
            var zeroReturnLine = new LineSeries
            {
                Title = "Zero Return",
                Values = new ChartValues<double> { 0, 0 },
                ScalesYAt = 0,
                Stroke = Brushes.Gray,
                StrokeDashArray = new DoubleCollection { 4, 2 },
                Fill = Brushes.Transparent,
                PointGeometry = null,
                LineSmoothness = 0
            };
            
            // Calculate average drawdown
            double avgDrawdown = riskMetricsData.Average(d => d.MaxDrawdown * 100);
            
            var avgDrawdownLine = new LineSeries
            {
                Title = "Avg Max DD",
                Values = new ChartValues<double> { avgDrawdown, avgDrawdown },
                ScalesXAt = 0,
                Stroke = Brushes.Gray,
                StrokeDashArray = new DoubleCollection { 4, 2 },
                Fill = Brushes.Transparent,
                PointGeometry = null,
                LineSmoothness = 0
            };
            
            RiskReturnScatterChart.Series.Add(zeroReturnLine);
            RiskReturnScatterChart.Series.Add(avgDrawdownLine);
        }
        
        private void UpdatePerformanceAttributionChart()
        {
            if (_benchmarkData == null || _benchmarkData.Count == 0 || _currentResult == null)
                return;
            
            // Clear existing series
            PerformanceAttributionChart.Series = new SeriesCollection();
            
            // Select the first benchmark for attribution analysis
            var benchmark = _benchmarkData[0]; // Using the first benchmark (usually S&P 500)
            
            // Calculate relative performance (strategy returns - benchmark returns)
            // First align the dates to ensure we're comparing the same time periods
            var alignedReturns = AlignDailyReturnsForAttribution(_currentResult, benchmark);
            
            if (alignedReturns.strategyReturns.Count == 0 || alignedReturns.benchmarkReturns.Count == 0)
                return;
            
            // Calculate cumulative outperformance
            var cumulativeOutperformance = new List<double>();
            double cumulativeValue = 0;
            
            // Create daily outperformance values
            var dailyOutperformance = new List<double>();
            var dates = new List<DateTime>();
            
            for (int i = 0; i < alignedReturns.strategyReturns.Count; i++)
            {
                double diff = alignedReturns.strategyReturns[i] - alignedReturns.benchmarkReturns[i];
                dailyOutperformance.Add(diff * 100); // Convert to percentage
                
                cumulativeValue += diff;
                cumulativeOutperformance.Add(cumulativeValue * 100); // Convert to percentage
                
                dates.Add(alignedReturns.dates[i]);
            }
            
            // Create series for cumulative outperformance
            var cumulativeSeries = new LineSeries
            {
                Title = $"Cumulative Outperformance vs {benchmark.Name}",
                Values = new ChartValues<double>(cumulativeOutperformance),
                Stroke = Brushes.DarkBlue,
                Fill = Brushes.Transparent,
                StrokeThickness = 2,
                PointGeometry = null,
                LineSmoothness = 0.5,
                DataLabels = false
            };
            
            // Create area series for positive daily outperformance
            var positiveOutperformanceSeries = new LineSeries
            {
                Title = "Daily Outperformance",
                Values = new ChartValues<double>(dailyOutperformance.Select(v => v > 0 ? v : 0)),
                Stroke = Brushes.Green.Clone(),
                StrokeThickness = 1,
                Fill = new SolidColorBrush(Color.FromArgb(80, 0, 128, 0)), // Transparent green
                PointGeometry = null,
                LineSmoothness = 0
            };
            
            // Create area series for negative daily outperformance
            var negativeOutperformanceSeries = new LineSeries
            {
                Title = "Daily Underperformance",
                Values = new ChartValues<double>(dailyOutperformance.Select(v => v < 0 ? v : 0)),
                Stroke = Brushes.Red.Clone(),
                StrokeThickness = 1,
                Fill = new SolidColorBrush(Color.FromArgb(80, 128, 0, 0)), // Transparent red
                PointGeometry = null,
                LineSmoothness = 0
            };
            
            // Create zero line
            var zeroLine = new LineSeries
            {
                Title = "Zero Line",
                Values = new ChartValues<double>(Enumerable.Repeat(0.0, cumulativeOutperformance.Count)),
                Stroke = Brushes.Gray,
                StrokeDashArray = new DoubleCollection { 4, 2 },
                Fill = Brushes.Transparent,
                PointGeometry = null,
                LineSmoothness = 0
            };
            
            // Add series to chart
            PerformanceAttributionChart.Series.Add(positiveOutperformanceSeries);
            PerformanceAttributionChart.Series.Add(negativeOutperformanceSeries);
            PerformanceAttributionChart.Series.Add(zeroLine);
            PerformanceAttributionChart.Series.Add(cumulativeSeries);
            
            // Setup axes
            PerformanceAttributionChart.AxisX.Clear();
            PerformanceAttributionChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Date",
                Labels = dates.Select(d => d.ToShortDateString()).ToArray(),
                LabelsRotation = 45,
                Separator = new LiveCharts.Wpf.Separator { Step = Math.Max(1, dates.Count / 10) }
            });
            
            PerformanceAttributionChart.AxisY.Clear();
            PerformanceAttributionChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Outperformance (%)",
                LabelFormatter = value => value.ToString("F2") + "%"
            });
            
            // Configure chart for better display
            PerformanceAttributionChart.LegendLocation = LegendLocation.Top;
            PerformanceAttributionChart.Zoom = ZoomingOptions.X;
            PerformanceAttributionChart.DisableAnimations = true;
            
            // Calculate statistics for attribution
            double totalOutperformanceDays = dailyOutperformance.Count(v => v > 0);
            double totalDays = dailyOutperformance.Count;
            double outperformancePercentage = totalOutperformanceDays / totalDays * 100;
            double avgOutperformance = dailyOutperformance.Average();
            double totalCumulativeOutperformance = cumulativeOutperformance.Last();
            
            // Add tooltip with attribution statistics
            PerformanceAttributionChart.ToolTip = new ToolTip
            {
                Content = $"Strategy vs {benchmark.Name}:\n" +
                         $"Cumulative Outperformance: {totalCumulativeOutperformance:F2}%\n" +
                         $"Outperformance Days: {outperformancePercentage:F1}% of Days\n" +
                         $"Average Daily Outperformance: {avgOutperformance:F2}%"
            };
        }
        
        private (List<double> strategyReturns, List<double> benchmarkReturns, List<DateTime> dates) 
            AlignDailyReturnsForAttribution(BacktestingEngine.BacktestResult strategyResult, BenchmarkComparisonData benchmark)
        {
            List<double> strategyReturns = new List<double>();
            List<double> benchmarkReturns = new List<double>();
            List<DateTime> dates = new List<DateTime>();
            
            // Create dictionary of benchmark data for quick lookup
            var benchmarkByDate = benchmark.HistoricalData.ToDictionary(h => h.Date.Date, h => h);
            
            // For each day in the strategy result, find the corresponding benchmark data
            for (int i = 1; i < strategyResult.EquityCurve.Count; i++)
            {
                DateTime currentDate = strategyResult.EquityCurve[i].Date.Date;
                DateTime previousDate = strategyResult.EquityCurve[i - 1].Date.Date;
                
                if (benchmarkByDate.ContainsKey(currentDate) && benchmarkByDate.ContainsKey(previousDate))
                {
                    double strategyReturn = (strategyResult.EquityCurve[i].Equity - strategyResult.EquityCurve[i - 1].Equity) / 
                                          strategyResult.EquityCurve[i - 1].Equity;
                    
                    double benchmarkReturn = (benchmarkByDate[currentDate].Close - benchmarkByDate[previousDate].Close) / 
                                          benchmarkByDate[previousDate].Close;
                    
                    strategyReturns.Add(strategyReturn);
                    benchmarkReturns.Add(benchmarkReturn);
                    dates.Add(currentDate);
                }
            }
            
            return (strategyReturns, benchmarkReturns, dates);
        }
        
        private void BenchmarkCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            // Update active benchmark if a checkbox was just checked
            if (sender is CheckBox checkbox && checkbox.IsChecked == true)
            {
                string benchmarkType = null;
                if (checkbox == SPYCheckBox) benchmarkType = "SPY";
                else if (checkbox == QQQCheckBox) benchmarkType = "QQQ";
                else if (checkbox == IWMCheckBox) benchmarkType = "IWM";
                else if (checkbox == DIACheckBox) benchmarkType = "DIA";
                
                if (benchmarkType != null)
                {
                    UserSettingsService.SetActiveBenchmark(benchmarkType);
                    UpdateActiveBenchmarkDisplay();
                }
            }
            
            // Refresh benchmarks on checkbox change
            RefreshBenchmarks();
        }
        
        private void RefreshBenchmarksButton_Click(object sender, RoutedEventArgs e)
        {
            RefreshBenchmarks();
        }
        
        private async void RefreshBenchmarks()
        {
            if (_currentResult == null || _historicalData == null || _historicalData.Count == 0)
            {
                return;
            }
            
            await LoadBenchmarkData();
        }
        
        private void ManageCustomBenchmarksButton_Click(object sender, RoutedEventArgs e)
        {
            var manager = new CustomBenchmarkManager();
            bool? result = manager.ShowDialog();
            
            if (result == true && manager.SelectedBenchmark != null)
            {
                // The active benchmark has already been set in UserSettings by the CustomBenchmarkManager
                // Now we need to reload benchmarks and apply the new active selection
                LoadCustomBenchmarks();
                ApplyActiveBenchmarkSelection();
                
                // Refresh the benchmark data to reflect the new active benchmark
                RefreshBenchmarks();
            }
            else
            {
                // Just refresh the custom benchmarks list
                LoadCustomBenchmarks();
            }
        }
        
        private void LoadCustomBenchmarks()
        {
            // Save current selection
            CustomBenchmark currentSelection = CustomBenchmarkComboBox.SelectedItem as CustomBenchmark;
            
            // Clear and reload custom benchmarks
            _customBenchmarks.Clear();
            
            var benchmarks = _customBenchmarkService.GetCustomBenchmarks();
            foreach (var benchmark in benchmarks)
            {
                _customBenchmarks.Add(benchmark);
                
                // Log the benchmark details for debugging
                //DatabaseMonolith.Log("Debug", $"Loaded custom benchmark: ID={benchmark.Id}, Name={benchmark.Name}, DisplaySymbol={benchmark.DisplaySymbol}");
            }
            
            // Apply custom benchmarks to ComboBox
            CustomBenchmarkComboBox.ItemsSource = null; // Clear first
            CustomBenchmarkComboBox.ItemsSource = _customBenchmarks;
            
            // Log the count and types for debugging
            //DatabaseMonolith.Log("Debug", $"CustomBenchmarkComboBox ItemsSource set with {_customBenchmarks.Count} items");
            
            // Restore previous selection if possible
            if (currentSelection != null)
            {
                CustomBenchmark newSelection = _customBenchmarks.FirstOrDefault(b => b.Id == currentSelection.Id);
                if (newSelection != null)
                {
                    CustomBenchmarkComboBox.SelectedItem = newSelection;
                }
            }
            
            // Update the display in case custom benchmarks changed
            UpdateActiveBenchmarkDisplay();
        }
        
        /// <summary>
        /// Apply the active benchmark selection from user settings
        /// </summary>
        private void ApplyActiveBenchmarkSelection()
        {
            var (benchmarkType, benchmarkId) = UserSettingsService.GetActiveBenchmark();
            
            // Clear all selections first
            SPYCheckBox.IsChecked = false;
            QQQCheckBox.IsChecked = false;
            IWMCheckBox.IsChecked = false;
            DIACheckBox.IsChecked = false;
            CustomBenchmarkComboBox.SelectedItem = null;
            
            // Set the active benchmark based on user settings
            switch (benchmarkType)
            {
                case "SPY":
                    SPYCheckBox.IsChecked = true;
                    break;
                case "QQQ":
                    QQQCheckBox.IsChecked = true;
                    break;
                case "IWM":
                    IWMCheckBox.IsChecked = true;
                    break;
                case "DIA":
                    DIACheckBox.IsChecked = true;
                    break;
                case "CUSTOM":
                    if (!string.IsNullOrEmpty(benchmarkId))
                    {
                        var customBenchmark = _customBenchmarks.FirstOrDefault(b => b.Id == benchmarkId);
                        if (customBenchmark != null)
                        {
                            CustomBenchmarkComboBox.SelectedItem = customBenchmark;
                        }
                    }
                    break;
            }
            
            // Update the UI display
            UpdateActiveBenchmarkDisplay();
        }
        
        /// <summary>
        /// Update the active benchmark display text
        /// </summary>
        private void UpdateActiveBenchmarkDisplay()
        {
            var (benchmarkType, benchmarkId) = UserSettingsService.GetActiveBenchmark();
            
            switch (benchmarkType)
            {
                case "SPY":
                    ActiveBenchmarkText.Text = "S&P 500 (SPY)";
                    break;
                case "QQQ":
                    ActiveBenchmarkText.Text = "NASDAQ (QQQ)";
                    break;
                case "IWM":
                    ActiveBenchmarkText.Text = "Russell 2000 (IWM)";
                    break;
                case "DIA":
                    ActiveBenchmarkText.Text = "Dow Jones (DIA)";
                    break;
                case "CUSTOM":
                    if (!string.IsNullOrEmpty(benchmarkId))
                    {
                        var customBenchmark = _customBenchmarks.FirstOrDefault(b => b.Id == benchmarkId);
                        if (customBenchmark != null)
                        {
                            ActiveBenchmarkText.Text = $"{customBenchmark.Name} ({customBenchmark.DisplaySymbol})";
                        }
                        else
                        {
                            ActiveBenchmarkText.Text = "Custom (Not Found)";
                        }
                    }
                    else
                    {
                        ActiveBenchmarkText.Text = "Custom (None Selected)";
                    }
                    break;
                default:
                    ActiveBenchmarkText.Text = "S&P 500 (SPY)"; // Default fallback
                    break;
            }
        }
        
        private void CustomBenchmarkComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Set active custom benchmark if a selection was made
            var selectedBenchmark = CustomBenchmarkComboBox.SelectedItem as CustomBenchmark;
            if (selectedBenchmark != null)
            {
                UserSettingsService.SetActiveCustomBenchmark(selectedBenchmark.Id);
                UpdateActiveBenchmarkDisplay();
            }
            
            // Refresh benchmarks when custom benchmark selection changes
            RefreshBenchmarks();
        }
        
        /// <summary>
        /// Gets the color for a benchmark symbol, with fallbacks for custom benchmarks
        /// </summary>
        private Brush GetBenchmarkColor(string symbol)
        {
            if (_benchmarkColors.ContainsKey(symbol))
                return _benchmarkColors[symbol];
            
            // For custom benchmarks, use the CUSTOM color or create a unique color based on the symbol
            if (symbol.Contains("+") || symbol.Contains("%"))
            {
                // This is likely a custom benchmark
                return _benchmarkColors["CUSTOM"];
            }
            
            // Default fallback
            return Brushes.Gray;
        }
        
        // Event handlers for interactive chart controls
        
        private void ShowRelativeReturns_Changed(object sender, RoutedEventArgs e)
        {
            _showRelativeReturns = ShowRelativeReturnsToggle.IsChecked ?? false;
            UpdateBenchmarkReturnsChart();
        }
        
        private void ResetChartZoom_Click(object sender, RoutedEventArgs e)
        {
            // Reset zoom on all charts
            BenchmarkReturnsChart.AxisX[0].MinValue = double.NaN;
            BenchmarkReturnsChart.AxisX[0].MaxValue = double.NaN;
            BenchmarkReturnsChart.AxisY[0].MinValue = double.NaN;
            BenchmarkReturnsChart.AxisY[0].MaxValue = double.NaN;
            
            BenchmarkDrawdownChart.AxisX[0].MinValue = double.NaN;
            BenchmarkDrawdownChart.AxisX[0].MaxValue = double.NaN;
            BenchmarkDrawdownChart.AxisY[0].MinValue = double.NaN;
            BenchmarkDrawdownChart.AxisY[0].MaxValue = double.NaN;
            
            CombinedPerformanceChart.AxisX[0].MinValue = double.NaN;
            CombinedPerformanceChart.AxisX[0].MaxValue = double.NaN;
            CombinedPerformanceChart.AxisY[0].MinValue = double.NaN;
            CombinedPerformanceChart.AxisY[0].MaxValue = double.NaN;
            
            PerformanceAttributionChart.AxisX[0].MinValue = double.NaN;
            PerformanceAttributionChart.AxisX[0].MaxValue = double.NaN;
            PerformanceAttributionChart.AxisY[0].MinValue = double.NaN;
            PerformanceAttributionChart.AxisY[0].MaxValue = double.NaN;
        }
        
        private void TimeRangeSelector_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_currentResult == null || _historicalData == null || _benchmarkData == null || _benchmarkData.Count == 0)
                return;
            
            // Get selected time range
            var selectedItem = (ComboBoxItem)TimeRangeSelector.SelectedItem;
            string range = selectedItem.Content.ToString();
            
            // Calculate the start date based on the selected range
            DateTime endDate = _currentResult.EndDate;
            DateTime startDate;
            
            switch (range)
            {
                case "YTD":
                    startDate = new DateTime(endDate.Year, 1, 1);
                    break;
                case "1 Month":
                    startDate = endDate.AddMonths(-1);
                    break;
                case "3 Months":
                    startDate = endDate.AddMonths(-3);
                    break;
                case "6 Months":
                    startDate = endDate.AddMonths(-6);
                    break;
                case "1 Year":
                    startDate = endDate.AddYears(-1);
                    break;
                default: // "All Data"
                    startDate = _currentResult.StartDate;
                    break;
            }
            
            // Apply filter to charts
            ApplyDateRangeFilter(startDate, endDate);
        }
        
        private void ApplyDateRangeFilter(DateTime startDate, DateTime endDate)
        {
            if (_currentResult == null)
                return;
            
            // Find the index of the start and end dates
            int startIndex = _currentResult.EquityCurve.FindIndex(e => e.Date >= startDate);
            int endIndex = _currentResult.EquityCurve.FindIndex(e => e.Date > endDate);
            
            if (startIndex < 0) startIndex = 0;
            if (endIndex < 0) endIndex = _currentResult.EquityCurve.Count - 1;
            
            // Apply the filter to returns chart
            BenchmarkReturnsChart.AxisX[0].MinValue = startIndex;
            BenchmarkReturnsChart.AxisX[0].MaxValue = endIndex;
            
            // Apply to performance attribution chart
            PerformanceAttributionChart.AxisX[0].MinValue = startIndex;
            PerformanceAttributionChart.AxisX[0].MaxValue = endIndex;
        }
        
        private void HighlightOutperformance_Click(object sender, RoutedEventArgs e)
        {
            // Toggle highlighting of outperformance periods on the returns chart
            UpdatePerformanceAttributionChart();
        }
        
        private void RunMonteCarloButton_Click(object sender, RoutedEventArgs e)
        {
            RunMonteCarloSimulation();
        }
        
        /// <summary>
        /// Run Monte Carlo simulation on the current backtest result
        /// </summary>
        private async void RunMonteCarloSimulation()
        {
            if (_currentResult == null)
            {
                MonteCarloStatusText.Text = "No backtest results to simulate";
                return;
            }
            
            // Get simulation count from UI
            int simulationCount = 1000;
            if (SimulationCountComboBox.SelectedItem is ComboBoxItem selectedItem)
            {
                if (int.TryParse(selectedItem.Content.ToString(), out int count))
                {
                    simulationCount = count;
                }
            }
            
            try
            {
                // Set status to running
                MonteCarloStatusText.Text = $"Running {simulationCount} simulations...";
                RunMonteCarloButton.IsEnabled = false;
                
                // Create a new BacktestingEngine with the same historical data service
                var engine = new BacktestingEngine(_historicalDataService);
                
                // Run simulation on a background thread
                await Task.Run(() => {
                    _currentResult = engine.RunMonteCarloSimulation(_currentResult, simulationCount);
                });
                
                // Update UI with results
                UpdateMonteCarloVisualization();
                
                MonteCarloStatusText.Text = $"Completed {simulationCount} simulations";
            }
            catch (Exception ex)
            {
                MonteCarloStatusText.Text = "Simulation failed";
                //DatabaseMonolith.Log("Error", "Monte Carlo simulation failed", ex.ToString());
            }
            finally
            {
                RunMonteCarloButton.IsEnabled = true;
            }
        }
        
        /// <summary>
        /// Update the Monte Carlo visualization with the current simulation results
        /// </summary>
        private void UpdateMonteCarloVisualization()
        {
            if (_currentResult == null || !_currentResult.HasMonteCarloResults)
            {
                return;
            }
            
            var mcResult = _currentResult.MonteCarloResults;
            
            // Update statistics display
            UpdateMonteCarloStatistics(mcResult);
            
            // Update equity fan chart
            UpdateMonteCarloEquityChart(mcResult);
            
            // Update return distribution chart
            UpdateReturnDistributionChart(mcResult);
            
            // Update drawdown distribution chart
            UpdateDrawdownDistributionChart(mcResult);
        }
        
        /// <summary>
        /// Update the Monte Carlo statistics display
        /// </summary>
        private void UpdateMonteCarloStatistics(BacktestingEngine.MonteCarloSimulationResult mcResult)
        {
            // Get initial capital from the first equity point
            double initialCapital = _currentResult.EquityCurve.FirstOrDefault()?.Equity ?? 10000;
            
            // Calculate percentages for returns
            Return5PercentText.Text = ((mcResult.ReturnPercentiles["5%"] - initialCapital) / initialCapital).ToString("P2");
            Return25PercentText.Text = ((mcResult.ReturnPercentiles["25%"] - initialCapital) / initialCapital).ToString("P2");
            Return50PercentText.Text = ((mcResult.ReturnPercentiles["50%"] - initialCapital) / initialCapital).ToString("P2");
            Return75PercentText.Text = ((mcResult.ReturnPercentiles["75%"] - initialCapital) / initialCapital).ToString("P2");
            Return95PercentText.Text = ((mcResult.ReturnPercentiles["95%"] - initialCapital) / initialCapital).ToString("P2");
            
            // Format drawdowns as percentages
            Drawdown5PercentText.Text = mcResult.DrawdownPercentiles["5%"].ToString("P2");
            Drawdown25PercentText.Text = mcResult.DrawdownPercentiles["25%"].ToString("P2");
            Drawdown50PercentText.Text = mcResult.DrawdownPercentiles["50%"].ToString("P2");
            Drawdown75PercentText.Text = mcResult.DrawdownPercentiles["75%"].ToString("P2");
            Drawdown95PercentText.Text = mcResult.DrawdownPercentiles["95%"].ToString("P2");
            
            // Risk metrics
            VaR95Text.Text = mcResult.ValueAtRisk95.ToString("P2");
            VaR99Text.Text = mcResult.ValueAtRisk99.ToString("P2");
            CVaR95Text.Text = mcResult.ConditionalValueAtRisk95.ToString("P2");
            ProfitProbabilityText.Text = mcResult.ProbabilityOfProfit.ToString("P1");
            BeatBacktestProbabilityText.Text = mcResult.ProbabilityOfExceedingBacktestReturn.ToString("P1");
        }
        
        /// <summary>
        /// Update the Monte Carlo equity fan chart
        /// </summary>
        private void UpdateMonteCarloEquityChart(BacktestingEngine.MonteCarloSimulationResult mcResult)
        {
            // Clear existing series
            MonteCarloEquityChart.Series = new SeriesCollection();
            
            // Add the original equity curve as a reference
            var originalSeries = new LineSeries
            {
                Title = "Original Backtest",
                Values = new ChartValues<double>(_currentResult.EquityCurve.Select(e => e.Equity)),
                Stroke = Brushes.Blue,
                StrokeThickness = 3,
                Fill = Brushes.Transparent,
                PointGeometry = null
            };
            MonteCarloEquityChart.Series.Add(originalSeries);
            
            // Add the representative percentile curves
            string[] percentiles = new[] { "5%", "25%", "50%", "75%", "95%" };
            Brush[] colors = new[] { Brushes.Red, Brushes.Orange, Brushes.Green, Brushes.SkyBlue, Brushes.Purple };
            
            for (int i = 0; i < percentiles.Length; i++)
            {
                string percentile = percentiles[i];
                if (mcResult.PercentileEquityCurves.ContainsKey(percentile))
                {
                    var series = new LineSeries
                    {
                        Title = percentile + " Percentile",
                        Values = new ChartValues<double>(mcResult.PercentileEquityCurves[percentile].Select(e => e.Equity)),
                        Stroke = colors[i],
                        StrokeThickness = 2,
                        Fill = Brushes.Transparent,
                        PointGeometry = null,
                        LineSmoothness = 0.5
                    };
                    MonteCarloEquityChart.Series.Add(series);
                }
            }
            
            // Setup axes
            MonteCarloEquityChart.AxisX.Clear();
            MonteCarloEquityChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Time",
                Labels = _currentResult.EquityCurve.Select((e, i) => i % Math.Max(1, _currentResult.EquityCurve.Count / 10) == 0 ? 
                    e.Date.ToShortDateString() : "").ToArray()
            });
            
            MonteCarloEquityChart.AxisY.Clear();
            MonteCarloEquityChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Equity",
                LabelFormatter = value => value.ToString("C0")
            });
            
            // Configure chart for better display
            MonteCarloEquityChart.LegendLocation = LegendLocation.Top;
            MonteCarloEquityChart.DisableAnimations = true;
        }
        
        /// <summary>
        /// Update the return distribution histogram
        /// </summary>
        private void UpdateReturnDistributionChart(BacktestingEngine.MonteCarloSimulationResult mcResult)
        {
            // Clear existing series
            ReturnDistributionChart.Series = new SeriesCollection();
            
            // Get initial capital from the first equity point
            double initialCapital = _currentResult.EquityCurve.FirstOrDefault()?.Equity ?? 10000;
            
            // Create histogram data from final values
            var returnPercentages = mcResult.FinalValues.Select(v => (v - initialCapital) / initialCapital).ToList();
            
            // Calculate histogram bins - create 20 bins between min and max
            double min = returnPercentages.Min();
            double max = returnPercentages.Max();
            double binWidth = (max - min) / 20.0;
            
            var bins = new int[20];
            foreach (var ret in returnPercentages)
            {
                int binIndex = Math.Min(19, Math.Max(0, (int)((ret - min) / binWidth)));
                bins[binIndex]++;
            }
            
            // Create a column series for the histogram
            var histogramSeries = new ColumnSeries
            {
                Title = "Return Distribution",
                Values = new ChartValues<int>(bins),
                Fill = new SolidColorBrush(Color.FromRgb(30, 144, 255)),
                MaxColumnWidth = 15,
                ColumnPadding = 1
            };
            
            ReturnDistributionChart.Series.Add(histogramSeries);
            
            // Add vertical line for original backtest return
            var originalReturn = _currentResult.TotalReturn;
            double originalReturnBin = Math.Min(19, Math.Max(0, (int)((originalReturn - min) / binWidth)));
            
            var originalReturnSeries = new LineSeries
            {
                Title = "Backtest Return",
                Values = new ChartValues<ObservablePoint> { 
                    new ObservablePoint(originalReturnBin, 0),
                    new ObservablePoint(originalReturnBin, bins.Max() * 1.1)
                },
                Stroke = Brushes.Red,
                StrokeThickness = 2,
                Fill = Brushes.Transparent,
                PointGeometry = null,
                LineSmoothness = 0
            };
            ReturnDistributionChart.Series.Add(originalReturnSeries);
            
            // Setup axes
            ReturnDistributionChart.AxisX.Clear();
            ReturnDistributionChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Return",
                Labels = Enumerable.Range(0, 20).Select(i => {
                    double binValue = min + i * binWidth;
                    return i % 5 == 0 ? (binValue * 100).ToString("F0") + "%" : "";
                }).ToArray()
            });
            
            ReturnDistributionChart.AxisY.Clear();
            ReturnDistributionChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Frequency",
                MinValue = 0
            });
            
            // Configure chart
            ReturnDistributionChart.LegendLocation = LegendLocation.None;
            ReturnDistributionChart.DisableAnimations = true;
        }
        
        /// <summary>
        /// Update the drawdown distribution histogram
        /// </summary>
        private void UpdateDrawdownDistributionChart(BacktestingEngine.MonteCarloSimulationResult mcResult)
        {
            // Clear existing series
            DrawdownDistributionChart.Series = new SeriesCollection();
            
            // Calculate histogram bins - create 20 bins between 0 and max drawdown
            double max = mcResult.MaxDrawdowns.Max();
            double binWidth = max / 20.0;
            
            var bins = new int[20];
            foreach (var dd in mcResult.MaxDrawdowns)
            {
                int binIndex = Math.Min(19, Math.Max(0, (int)(dd / binWidth)));
                bins[binIndex]++;
            }
            
            // Create a column series for the histogram
            var histogramSeries = new ColumnSeries
            {
                Title = "Drawdown Distribution",
                Values = new ChartValues<int>(bins),
                Fill = new SolidColorBrush(Color.FromRgb(220, 20, 60)), // Crimson
                MaxColumnWidth = 15,
                ColumnPadding = 1
            };
            
            DrawdownDistributionChart.Series.Add(histogramSeries);
            
            // Add vertical line for original backtest drawdown
            var originalDD = _currentResult.MaxDrawdown;
            double originalDDBin = Math.Min(19, Math.Max(0, (int)(originalDD / binWidth)));
            
            var originalDDSeries = new LineSeries
            {
                Title = "Backtest Drawdown",
                Values = new ChartValues<ObservablePoint> { 
                    new ObservablePoint(originalDDBin, 0),
                    new ObservablePoint(originalDDBin, bins.Max() * 1.1)
                },
                Stroke = Brushes.Blue,
                StrokeThickness = 2,
                Fill = Brushes.Transparent,
                PointGeometry = null,
                LineSmoothness = 0
            };
            DrawdownDistributionChart.Series.Add(originalDDSeries);
            
            // Setup axes
            DrawdownDistributionChart.AxisX.Clear();
            DrawdownDistributionChart.AxisX.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Maximum Drawdown",
                Labels = Enumerable.Range(0, 20).Select(i => {
                    double binValue = i * binWidth;
                    return i % 4 == 0 ? (binValue * 100).ToString("F0") + "%" : "";
                }).ToArray()
            });
            
            DrawdownDistributionChart.AxisY.Clear();
            DrawdownDistributionChart.AxisY.Add(new LiveCharts.Wpf.Axis
            {
                Title = "Frequency",
                MinValue = 0
            });
            
            // Configure chart
            DrawdownDistributionChart.LegendLocation = LegendLocation.None;
            DrawdownDistributionChart.DisableAnimations = true;
        }

        /// <summary>
        /// Handles the selection changed event for the benchmark comparison tab control
        /// </summary>
        private void BenchmarkComparisonTabControl_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Guard against events during initialization or when no tab is selected
            if (sender is not TabControl tabControl || tabControl.SelectedItem is not TabItem selectedTab)
                return;

            try
            {
                string tabHeader = selectedTab.Header?.ToString() ?? "";
                
                // Log the tab selection for debugging
                //DatabaseMonolith.Log("Info", $"BacktestResultsControl: Tab selected - {tabHeader}");

                // Handle specific tab selections that might need data refresh or special handling
                switch (tabHeader)
                {
                    case "Performance Summary":
                        // Ensure the combined performance chart and comparison grid are up to date
                        if (_benchmarkData != null && _benchmarkData.Count > 0 && _currentResult != null)
                        {
                            UpdateCombinedPerformanceChart();
                            if (_benchmarkData.Count > 0)
                            {
                                UpdatePerformanceComparisonGrid(_benchmarkData[0]);
                            }
                        }
                        break;

                    case "Cumulative Returns":
                        // Refresh the cumulative returns chart
                        if (_benchmarkData != null && _benchmarkData.Count > 0 && _currentResult != null)
                        {
                            UpdateBenchmarkReturnsChart();
                        }
                        break;

                    case "Drawdown Comparison":
                        // Refresh the drawdown comparison chart
                        if (_benchmarkData != null && _benchmarkData.Count > 0 && _currentResult != null)
                        {
                            UpdateBenchmarkDrawdownChart();
                        }
                        break;

                    case "Volatility Comparison":
                        // Refresh the volatility comparison chart
                        if (_benchmarkData != null && _benchmarkData.Count > 0 && _currentResult != null)
                        {
                            UpdateVolatilityComparisonChart();
                        }
                        break;

                    case "Performance Attribution":
                        // Refresh the performance attribution chart
                        if (_benchmarkData != null && _benchmarkData.Count > 0 && _currentResult != null)
                        {
                            UpdatePerformanceAttributionChart();
                        }
                        break;

                    case "Risk-Adjusted Metrics":
                        // Refresh the risk metrics comparison
                        if (_benchmarkData != null && _benchmarkData.Count > 0 && _currentResult != null)
                        {
                            UpdateRiskMetricsComparison();
                        }
                        break;

                    case "Monte Carlo Simulation":
                        // Ensure Monte Carlo visualization is current if results exist
                        if (_currentResult != null && _currentResult.HasMonteCarloResults)
                        {
                            UpdateMonteCarloVisualization();
                        }
                        break;
                }

                // Mark the event as handled to prevent it from bubbling up to MainWindow
                e.Handled = true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error handling tab selection change in BacktestResultsControl: {ex.Message}", ex.ToString());
            }
        }
    }
}
