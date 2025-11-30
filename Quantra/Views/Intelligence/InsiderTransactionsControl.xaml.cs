using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Views.Intelligence
{
    /// <summary>
    /// Converter to get appropriate color for transaction types
    /// </summary>
    public class TransactionTypeColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var code = value as string;
            if (string.IsNullOrEmpty(code))
                return new SolidColorBrush(Color.FromRgb(62, 62, 86)); // Default gray

            return code switch
            {
                "P" => new SolidColorBrush(Color.FromRgb(32, 192, 64)), // Purchase - Green
                "S" => new SolidColorBrush(Color.FromRgb(192, 32, 32)), // Sale - Red
                "M" or "X" => new SolidColorBrush(Color.FromRgb(100, 149, 237)), // Exercise - Blue
                "A" or "G" => new SolidColorBrush(Color.FromRgb(255, 204, 0)), // Award - Yellow
                "D" => new SolidColorBrush(Color.FromRgb(255, 140, 0)), // Sale to Issuer - Orange
                "F" => new SolidColorBrush(Color.FromRgb(138, 43, 226)), // Tax Payment - Purple
                _ => new SolidColorBrush(Color.FromRgb(62, 62, 86)) // Default gray
            };
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Simple bool to visibility converter
    /// </summary>
    public class BoolToVisibilityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool boolValue)
            {
                return boolValue ? Visibility.Visible : Visibility.Collapsed;
            }
            return Visibility.Collapsed;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Interaction logic for InsiderTransactionsControl.xaml
    /// </summary>
    public partial class InsiderTransactionsControl : UserControl
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private List<InsiderTransactionData> _allTransactions;
        private List<InsiderTransactionData> _filteredTransactions;
        private readonly IUserSettingsService _userSettingsService;
        private readonly System.Data.IDbConnection _connection;
        private bool _isLoadingAllSymbols = false;

        public InsiderTransactionsControl()
        {
            InitializeComponent();
            // Get services from DI if available
            try
            {
                _alphaVantageService = App.ServiceProvider?.GetService(typeof(AlphaVantageService)) as AlphaVantageService;
                _loggingService = App.ServiceProvider?.GetService(typeof(LoggingService)) as LoggingService;
                _userSettingsService = App.ServiceProvider?.GetService(typeof(IUserSettingsService)) as IUserSettingsService;
                
                // Get database connection
                _connection = Quantra.DAL.Data.ConnectionHelper.GetConnection();
                
                // Initialize database table if needed
                InitializeInsiderTransactionsTable();
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Service initialization error: {ex.Message}";
            }

            // Set default date range (last 90 days)
            EndDatePicker.SelectedDate = DateTime.Today;
            StartDatePicker.SelectedDate = DateTime.Today.AddDays(-90);
        }

        private async void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            if (LoadAllSymbolsCheckBox.IsChecked == true)
            {
                await LoadAllSymbolsInsiderTransactions();
            }
            else
            {
                await LoadInsiderTransactions();
            }
        }

        private void LoadAllSymbolsCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            // Enable/disable symbol textbox based on checkbox state
            SymbolTextBox.IsEnabled = LoadAllSymbolsCheckBox.IsChecked != true;
        }

        private async void SymbolTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                await LoadInsiderTransactions();
            }
        }

        private async System.Threading.Tasks.Task LoadInsiderTransactions()
        {
            var symbol = SymbolTextBox.Text?.Trim().ToUpper();
            if (string.IsNullOrWhiteSpace(symbol))
            {
                StatusText.Text = "Please enter a valid stock symbol.";
                return;
            }

            if (_alphaVantageService == null)
            {
                StatusText.Text = "Alpha Vantage service not available.";
                return;
            }

            try
            {
                LoadingIndicator.Visibility = Visibility.Visible;
                LoadButton.IsEnabled = false;
                StatusText.Text = $"Loading insider transactions for {symbol}...";

                var response = await _alphaVantageService.GetInsiderTransactionsAsync(symbol);

                if (response != null && response.Transactions.Count > 0)
                {
                    _allTransactions = response.Transactions;
                    ApplyFilters();
                    UpdateSummary();
                    StatusText.Text = $"Loaded {_allTransactions.Count} transactions for {symbol}";
                    _loggingService?.Log("Info", $"Loaded {_allTransactions.Count} insider transactions for {symbol}");
                }
                else
                {
                    _allTransactions = null;
                    _filteredTransactions = null;
                    TransactionsGrid.ItemsSource = null;
                    TotalBuysText.Text = "";
                    TotalSellsText.Text = "";
                    StatusText.Text = $"No insider transactions found for {symbol}.";
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                _loggingService?.LogErrorWithContext(ex, $"Error loading insider transactions for {symbol}");
            }
            finally
            {
                LoadingIndicator.Visibility = Visibility.Collapsed;
                LoadButton.IsEnabled = true;
            }
        }

        private void TransactionTypeFilter_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            ApplyFilters();
        }

        private void DateFilter_Changed(object sender, SelectionChangedEventArgs e)
        {
            ApplyFilters();
        }

        private void ClearFilterButton_Click(object sender, RoutedEventArgs e)
        {
            TransactionTypeFilter.SelectedIndex = 0;
            EndDatePicker.SelectedDate = DateTime.Today;
            StartDatePicker.SelectedDate = DateTime.Today.AddDays(-90);
            ApplyFilters();
        }

        private void ApplyFilters()
        {
            if (TransactionsGrid == null)
                return;

            if (_allTransactions == null || _allTransactions.Count == 0)
            {
                _filteredTransactions = new List<InsiderTransactionData>();
                TransactionsGrid.ItemsSource = _filteredTransactions;
                return;
            }

            _filteredTransactions = _allTransactions.ToList();

            // Apply transaction type filter
            if (TransactionTypeFilter.SelectedItem is ComboBoxItem selectedType)
            {
                var filter = selectedType.Content?.ToString();
                _filteredTransactions = filter switch
                {
                    "Purchases" => _filteredTransactions.Where(t => t.TransactionCode == "P").ToList(),
                    "Sales" => _filteredTransactions.Where(t => t.TransactionCode == "S").ToList(),
                    "Option Exercises" => _filteredTransactions.Where(t => t.TransactionCode == "M" || t.TransactionCode == "X").ToList(),
                    "Awards/Grants" => _filteredTransactions.Where(t => t.TransactionCode == "A" || t.TransactionCode == "G").ToList(),
                    _ => _filteredTransactions
                };
            }

            // Apply date range filter
            if (StartDatePicker.SelectedDate.HasValue)
            {
                var startDate = StartDatePicker.SelectedDate.Value;
                _filteredTransactions = _filteredTransactions.Where(t => t.TransactionDate >= startDate).ToList();
            }

            if (EndDatePicker.SelectedDate.HasValue)
            {
                var endDate = EndDatePicker.SelectedDate.Value.AddDays(1); // Include the end date
                _filteredTransactions = _filteredTransactions.Where(t => t.TransactionDate < endDate).ToList();
            }

            // Sort by filing date descending
            _filteredTransactions = _filteredTransactions.OrderByDescending(t => t.FilingDate).ToList();

            TransactionsGrid.ItemsSource = _filteredTransactions;
            UpdateSummary();
        }

        private void UpdateSummary()
        {
            if (_filteredTransactions == null || _filteredTransactions.Count == 0)
            {
                TotalBuysText.Text = "";
                TotalSellsText.Text = "";
                return;
            }

            var buys = _filteredTransactions.Where(t => t.IsBuy).ToList();
            var sells = _filteredTransactions.Where(t => t.IsSell).ToList();

            double totalBuyValue = buys.Sum(t => t.TotalValue);
            double totalSellValue = sells.Sum(t => t.TotalValue);

            TotalBuysText.Text = $"Buys: {buys.Count} (${totalBuyValue:N0})";
            TotalSellsText.Text = $"Sells: {sells.Count} (${totalSellValue:N0})";
        }

        /// <summary>
        /// Public method to load data for a specific symbol programmatically
        /// </summary>
        public async System.Threading.Tasks.Task LoadSymbolAsync(string symbol)
        {
            if (!string.IsNullOrWhiteSpace(symbol))
            {
                SymbolTextBox.Text = symbol.ToUpper();
                await LoadInsiderTransactions();
            }
        }

        #region Database Caching Methods

        /// <summary>
        /// Initializes the InsiderTransactions table in the database
        /// </summary>
        private void InitializeInsiderTransactionsTable()
        {
            try
            {
                if (_connection.State != System.Data.ConnectionState.Open)
                {
                    _connection.Open();
                }

                using (var cmd = _connection.CreateCommand())
                {
                    cmd.CommandText = @"
                        CREATE TABLE IF NOT EXISTS InsiderTransactions (
                            Id INTEGER PRIMARY KEY AUTOINCREMENT,
                            Symbol TEXT NOT NULL,
                            FilingDate TEXT NOT NULL,
                            TransactionDate TEXT NOT NULL,
                            OwnerName TEXT,
                            OwnerCik TEXT,
                            OwnerTitle TEXT,
                            SecurityType TEXT,
                            TransactionCode TEXT,
                            SharesTraded INTEGER NOT NULL DEFAULT 0,
                            PricePerShare REAL NOT NULL DEFAULT 0.0,
                            SharesOwnedFollowing INTEGER NOT NULL DEFAULT 0,
                            AcquisitionOrDisposal TEXT,
                            LastUpdated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(Symbol, FilingDate, TransactionDate, OwnerCik)
                        );";
                    cmd.ExecuteNonQuery();
                }

                // Create indexes
                using (var cmd = _connection.CreateCommand())
                {
                    cmd.CommandText = @"
                        CREATE INDEX IF NOT EXISTS idx_insider_transactions_symbol ON InsiderTransactions(Symbol);
                        CREATE INDEX IF NOT EXISTS idx_insider_transactions_date ON InsiderTransactions(TransactionDate);
                        CREATE INDEX IF NOT EXISTS idx_insider_transactions_lastupdated ON InsiderTransactions(LastUpdated);";
                    cmd.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Failed to initialize InsiderTransactions table", ex.ToString());
            }
        }

        /// <summary>
        /// Saves insider transactions to the database cache
        /// </summary>
        private void SaveTransactionsToCache(List<InsiderTransactionData> transactions)
        {
            if (transactions == null || transactions.Count == 0)
                return;

            try
            {
                if (_connection.State != System.Data.ConnectionState.Open)
                {
                    _connection.Open();
                }

                foreach (var transaction in transactions)
                {
                    using (var cmd = _connection.CreateCommand())
                    {
                        cmd.CommandText = @"
                            INSERT OR REPLACE INTO InsiderTransactions 
                            (Symbol, FilingDate, TransactionDate, OwnerName, OwnerCik, OwnerTitle, 
                             SecurityType, TransactionCode, SharesTraded, PricePerShare, 
                             SharesOwnedFollowing, AcquisitionOrDisposal, LastUpdated)
                            VALUES (@Symbol, @FilingDate, @TransactionDate, @OwnerName, @OwnerCik, @OwnerTitle,
                                    @SecurityType, @TransactionCode, @SharesTraded, @PricePerShare,
                                    @SharesOwnedFollowing, @AcquisitionOrDisposal, @LastUpdated)";

                        var param = cmd.CreateParameter();
                        param.ParameterName = "@Symbol";
                        param.Value = transaction.Symbol;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@FilingDate";
                        param.Value = transaction.FilingDate.ToString("yyyy-MM-dd HH:mm:ss");
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@TransactionDate";
                        param.Value = transaction.TransactionDate.ToString("yyyy-MM-dd HH:mm:ss");
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@OwnerName";
                        param.Value = (object)transaction.OwnerName ?? DBNull.Value;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@OwnerCik";
                        param.Value = (object)transaction.OwnerCik ?? DBNull.Value;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@OwnerTitle";
                        param.Value = (object)transaction.OwnerTitle ?? DBNull.Value;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@SecurityType";
                        param.Value = (object)transaction.SecurityType ?? DBNull.Value;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@TransactionCode";
                        param.Value = (object)transaction.TransactionCode ?? DBNull.Value;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@SharesTraded";
                        param.Value = transaction.SharesTraded;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@PricePerShare";
                        param.Value = transaction.PricePerShare;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@SharesOwnedFollowing";
                        param.Value = transaction.SharesOwnedFollowing;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@AcquisitionOrDisposal";
                        param.Value = (object)transaction.AcquisitionOrDisposal ?? DBNull.Value;
                        cmd.Parameters.Add(param);

                        param = cmd.CreateParameter();
                        param.ParameterName = "@LastUpdated";
                        param.Value = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
                        cmd.Parameters.Add(param);

                        cmd.ExecuteNonQuery();
                    }
                }

                _loggingService?.Log("Info", $"Cached {transactions.Count} insider transactions to database");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to save insider transactions to cache");
            }
        }

        /// <summary>
        /// Loads insider transactions from the database cache
        /// </summary>
        private List<InsiderTransactionData> LoadTransactionsFromCache(string symbol = null)
        {
            var transactions = new List<InsiderTransactionData>();

            try
            {
                if (_connection.State != System.Data.ConnectionState.Open)
                {
                    _connection.Open();
                }

                using (var cmd = _connection.CreateCommand())
                {
                    if (string.IsNullOrWhiteSpace(symbol))
                    {
                        // Load all transactions
                        cmd.CommandText = @"SELECT Symbol, FilingDate, TransactionDate, OwnerName, OwnerCik, OwnerTitle,
                                            SecurityType, TransactionCode, SharesTraded, PricePerShare,
                                            SharesOwnedFollowing, AcquisitionOrDisposal
                                            FROM InsiderTransactions
                                            ORDER BY TransactionDate DESC";
                    }
                    else
                    {
                        // Load transactions for specific symbol
                        cmd.CommandText = @"SELECT Symbol, FilingDate, TransactionDate, OwnerName, OwnerCik, OwnerTitle,
                                            SecurityType, TransactionCode, SharesTraded, PricePerShare,
                                            SharesOwnedFollowing, AcquisitionOrDisposal
                                            FROM InsiderTransactions
                                            WHERE Symbol = @Symbol
                                            ORDER BY TransactionDate DESC";
                        
                        var param = cmd.CreateParameter();
                        param.ParameterName = "@Symbol";
                        param.Value = symbol.ToUpper();
                        cmd.Parameters.Add(param);
                    }

                    using (var reader = cmd.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            transactions.Add(new InsiderTransactionData
                            {
                                Symbol = reader["Symbol"].ToString(),
                                FilingDate = DateTime.Parse(reader["FilingDate"].ToString()),
                                TransactionDate = DateTime.Parse(reader["TransactionDate"].ToString()),
                                OwnerName = reader["OwnerName"].ToString(),
                                OwnerCik = reader["OwnerCik"].ToString(),
                                OwnerTitle = reader["OwnerTitle"].ToString(),
                                SecurityType = reader["SecurityType"].ToString(),
                                TransactionCode = reader["TransactionCode"].ToString(),
                                SharesTraded = Convert.ToInt32(reader["SharesTraded"]),
                                PricePerShare = Convert.ToDouble(reader["PricePerShare"]),
                                SharesOwnedFollowing = Convert.ToInt32(reader["SharesOwnedFollowing"]),
                                AcquisitionOrDisposal = reader["AcquisitionOrDisposal"].ToString()
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to load insider transactions from cache");
            }

            return transactions;
        }

        /// <summary>
        /// Gets a list of all unique symbols in the database
        /// </summary>
        private List<string> GetCachedSymbols()
        {
            var symbols = new List<string>();

            try
            {
                if (_connection.State != System.Data.ConnectionState.Open)
                {
                    _connection.Open();
                }

                using (var cmd = _connection.CreateCommand())
                {
                    cmd.CommandText = "SELECT DISTINCT Symbol FROM InsiderTransactions ORDER BY Symbol";
                    
                    using (var reader = cmd.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            symbols.Add(reader["Symbol"].ToString());
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to get cached symbols");
            }

            return symbols;
        }

        #endregion

        #region Load All Symbols Methods

        /// <summary>
        /// Loads insider transactions for all symbols
        /// </summary>
        private async System.Threading.Tasks.Task LoadAllSymbolsInsiderTransactions()
        {
            if (_alphaVantageService == null)
            {
                StatusText.Text = "Alpha Vantage service not available.";
                return;
            }

            if (_isLoadingAllSymbols)
            {
                StatusText.Text = "Already loading all symbols...";
                return;
            }

            try
            {
                _isLoadingAllSymbols = true;
                LoadingIndicator.Visibility = Visibility.Visible;
                LoadButton.IsEnabled = false;
                StatusText.Text = "Loading insider transactions for all symbols...";

                // First, try to load from cache
                _allTransactions = LoadTransactionsFromCache();

                if (_allTransactions.Count > 0)
                {
                    StatusText.Text = $"Loaded {_allTransactions.Count} cached transactions. Fetching updates...";
                    ApplyFilters();
                    UpdateSummary();
                }

                // Get list of symbols to fetch (either from settings or default list)
                var symbolsToFetch = GetSymbolsToFetch();

                // Fetch data for each symbol
                int successCount = 0;
                int errorCount = 0;
                var newTransactions = new List<InsiderTransactionData>();

                for (int i = 0; i < symbolsToFetch.Count; i++)
                {
                    var symbol = symbolsToFetch[i];
                    StatusText.Text = $"Loading {symbol}... ({i + 1}/{symbolsToFetch.Count})";

                    try
                    {
                        var response = await _alphaVantageService.GetInsiderTransactionsAsync(symbol);

                        if (response != null && response.Transactions.Count > 0)
                        {
                            newTransactions.AddRange(response.Transactions);
                            successCount++;
                        }
                    }
                    catch (Exception ex)
                    {
                        errorCount++;
                        _loggingService?.Log("Warning", $"Failed to load insider transactions for {symbol}", ex.Message);
                    }

                    // Small delay to avoid rate limiting
                    await System.Threading.Tasks.Task.Delay(800);
                }

                // Save new transactions to cache
                if (newTransactions.Count > 0)
                {
                    SaveTransactionsToCache(newTransactions);
                }

                // Reload all from cache to get merged data
                _allTransactions = LoadTransactionsFromCache();
                ApplyFilters();
                UpdateSummary();

                StatusText.Text = $"Loaded {_allTransactions.Count} total transactions ({successCount} symbols successful, {errorCount} errors)";
                _loggingService?.Log("Info", $"Loaded insider transactions for all symbols: {successCount} successful, {errorCount} errors");
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                _loggingService?.LogErrorWithContext(ex, "Error loading insider transactions for all symbols");
            }
            finally
            {
                _isLoadingAllSymbols = false;
                LoadingIndicator.Visibility = Visibility.Collapsed;
                LoadButton.IsEnabled = true;
            }
        }

        /// <summary>
        /// Gets the list of symbols to fetch insider transactions for
        /// </summary>
        private List<string> GetSymbolsToFetch()
        {
            try
            {
                // Try to get symbols from user settings or a predefined list
                var settingsSymbols = _userSettingsService?.GetUserPreference("InsiderTransactionsSymbols", null);
                
                if (!string.IsNullOrEmpty(settingsSymbols))
                {
                    return settingsSymbols.Split(',').Select(s => s.Trim().ToUpper()).Where(s => !string.IsNullOrEmpty(s)).ToList();
                }
            }
            catch { }

            // Default to S&P 500 top stocks
            return new List<string>
            {
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "XOM",
                "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "COST",
                "AVGO", "KO", "LLY", "TMO", "ADBE", "MCD", "ACN", "CSCO", "ABT", "WMT",
                "DHR", "NKE", "NEE", "TXN", "DIS", "INTC", "VZ", "PM", "CRM", "CMCSA",
                "NFLX", "UNP", "AMD", "QCOM", "UPS", "ORCL", "HON", "BMY", "RTX", "BA"
            };
        }

        #endregion
    }
}
