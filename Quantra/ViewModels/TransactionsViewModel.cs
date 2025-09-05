using Quantra.Commands;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.ViewModels.Base;
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Windows.Input;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;
using Microsoft.Win32;
using System.IO;
using System.Text;
using Quantra.DAL.Notifications;

namespace Quantra.ViewModels
{
    public class TransactionsViewModel : ViewModelBase
    {
        private readonly ITransactionService _transactionService;
        private readonly INotificationService _notificationService;
        
        #region Properties
        
        private ObservableCollection<TransactionModel> _transactions;
        public ObservableCollection<TransactionModel> Transactions
        {
            get => _transactions;
            set => SetProperty(ref _transactions, value);
        }

        private ObservableCollection<TransactionModel> _filteredTransactions;
        public ObservableCollection<TransactionModel> FilteredTransactions
        {
            get => _filteredTransactions;
            set => SetProperty(ref _filteredTransactions, value);
        }

        private TransactionModel _selectedTransaction;
        public TransactionModel SelectedTransaction
        {
            get => _selectedTransaction;
            set => SetProperty(ref _selectedTransaction, value);
        }

        private DateTime _startDate = DateTime.Now.AddMonths(-1);
        public DateTime StartDate
        {
            get => _startDate;
            set => SetProperty(ref _startDate, value);
        }

        private DateTime _endDate = DateTime.Now;
        public DateTime EndDate
        {
            get => _endDate;
            set => SetProperty(ref _endDate, value);
        }

        private string _searchText;
        public string SearchText
        {
            get => _searchText;
            set => SetProperty(ref _searchText, value);
        }

        private string _selectedTransactionType = "All";
        public string SelectedTransactionType
        {
            get => _selectedTransactionType;
            set => SetProperty(ref _selectedTransactionType, value);
        }

        private string _notificationText;
        public string NotificationText
        {
            get => _notificationText;
            set => SetProperty(ref _notificationText, value);
        }

        private PackIconKind _notificationIcon;
        public PackIconKind NotificationIcon
        {
            get => _notificationIcon;
            set => SetProperty(ref _notificationIcon, value);
        }

        private Brush _notificationIconColor;
        public Brush NotificationIconColor
        {
            get => _notificationIconColor;
            set => SetProperty(ref _notificationIconColor, value);
        }

        private Brush _notificationBorderBrush;
        public Brush NotificationBorderBrush
        {
            get => _notificationBorderBrush;
            set => SetProperty(ref _notificationBorderBrush, value);
        }

        private double _realTradingPnL;
        public double RealTradingPnL
        {
            get => _realTradingPnL;
            set => SetProperty(ref _realTradingPnL, value);
        }

        private double _paperTradingPnL;
        public double PaperTradingPnL
        {
            get => _paperTradingPnL;
            set => SetProperty(ref _paperTradingPnL, value);
        }

        private double _winRate;
        public double WinRate
        {
            get => _winRate;
            set => SetProperty(ref _winRate, value);
        }

        private double _totalFees;
        public double TotalFees
        {
            get => _totalFees;
            set => SetProperty(ref _totalFees, value);
        }
        
        #endregion

        #region Commands
        
        public ICommand SearchCommand { get; private set; }
        public ICommand ApplyFiltersCommand { get; private set; }
        public ICommand LoadTransactionsCommand { get; private set; }
        public ICommand ExportDataCommand { get; private set; }
        public ICommand ViewDetailsCommand { get; private set; }
        public ICommand CloseCommand { get; private set; }
        public ICommand SearchTextKeyUpCommand { get; private set; }
        
        #endregion

        // Event to view transaction details
        public event Action<TransactionModel> ViewDetails;
        
        // Event to close the window
        public event Action Close;

        // Constructor with dependency injection
        public TransactionsViewModel(ITransactionService transactionService, INotificationService notificationService)
        {
            _transactionService = transactionService ?? throw new ArgumentNullException(nameof(transactionService));
            _notificationService = notificationService ?? throw new ArgumentNullException(nameof(notificationService));
            
            // Initialize collections
            _transactions = new ObservableCollection<TransactionModel>();
            _filteredTransactions = new ObservableCollection<TransactionModel>();
            
            // Initialize commands
            SearchCommand = new RelayCommand((object _) => ExecuteSearch());
            ApplyFiltersCommand = new RelayCommand((object _) => ExecuteApplyFilters());
            LoadTransactionsCommand = new RelayCommand((object _) => ExecuteLoadTransactions());
            ExportDataCommand = new RelayCommand((object _) => ExecuteExportData());
            ViewDetailsCommand = new RelayCommand((object _) => ExecuteViewDetails(), (object _) => SelectedTransaction != null);
            CloseCommand = new RelayCommand((object _) => Close?.Invoke());
            SearchTextKeyUpCommand = new RelayCommand((object param) => ExecuteSearchTextKeyUp(param));
            
            // Load transactions
            LoadTransactions();
        }
        
        #region Helper Methods
        
        /// <summary>
        /// Converts DAL NotificationIcon enum to MaterialDesign PackIconKind
        /// </summary>
        private static PackIconKind ConvertToPackIconKind(Quantra.DAL.Notifications.NotificationIcon notificationIcon)
        {
            return notificationIcon switch
            {
                Quantra.DAL.Notifications.NotificationIcon.Info => PackIconKind.Information,
                Quantra.DAL.Notifications.NotificationIcon.Success => PackIconKind.CheckCircle,
                Quantra.DAL.Notifications.NotificationIcon.Warning => PackIconKind.AlertCircle,
                Quantra.DAL.Notifications.NotificationIcon.Error => PackIconKind.AlertCircle,
                Quantra.DAL.Notifications.NotificationIcon.TrendingUp => PackIconKind.TrendingUp,
                Quantra.DAL.Notifications.NotificationIcon.ChartLine => PackIconKind.ChartLine,
                Quantra.DAL.Notifications.NotificationIcon.ChartBubble => PackIconKind.ChartBubble,
                Quantra.DAL.Notifications.NotificationIcon.Calculator => PackIconKind.Calculator,
                _ => PackIconKind.Information
            };
        }
        
        /// <summary>
        /// Converts hex color string to Color object
        /// </summary>
        private static Color ConvertHexToColor(string hexColor)
        {
            try
            {
                return (Color)ColorConverter.ConvertFromString(hexColor);
            }
            catch
            {
                return Colors.Blue; // Default fallback color
            }
        }
        
        #endregion
        
        #region Command Execution Methods
        
        private void ExecuteSearchTextKeyUp(object param)
        {
            if (param is KeyEventArgs keyArgs && keyArgs.Key == Key.Enter)
            {
                ExecuteSearch();
            }
        }
        
        private void ExecuteSearch()
        {
            Search(SearchText);
            _notificationService.ShowNotification("Search applied.", 
                ConvertToPackIconKind(Quantra.DAL.Notifications.NotificationIcon.Info), 
                ConvertHexToColor("#2196F3"));
        }
        
        private void ExecuteApplyFilters()
        {
            ApplyFilters();
            _notificationService.ShowNotification("Filters applied.", 
                ConvertToPackIconKind(Quantra.DAL.Notifications.NotificationIcon.Info), 
                ConvertHexToColor("#2196F3"));
        }
        
        private void ExecuteLoadTransactions()
        {
            LoadTransactions();
            _notificationService.ShowNotification("Transaction data refreshed.", 
                ConvertToPackIconKind(Quantra.DAL.Notifications.NotificationIcon.Success), 
                ConvertHexToColor("#00C853"));
        }
        
        private void ExecuteExportData()
        {
            try
            {
                ExportData();
                _notificationService.ShowNotification(NotificationText, 
                    ConvertToPackIconKind(Quantra.DAL.Notifications.NotificationIcon.Success), 
                    ConvertHexToColor("#00C853"));
            }
            catch (Exception ex)
            {
                _notificationService.ShowNotification($"Export failed: {ex.Message}", 
                    ConvertToPackIconKind(Quantra.DAL.Notifications.NotificationIcon.Error), 
                    ConvertHexToColor("#FF1744"));
                DatabaseMonolith.Log("Error", "Failed to export transaction data", ex.ToString());
            }
        }
        
        private void ExecuteViewDetails()
        {
            if (SelectedTransaction != null)
            {
                // Event to notify the view to show transaction details
                ViewDetails?.Invoke(SelectedTransaction);
                
                // Also update notification text for feedback
                NotificationText = $"Viewing details for {SelectedTransaction.Symbol} transaction";
            }
        }
        
        #endregion

        #region Methods
        
        public void LoadTransactions()
        {
            try
            {
                var transactions = _transactionService.GetTransactions();
                
                // Rebuild collections to avoid any IEnumerable type mismatch issues
                Transactions = new ObservableCollection<TransactionModel>();
                foreach (var tx in transactions)
                {
                    Transactions.Add(tx);
                }
                
                FilteredTransactions = new ObservableCollection<TransactionModel>(Transactions);
                
                UpdateStatistics();
            }
            catch (Exception ex)
            {
                // Log error
                DatabaseMonolith.Log("Error", "Failed to load transactions", ex.ToString());
                NotificationText = $"Failed to load transactions: {ex.Message}";
            }
        }

        public void Search(string searchText)
        {
            if (string.IsNullOrWhiteSpace(searchText))
            {
                FilteredTransactions = new ObservableCollection<TransactionModel>(Transactions);
            }
            else
            {
                var filtered = Transactions.Where(t => 
                    t.Symbol.Contains(searchText, StringComparison.OrdinalIgnoreCase) ||
                    t.Notes?.Contains(searchText, StringComparison.OrdinalIgnoreCase) == true ||
                    t.TransactionType.Contains(searchText, StringComparison.OrdinalIgnoreCase)
                ).ToList();
                
                FilteredTransactions = new ObservableCollection<TransactionModel>(filtered);
            }
        }

        public void ApplyFilters()
        {
            if (StartDate > EndDate)
            {
                NotificationText = "Start date must be before end date";
                return;
            }
            
            var filtered = Transactions.Where(t => 
                t.ExecutionTime >= StartDate && 
                t.ExecutionTime <= EndDate &&
                (SelectedTransactionType == "All" || t.TransactionType == SelectedTransactionType)
            ).ToList();
            
            FilteredTransactions = new ObservableCollection<TransactionModel>(filtered);
        }

        public void ExportData()
        {
            try
            {
                var saveFileDialog = new SaveFileDialog
                {
                    Filter = "CSV Files (*.csv)|*.csv|All files (*.*)|*.*",
                    DefaultExt = "csv",
                    FileName = $"Transactions_Export_{DateTime.Now:yyyyMMdd}"
                };

                if (saveFileDialog.ShowDialog() == true)
                {
                    var filePath = saveFileDialog.FileName;

                    using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
                    {
                        // Write header
                        writer.WriteLine("Symbol,Type,Quantity,Price,Total,Date,IsPaperTrade,Fees,PnL,PnLPercent,Notes");
                        
                        // Write data
                        foreach (var transaction in FilteredTransactions)
                        {
                            writer.WriteLine($"{transaction.Symbol},{transaction.TransactionType},{transaction.Quantity}," +
                                           $"{transaction.ExecutionPrice},{transaction.TotalValue},{transaction.ExecutionTime:yyyy-MM-dd HH:mm:ss}," +
                                           $"{transaction.IsPaperTrade},{transaction.Fees},{transaction.RealizedPnL}," +
                                           $"{transaction.RealizedPnLPercentage},{transaction.Notes?.Replace(',', ' ')}");
                        }
                    }

                    NotificationText = $"Exported {FilteredTransactions.Count} transactions to {Path.GetFileName(filePath)}";
                }
                else
                {
                    NotificationText = "Export cancelled";
                }
            }
            catch (Exception ex)
            {
                NotificationText = $"Export failed: {ex.Message}";
                DatabaseMonolith.Log("Error", "Failed to export transactions", ex.ToString());
                throw;
            }
        }

        private void UpdateStatistics()
        {
            if (Transactions.Count == 0)
            {
                RealTradingPnL = 0;
                PaperTradingPnL = 0;
                WinRate = 0;
                TotalFees = 0;
                return;
            }

            var realTrades = Transactions.Where(t => !t.IsPaperTrade).ToList();
            var paperTrades = Transactions.Where(t => t.IsPaperTrade).ToList();
            
            RealTradingPnL = realTrades.Sum(t => t.RealizedPnL);
            PaperTradingPnL = paperTrades.Sum(t => t.RealizedPnL);
            
            TotalFees = Transactions.Sum(t => t.Fees);
            
            var profitableTrades = Transactions.Count(t => t.RealizedPnL > 0);
            WinRate = Transactions.Count > 0 ? (double)profitableTrades / Transactions.Count : 0;
        }
        
        #endregion
    }
}
