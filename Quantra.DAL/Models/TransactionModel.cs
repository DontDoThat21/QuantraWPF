using System;
using System.ComponentModel;

namespace Quantra.Models
{
    public class TransactionModel : INotifyPropertyChanged
    {
        private string symbol;
        private string transactionType; // BUY or SELL
        private int quantity;
        private double executionPrice;
        private DateTime executionTime;
        private bool isPaperTrade;
        private double fees;
        private double realizedPnL;
        private double realizedPnLPercentage;
        private double totalValue;
        private string notes;
        private string status;
        private string orderSource; // "Manual" or "Automated"

        public string Symbol
        {
            get => symbol;
            set
            {
                symbol = value;
                OnPropertyChanged(nameof(Symbol));
                CalculateTotalValue();
            }
        }

        public string TransactionType
        {
            get => transactionType;
            set
            {
                transactionType = value;
                OnPropertyChanged(nameof(TransactionType));
            }
        }

        public int Quantity
        {
            get => quantity;
            set
            {
                quantity = value;
                OnPropertyChanged(nameof(Quantity));
                CalculateTotalValue();
            }
        }

        public double ExecutionPrice
        {
            get => executionPrice;
            set
            {
                executionPrice = value;
                OnPropertyChanged(nameof(ExecutionPrice));
                CalculateTotalValue();
            }
        }

        public DateTime ExecutionTime
        {
            get => executionTime;
            set
            {
                executionTime = value;
                OnPropertyChanged(nameof(ExecutionTime));
            }
        }

        public bool IsPaperTrade
        {
            get => isPaperTrade;
            set
            {
                isPaperTrade = value;
                OnPropertyChanged(nameof(IsPaperTrade));
            }
        }

        public double Fees
        {
            get => fees;
            set
            {
                fees = value;
                OnPropertyChanged(nameof(Fees));
                // Fees affect RealizedPnL
                CalculateRealizedPnL();
            }
        }

        public double RealizedPnL
        {
            get => realizedPnL;
            set
            {
                realizedPnL = value;
                OnPropertyChanged(nameof(RealizedPnL));
            }
        }

        public double RealizedPnLPercentage
        {
            get => realizedPnLPercentage;
            set
            {
                realizedPnLPercentage = value;
                OnPropertyChanged(nameof(RealizedPnLPercentage));
            }
        }

        public double TotalValue
        {
            get => totalValue;
            set
            {
                totalValue = value;
                OnPropertyChanged(nameof(TotalValue));
            }
        }

        public string Notes
        {
            get => notes;
            set
            {
                notes = value;
                OnPropertyChanged(nameof(Notes));
            }
        }

        public string Status
        {
            get => status;
            set
            {
                status = value;
                OnPropertyChanged(nameof(Status));
            }
        }

        public string OrderSource
        {
            get => orderSource;
            set
            {
                orderSource = value;
                OnPropertyChanged(nameof(OrderSource));
            }
        }

        // Helper method to calculate total value
        private void CalculateTotalValue()
        {
            TotalValue = ExecutionPrice * Quantity;
            // When total value changes, profit and loss may change
            CalculateRealizedPnL();
        }

        // todo finish p and l calculations (do i need alpha vantage api?)
        // Default placeholder for P&L calculation - would be replaced with actual calculation
        private void CalculateRealizedPnL()
        {
            // In a real app, this would calculate P&L based on purchase price vs. sale price
            // For demo purposes, we'll just set a default value that could be overridden
            // RealizedPnL = TotalValue * 0.05; // 5% profit as placeholder
            OnPropertyChanged(nameof(RealizedPnL));
            OnPropertyChanged(nameof(RealizedPnLPercentage));
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
