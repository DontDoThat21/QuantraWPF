using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using Quantra.DAL.TradingEngine.Orders;

namespace Quantra.DAL.TradingEngine.Positions
{
    /// <summary>
    /// Represents a trading position (aggregated from multiple orders)
    /// </summary>
    public class TradingPosition : INotifyPropertyChanged
    {
        private string _symbol = string.Empty;
        private int _quantity;
        private decimal _averageCost;
        private decimal _currentPrice;
        private decimal _unrealizedPnL;
        private decimal _realizedPnL;
        private AssetType _assetType;
        private DateTime _openedTime;
        private DateTime _lastUpdateTime;
        private List<OrderFill> _fills = new List<OrderFill>();

        public event PropertyChangedEventHandler? PropertyChanged;

        public TradingPosition()
        {
            _openedTime = DateTime.UtcNow;
            _lastUpdateTime = DateTime.UtcNow;
        }

        /// <summary>
        /// Symbol of the position
        /// </summary>
        public string Symbol
        {
            get => _symbol;
            set => SetProperty(ref _symbol, value);
        }

        /// <summary>
        /// Current quantity (positive = long, negative = short)
        /// </summary>
        public int Quantity
        {
            get => _quantity;
            set
            {
                if (SetProperty(ref _quantity, value))
                {
                    OnPropertyChanged(nameof(IsLong));
                    OnPropertyChanged(nameof(IsShort));
                    OnPropertyChanged(nameof(IsFlat));
                    OnPropertyChanged(nameof(MarketValue));
                    UpdateUnrealizedPnL();
                }
            }
        }

        /// <summary>
        /// Average cost per share
        /// </summary>
        public decimal AverageCost
        {
            get => _averageCost;
            set
            {
                if (SetProperty(ref _averageCost, value))
                {
                    OnPropertyChanged(nameof(TotalCost));
                    UpdateUnrealizedPnL();
                }
            }
        }

        /// <summary>
        /// Current market price
        /// </summary>
        public decimal CurrentPrice
        {
            get => _currentPrice;
            set
            {
                if (SetProperty(ref _currentPrice, value))
                {
                    OnPropertyChanged(nameof(MarketValue));
                    UpdateUnrealizedPnL();
                }
            }
        }

        /// <summary>
        /// Unrealized profit/loss
        /// </summary>
        public decimal UnrealizedPnL
        {
            get => _unrealizedPnL;
            set
            {
                if (SetProperty(ref _unrealizedPnL, value))
                {
                    OnPropertyChanged(nameof(UnrealizedPnLPercent));
                    OnPropertyChanged(nameof(TotalPnL));
                }
            }
        }

        /// <summary>
        /// Realized profit/loss (from closed trades)
        /// </summary>
        public decimal RealizedPnL
        {
            get => _realizedPnL;
            set
            {
                if (SetProperty(ref _realizedPnL, value))
                {
                    OnPropertyChanged(nameof(TotalPnL));
                }
            }
        }

        /// <summary>
        /// Type of asset
        /// </summary>
        public AssetType AssetType
        {
            get => _assetType;
            set => SetProperty(ref _assetType, value);
        }

        /// <summary>
        /// Time the position was opened
        /// </summary>
        public DateTime OpenedTime
        {
            get => _openedTime;
            set => SetProperty(ref _openedTime, value);
        }

        /// <summary>
        /// Time of last update
        /// </summary>
        public DateTime LastUpdateTime
        {
            get => _lastUpdateTime;
            set => SetProperty(ref _lastUpdateTime, value);
        }

        /// <summary>
        /// All fills that make up this position
        /// </summary>
        public List<OrderFill> Fills
        {
            get => _fills;
            set => SetProperty(ref _fills, value);
        }

        /// <summary>
        /// Whether the position is long
        /// </summary>
        public bool IsLong => Quantity > 0;

        /// <summary>
        /// Whether the position is short
        /// </summary>
        public bool IsShort => Quantity < 0;

        /// <summary>
        /// Whether there is no position
        /// </summary>
        public bool IsFlat => Quantity == 0;

        /// <summary>
        /// Total cost basis
        /// </summary>
        public decimal TotalCost => Math.Abs(Quantity) * AverageCost;

        /// <summary>
        /// Current market value
        /// </summary>
        public decimal MarketValue => Math.Abs(Quantity) * CurrentPrice;

        /// <summary>
        /// Unrealized P&L as percentage of cost
        /// </summary>
        public decimal UnrealizedPnLPercent => TotalCost > 0 ? (UnrealizedPnL / TotalCost) * 100 : 0;

        /// <summary>
        /// Total P&L (realized + unrealized)
        /// </summary>
        public decimal TotalPnL => RealizedPnL + UnrealizedPnL;

        /// <summary>
        /// Updates the position with a new fill
        /// </summary>
        public void AddFill(OrderFill fill)
        {
            if (fill == null) return;

            Fills.Add(fill);
            
            int fillQuantity = fill.Side == OrderSide.Buy ? fill.Quantity : -fill.Quantity;
            int newQuantity = Quantity + fillQuantity;

            if (newQuantity == 0)
            {
                // Position closed
                RealizedPnL += CalculateRealizedPnL(fill);
                Quantity = 0;
                AverageCost = 0;
            }
            else if ((Quantity >= 0 && fillQuantity > 0) || (Quantity <= 0 && fillQuantity < 0))
            {
                // Adding to position
                decimal totalCost = (Math.Abs(Quantity) * AverageCost) + (fill.Quantity * fill.Price);
                int totalQuantity = Math.Abs(Quantity) + fill.Quantity;
                AverageCost = totalQuantity > 0 ? totalCost / totalQuantity : 0;
                Quantity = newQuantity;
            }
            else if (Math.Abs(fillQuantity) <= Math.Abs(Quantity))
            {
                // Partial close
                RealizedPnL += CalculateRealizedPnL(fill);
                Quantity = newQuantity;
            }
            else
            {
                // Flip position (close and open in opposite direction)
                int closeQuantity = Math.Abs(Quantity);
                int openQuantity = Math.Abs(fillQuantity) - closeQuantity;
                
                RealizedPnL += (fill.Price - AverageCost) * closeQuantity * (Quantity > 0 ? 1 : -1);
                
                Quantity = newQuantity;
                AverageCost = fill.Price;
            }

            LastUpdateTime = DateTime.UtcNow;
        }

        private decimal CalculateRealizedPnL(OrderFill fill)
        {
            if (Quantity > 0)
            {
                // Closing long position
                return (fill.Price - AverageCost) * fill.Quantity;
            }
            else
            {
                // Closing short position
                return (AverageCost - fill.Price) * fill.Quantity;
            }
        }

        private void UpdateUnrealizedPnL()
        {
            if (Quantity == 0 || CurrentPrice == 0)
            {
                UnrealizedPnL = 0;
            }
            else if (Quantity > 0)
            {
                // Long position
                UnrealizedPnL = (CurrentPrice - AverageCost) * Quantity;
            }
            else
            {
                // Short position
                UnrealizedPnL = (AverageCost - CurrentPrice) * Math.Abs(Quantity);
            }
        }

        protected bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (Equals(field, value)) return false;
            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
