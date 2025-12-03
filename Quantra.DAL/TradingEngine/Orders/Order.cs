using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Quantra.DAL.TradingEngine.Orders
{
    /// <summary>
    /// Base class for all order types in the trading engine.
    /// Supports stocks, ETFs, and options with full lifecycle management.
    /// </summary>
    public class Order : INotifyPropertyChanged
    {
        private Guid _id;
        private string _symbol = string.Empty;
        private OrderType _orderType;
        private OrderSide _side;
        private OrderState _state;
        private int _quantity;
        private int _filledQuantity;
        private decimal _limitPrice;
        private decimal _stopPrice;
        private decimal _avgFillPrice;
        private TimeInForce _timeInForce;
        private AssetType _assetType;
        private DateTime _createdTime;
        private DateTime? _submittedTime;
        private DateTime? _filledTime;
        private DateTime? _expirationTime;
        private string _rejectReason = string.Empty;
        private bool _isPaperTrade;
        private string _notes = string.Empty;

        public event PropertyChangedEventHandler? PropertyChanged;

        /// <summary>
        /// Creates a new order with a unique identifier
        /// </summary>
        public Order()
        {
            _id = Guid.NewGuid();
            _createdTime = DateTime.UtcNow;
            _state = OrderState.Pending;
            _timeInForce = TimeInForce.Day;
            _assetType = AssetType.Stock;
            _isPaperTrade = true; // Default to paper trading
        }

        /// <summary>
        /// Unique identifier for this order
        /// </summary>
        public Guid Id
        {
            get => _id;
            set => SetProperty(ref _id, value);
        }

        /// <summary>
        /// Symbol of the asset being traded (e.g., AAPL, SPY)
        /// </summary>
        public string Symbol
        {
            get => _symbol;
            set => SetProperty(ref _symbol, value);
        }

        /// <summary>
        /// Type of order (Market, Limit, Stop, etc.)
        /// </summary>
        public OrderType OrderType
        {
            get => _orderType;
            set => SetProperty(ref _orderType, value);
        }

        /// <summary>
        /// Order direction (Buy or Sell)
        /// </summary>
        public OrderSide Side
        {
            get => _side;
            set => SetProperty(ref _side, value);
        }

        /// <summary>
        /// Current state of the order
        /// </summary>
        public OrderState State
        {
            get => _state;
            set => SetProperty(ref _state, value);
        }

        /// <summary>
        /// Total quantity requested
        /// </summary>
        public int Quantity
        {
            get => _quantity;
            set => SetProperty(ref _quantity, value);
        }

        /// <summary>
        /// Quantity that has been filled
        /// </summary>
        public int FilledQuantity
        {
            get => _filledQuantity;
            set
            {
                if (SetProperty(ref _filledQuantity, value))
                {
                    OnPropertyChanged(nameof(RemainingQuantity));
                    OnPropertyChanged(nameof(FillPercentage));
                }
            }
        }

        /// <summary>
        /// Remaining quantity to be filled
        /// </summary>
        public int RemainingQuantity => Quantity - FilledQuantity;

        /// <summary>
        /// Percentage of order that has been filled
        /// </summary>
        public decimal FillPercentage => Quantity > 0 ? (decimal)FilledQuantity / Quantity * 100 : 0;

        /// <summary>
        /// Limit price for limit and stop-limit orders
        /// </summary>
        public decimal LimitPrice
        {
            get => _limitPrice;
            set => SetProperty(ref _limitPrice, value);
        }

        /// <summary>
        /// Stop price for stop and stop-limit orders
        /// </summary>
        public decimal StopPrice
        {
            get => _stopPrice;
            set => SetProperty(ref _stopPrice, value);
        }

        /// <summary>
        /// Average fill price for filled/partially filled orders
        /// </summary>
        public decimal AvgFillPrice
        {
            get => _avgFillPrice;
            set => SetProperty(ref _avgFillPrice, value);
        }

        /// <summary>
        /// Time-in-force specification
        /// </summary>
        public TimeInForce TimeInForce
        {
            get => _timeInForce;
            set => SetProperty(ref _timeInForce, value);
        }

        /// <summary>
        /// Type of asset (Stock, ETF, Option)
        /// </summary>
        public AssetType AssetType
        {
            get => _assetType;
            set => SetProperty(ref _assetType, value);
        }

        /// <summary>
        /// Time the order was created
        /// </summary>
        public DateTime CreatedTime
        {
            get => _createdTime;
            set => SetProperty(ref _createdTime, value);
        }

        /// <summary>
        /// Time the order was submitted
        /// </summary>
        public DateTime? SubmittedTime
        {
            get => _submittedTime;
            set => SetProperty(ref _submittedTime, value);
        }

        /// <summary>
        /// Time the order was completely filled
        /// </summary>
        public DateTime? FilledTime
        {
            get => _filledTime;
            set => SetProperty(ref _filledTime, value);
        }

        /// <summary>
        /// Expiration time for GTC orders or EOD for day orders
        /// </summary>
        public DateTime? ExpirationTime
        {
            get => _expirationTime;
            set => SetProperty(ref _expirationTime, value);
        }

        /// <summary>
        /// Reason for rejection if order was rejected
        /// </summary>
        public string RejectReason
        {
            get => _rejectReason;
            set => SetProperty(ref _rejectReason, value);
        }

        /// <summary>
        /// Indicates if this is a paper trade (simulation)
        /// </summary>
        public bool IsPaperTrade
        {
            get => _isPaperTrade;
            set => SetProperty(ref _isPaperTrade, value);
        }

        /// <summary>
        /// Optional notes or comments about the order
        /// </summary>
        public string Notes
        {
            get => _notes;
            set => SetProperty(ref _notes, value);
        }

        /// <summary>
        /// Checks if the order can be modified
        /// </summary>
        public bool CanModify => State == OrderState.Pending || State == OrderState.Submitted;

        /// <summary>
        /// Checks if the order can be cancelled
        /// </summary>
        public bool CanCancel => State == OrderState.Pending || State == OrderState.Submitted || State == OrderState.PartiallyFilled;

        /// <summary>
        /// Checks if the order is in a terminal state
        /// </summary>
        public bool IsTerminal => State == OrderState.Filled || State == OrderState.Cancelled || 
                                   State == OrderState.Expired || State == OrderState.Rejected;

        /// <summary>
        /// Creates a market order
        /// </summary>
        public static Order CreateMarketOrder(string symbol, OrderSide side, int quantity)
        {
            return new Order
            {
                Symbol = symbol,
                OrderType = OrderType.Market,
                Side = side,
                Quantity = quantity
            };
        }

        /// <summary>
        /// Creates a limit order
        /// </summary>
        public static Order CreateLimitOrder(string symbol, OrderSide side, int quantity, decimal limitPrice)
        {
            return new Order
            {
                Symbol = symbol,
                OrderType = OrderType.Limit,
                Side = side,
                Quantity = quantity,
                LimitPrice = limitPrice
            };
        }

        /// <summary>
        /// Creates a stop order
        /// </summary>
        public static Order CreateStopOrder(string symbol, OrderSide side, int quantity, decimal stopPrice)
        {
            return new Order
            {
                Symbol = symbol,
                OrderType = OrderType.Stop,
                Side = side,
                Quantity = quantity,
                StopPrice = stopPrice
            };
        }

        /// <summary>
        /// Creates a stop-limit order
        /// </summary>
        public static Order CreateStopLimitOrder(string symbol, OrderSide side, int quantity, decimal stopPrice, decimal limitPrice)
        {
            return new Order
            {
                Symbol = symbol,
                OrderType = OrderType.StopLimit,
                Side = side,
                Quantity = quantity,
                StopPrice = stopPrice,
                LimitPrice = limitPrice
            };
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
