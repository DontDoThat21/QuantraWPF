using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.TradingEngine.Data;
using Quantra.DAL.TradingEngine.Execution;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.DAL.TradingEngine.Positions;
using Quantra.DAL.TradingEngine.Time;
using Quantra.Models;

namespace Quantra.DAL.TradingEngine.Core
{
    /// <summary>
    /// Unified trading engine supporting both paper trading and backtesting.
    /// Uses pluggable data sources and clock implementations.
    /// </summary>
    public class TradingEngine : ITradingEngine
    {
        private IDataSource? _dataSource;
        private IClock? _clock;
        private IPortfolioManager _portfolio;
        private IFillSimulator _fillSimulator;

        private readonly ConcurrentDictionary<Guid, Order> _orders;
        private readonly List<OrderFill> _fills;
        private bool _isRunning;
        private bool _isInitialized;

        public event EventHandler<OrderFilledEventArgs>? OrderFilled;
        public event EventHandler<OrderStateChangedEventArgs>? OrderStateChanged;

        /// <summary>
        /// Creates a new trading engine with optional cost model
        /// </summary>
        public TradingEngine(TransactionCostModel? costModel = null)
        {
            _orders = new ConcurrentDictionary<Guid, Order>();
            _fills = new List<OrderFill>();
            _fillSimulator = new FillSimulator(costModel);
            _portfolio = new PortfolioManager();
            _isRunning = false;
            _isInitialized = false;
        }

        /// <summary>
        /// Creates a new trading engine with all dependencies
        /// </summary>
        public TradingEngine(
            IDataSource dataSource, 
            IClock clock, 
            IPortfolioManager portfolio,
            TransactionCostModel? costModel = null) : this(costModel)
        {
            Initialize(dataSource, clock, portfolio);
        }

        /// <summary>
        /// Gets whether the engine is running
        /// </summary>
        public bool IsRunning => _isRunning;

        /// <summary>
        /// Gets whether this is paper trading (uses simulated clock)
        /// </summary>
        public bool IsPaperTrading => _clock != null && !_clock.IsSimulated;

        /// <summary>
        /// Initializes the trading engine with components
        /// </summary>
        public void Initialize(IDataSource dataSource, IClock clock, IPortfolioManager portfolio)
        {
            _dataSource = dataSource ?? throw new ArgumentNullException(nameof(dataSource));
            _clock = clock ?? throw new ArgumentNullException(nameof(clock));
            _portfolio = portfolio ?? throw new ArgumentNullException(nameof(portfolio));

            // Subscribe to clock time changes
            _clock.TimeChanged += OnTimeChanged;

            _isInitialized = true;
        }

        /// <summary>
        /// Places a new order
        /// </summary>
        public async Task<Guid> PlaceOrderAsync(Order order)
        {
            if (order == null)
            {
                throw new ArgumentNullException(nameof(order));
            }

            if (!_isInitialized)
            {
                throw new InvalidOperationException("Trading engine not initialized");
            }

            // Validate order
            var validationResult = ValidateOrder(order);
            if (!validationResult.IsValid)
            {
                order.State = OrderState.Rejected;
                order.RejectReason = validationResult.Reason;
                RaiseOrderStateChanged(order, OrderState.Pending, OrderState.Rejected, validationResult.Reason);
                return order.Id;
            }

            // Set expiration time for Day orders
            if (order.TimeInForce == TimeInForce.Day && !order.ExpirationTime.HasValue)
            {
                var currentTime = _clock?.CurrentTime ?? DateTime.UtcNow;
                // Set expiration to 4 PM ET of the current trading day
                var expirationTime = currentTime.Date.AddHours(16);
                // If it's already past 4 PM, set expiration to next trading day 4 PM
                if (currentTime.Hour >= 16)
                {
                    expirationTime = expirationTime.AddDays(1);
                }
                order.ExpirationTime = expirationTime;
            }

            // Add order to tracking
            order.State = OrderState.Submitted;
            order.SubmittedTime = _clock?.CurrentTime ?? DateTime.UtcNow;
            _orders[order.Id] = order;

            RaiseOrderStateChanged(order, OrderState.Pending, OrderState.Submitted, "Order submitted");

            // Try immediate fill for market orders
            if (order.OrderType == OrderType.Market && _dataSource != null && _clock != null)
            {
                await TryFillOrderAsync(order, _clock.CurrentTime);
            }

            return order.Id;
        }

        /// <summary>
        /// Cancels an existing order
        /// </summary>
        public bool CancelOrder(Guid orderId)
        {
            if (!_orders.TryGetValue(orderId, out var order))
            {
                return false;
            }

            if (!order.CanCancel)
            {
                return false;
            }

            var oldState = order.State;
            order.State = OrderState.Cancelled;

            RaiseOrderStateChanged(order, oldState, OrderState.Cancelled, "Order cancelled by user");

            return true;
        }

        /// <summary>
        /// Modifies an existing order
        /// </summary>
        public bool ModifyOrder(Guid orderId, decimal? newPrice, int? newQuantity)
        {
            if (!_orders.TryGetValue(orderId, out var order))
            {
                return false;
            }

            if (!order.CanModify)
            {
                return false;
            }

            if (newPrice.HasValue)
            {
                if (order.OrderType == OrderType.Limit || order.OrderType == OrderType.StopLimit)
                {
                    order.LimitPrice = newPrice.Value;
                }
                else if (order.OrderType == OrderType.Stop || order.OrderType == OrderType.TrailingStop)
                {
                    order.StopPrice = newPrice.Value;
                }
            }

            if (newQuantity.HasValue && newQuantity.Value > order.FilledQuantity)
            {
                order.Quantity = newQuantity.Value;
            }

            return true;
        }

        /// <summary>
        /// Gets an order by ID
        /// </summary>
        public Order? GetOrder(Guid orderId)
        {
            _orders.TryGetValue(orderId, out var order);
            return order;
        }

        /// <summary>
        /// Gets all active orders
        /// </summary>
        public IEnumerable<Order> GetActiveOrders()
        {
            return _orders.Values.Where(o => !o.IsTerminal).ToList();
        }

        /// <summary>
        /// Gets all orders
        /// </summary>
        public IEnumerable<Order> GetAllOrders()
        {
            return _orders.Values.ToList();
        }

        /// <summary>
        /// Gets all positions
        /// </summary>
        public IEnumerable<TradingPosition> GetPositions()
        {
            return _portfolio.Positions.Values.ToList();
        }

        /// <summary>
        /// Gets the portfolio manager
        /// </summary>
        public IPortfolioManager GetPortfolio()
        {
            return _portfolio;
        }

        /// <summary>
        /// Gets the data source
        /// </summary>
        public IDataSource? GetDataSource()
        {
            return _dataSource;
        }

        /// <summary>
        /// Gets the clock
        /// </summary>
        public IClock? GetClock()
        {
            return _clock;
        }

        /// <summary>
        /// Processes a time step (for backtesting)
        /// </summary>
        public async Task ProcessTimeStepAsync(DateTime time)
        {
            if (_dataSource == null)
            {
                return;
            }

            // Update portfolio prices
            await _portfolio.UpdatePricesAsync(_dataSource, time);

            // Check all active orders for fills
            var activeOrders = GetActiveOrders().ToList();
            foreach (var order in activeOrders)
            {
                await TryFillOrderAsync(order, time);
            }

            // Check for expired orders
            CheckOrderExpirations(time);
        }

        /// <summary>
        /// Starts the engine
        /// </summary>
        public void Start()
        {
            if (!_isInitialized)
            {
                throw new InvalidOperationException("Trading engine not initialized");
            }

            _clock?.Start();
            _isRunning = true;
        }

        /// <summary>
        /// Stops the engine
        /// </summary>
        public void Stop()
        {
            _clock?.Stop();
            _isRunning = false;
        }

        private async Task TryFillOrderAsync(Order order, DateTime time)
        {
            if (_dataSource == null || order.IsTerminal)
            {
                return;
            }

            var quote = await _dataSource.GetQuoteAsync(order.Symbol, time);
            if (quote == null)
            {
                return;
            }

            var fillResult = _fillSimulator.TryFill(order, quote, time);

            if (fillResult.IsFilled)
            {
                var fill = new OrderFill
                {
                    OrderId = order.Id,
                    Symbol = order.Symbol,
                    Quantity = fillResult.FilledQuantity,
                    Price = fillResult.ExecutionPrice,
                    Side = order.Side,
                    FillTime = fillResult.FillTime,
                    Commission = fillResult.Commission,
                    Slippage = fillResult.Slippage,
                    IsPaperTrade = order.IsPaperTrade
                };

                _fills.Add(fill);

                // Update order
                var oldState = order.State;
                order.FilledQuantity += fillResult.FilledQuantity;
                order.AvgFillPrice = fillResult.ExecutionPrice;

                if (order.FilledQuantity >= order.Quantity)
                {
                    order.State = OrderState.Filled;
                    order.FilledTime = time;
                }
                else
                {
                    order.State = OrderState.PartiallyFilled;
                }

                // Update portfolio
                _portfolio.ProcessFill(fill);

                // Raise events
                OrderFilled?.Invoke(this, new OrderFilledEventArgs
                {
                    Order = order,
                    Fill = fill,
                    Time = time
                });

                if (oldState != order.State)
                {
                    RaiseOrderStateChanged(order, oldState, order.State, "Order filled");
                }
            }
        }

        private void CheckOrderExpirations(DateTime time)
        {
            var activeOrders = GetActiveOrders().ToList();
            foreach (var order in activeOrders)
            {
                bool shouldExpire = false;

                switch (order.TimeInForce)
                {
                    case TimeInForce.Day:
                        // Only expire if past the explicit expiration time
                        // Don't use simple hour comparison which causes immediate expiration
                        break;

                    case TimeInForce.IOC:
                    case TimeInForce.FOK:
                        // These should have been filled immediately or cancelled
                        if (order.State == OrderState.Submitted)
                        {
                            shouldExpire = true;
                        }
                        break;
                }

                // Check explicit expiration time
                if (order.ExpirationTime.HasValue && time >= order.ExpirationTime.Value)
                {
                    shouldExpire = true;
                }

                if (shouldExpire)
                {
                    var oldState = order.State;
                    order.State = OrderState.Expired;
                    RaiseOrderStateChanged(order, oldState, OrderState.Expired, "Order expired");
                }
            }
        }

        private (bool IsValid, string Reason) ValidateOrder(Order order)
        {
            if (string.IsNullOrEmpty(order.Symbol))
            {
                return (false, "Symbol is required");
            }

            if (order.Quantity <= 0)
            {
                return (false, "Quantity must be positive");
            }

            if (order.OrderType == OrderType.Limit && order.LimitPrice <= 0)
            {
                return (false, "Limit price must be positive for limit orders");
            }

            if ((order.OrderType == OrderType.Stop || order.OrderType == OrderType.StopLimit) && order.StopPrice <= 0)
            {
                return (false, "Stop price must be positive for stop orders");
            }

            // Check buying power for buy orders
            if (order.Side == OrderSide.Buy)
            {
                decimal estimatedCost = order.Quantity * (order.LimitPrice > 0 ? order.LimitPrice : 1000); // Rough estimate
                if (estimatedCost > _portfolio.BuyingPower)
                {
                    return (false, "Insufficient buying power");
                }
            }

            return (true, string.Empty);
        }

        private void OnTimeChanged(object? sender, DateTime time)
        {
            // Process time step when clock advances
            _ = ProcessTimeStepAsync(time);
        }

        private void RaiseOrderStateChanged(Order order, OrderState oldState, OrderState newState, string reason)
        {
            OrderStateChanged?.Invoke(this, new OrderStateChangedEventArgs
            {
                Order = order,
                OldState = oldState,
                NewState = newState,
                Time = _clock?.CurrentTime ?? DateTime.UtcNow,
                Reason = reason
            });
        }

        /// <summary>
        /// Gets all fills
        /// </summary>
        public IReadOnlyList<OrderFill> GetFills()
        {
            return _fills.AsReadOnly();
        }

        /// <summary>
        /// Clears all orders and fills (for reset)
        /// </summary>
        public void Reset()
        {
            _orders.Clear();
            _fills.Clear();
        }
    }
}
