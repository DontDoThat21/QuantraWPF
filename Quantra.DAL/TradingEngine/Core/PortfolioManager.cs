using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.TradingEngine.Data;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.DAL.TradingEngine.Positions;

namespace Quantra.DAL.TradingEngine.Core
{
    /// <summary>
    /// Manages portfolio positions, cash, and P&L
    /// </summary>
    public class PortfolioManager : IPortfolioManager
    {
        private readonly Dictionary<string, TradingPosition> _positions;
        private readonly List<PortfolioSnapshot> _snapshots;
        private decimal _cashBalance;
        private decimal _realizedPnL;
        private decimal _initialCash;

        public event EventHandler<PortfolioChangedEventArgs>? PortfolioChanged;

        /// <summary>
        /// Creates a new portfolio manager with initial cash
        /// </summary>
        public PortfolioManager(decimal initialCash = 100000m)
        {
            _positions = new Dictionary<string, TradingPosition>(StringComparer.OrdinalIgnoreCase);
            _snapshots = new List<PortfolioSnapshot>();
            _cashBalance = initialCash;
            _initialCash = initialCash;
            _realizedPnL = 0;
        }

        /// <summary>
        /// Gets the current cash balance
        /// </summary>
        public decimal CashBalance => _cashBalance;

        /// <summary>
        /// Gets the total unrealized P&L
        /// </summary>
        public decimal UnrealizedPnL
        {
            get
            {
                decimal total = 0;
                foreach (var position in _positions.Values)
                {
                    total += position.UnrealizedPnL;
                }
                return total;
            }
        }

        /// <summary>
        /// Gets the total realized P&L
        /// </summary>
        public decimal RealizedPnL => _realizedPnL;

        /// <summary>
        /// Gets the total value of positions
        /// </summary>
        public decimal PositionsValue
        {
            get
            {
                decimal total = 0;
                foreach (var position in _positions.Values)
                {
                    total += position.MarketValue;
                }
                return total;
            }
        }

        /// <summary>
        /// Gets the total portfolio value
        /// </summary>
        public decimal TotalValue => CashBalance + PositionsValue;

        /// <summary>
        /// Gets all current positions
        /// </summary>
        public IReadOnlyDictionary<string, TradingPosition> Positions => _positions;

        /// <summary>
        /// Gets buying power (simplified: just cash for now)
        /// </summary>
        public decimal BuyingPower => _cashBalance;

        /// <summary>
        /// Gets a position for a symbol
        /// </summary>
        public TradingPosition? GetPosition(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                return null;
            }

            _positions.TryGetValue(symbol, out var position);
            return position;
        }

        /// <summary>
        /// Processes a fill and updates positions/cash
        /// </summary>
        public void ProcessFill(OrderFill fill)
        {
            if (fill == null)
            {
                return;
            }

            decimal oldValue = TotalValue;

            // Get or create position
            if (!_positions.TryGetValue(fill.Symbol, out var position))
            {
                position = new TradingPosition
                {
                    Symbol = fill.Symbol,
                    CurrentPrice = fill.Price
                };
                _positions[fill.Symbol] = position;
            }

            // Track realized P&L before the fill
            decimal previousRealizedPnL = position.RealizedPnL;

            // Update position with the fill
            position.AddFill(fill);

            // Update realized P&L
            decimal fillRealizedPnL = position.RealizedPnL - previousRealizedPnL;
            _realizedPnL += fillRealizedPnL;

            // Update cash
            if (fill.Side == OrderSide.Buy)
            {
                _cashBalance -= (fill.Quantity * fill.Price) + fill.Commission;
            }
            else
            {
                _cashBalance += (fill.Quantity * fill.Price) - fill.Commission;
            }

            // Remove flat positions
            if (position.IsFlat)
            {
                _positions.Remove(fill.Symbol);
            }

            // Raise event
            PortfolioChanged?.Invoke(this, new PortfolioChangedEventArgs
            {
                OldValue = oldValue,
                NewValue = TotalValue,
                Reason = $"Fill: {fill.Side} {fill.Quantity} {fill.Symbol} @ {fill.Price}",
                Time = fill.FillTime
            });
        }

        /// <summary>
        /// Updates all positions with current market prices
        /// </summary>
        public async Task UpdatePricesAsync(IDataSource dataSource, DateTime time)
        {
            if (dataSource == null)
            {
                return;
            }

            decimal oldValue = TotalValue;

            foreach (var position in _positions.Values)
            {
                var quote = await dataSource.GetQuoteAsync(position.Symbol, time);
                if (quote != null)
                {
                    position.CurrentPrice = quote.Last;
                    position.LastUpdateTime = time;
                }
            }

            decimal newValue = TotalValue;

            if (oldValue != newValue)
            {
                PortfolioChanged?.Invoke(this, new PortfolioChangedEventArgs
                {
                    OldValue = oldValue,
                    NewValue = newValue,
                    Reason = "Price update",
                    Time = time
                });
            }
        }

        /// <summary>
        /// Takes a snapshot of the current portfolio state
        /// </summary>
        public PortfolioSnapshot TakeSnapshot(DateTime time)
        {
            var snapshot = new PortfolioSnapshot
            {
                Time = time,
                CashBalance = _cashBalance,
                PositionsValue = PositionsValue,
                TotalValue = TotalValue,
                UnrealizedPnL = UnrealizedPnL,
                RealizedPnL = RealizedPnL
            };

            foreach (var kvp in _positions)
            {
                snapshot.Positions[kvp.Key] = new PositionSnapshot
                {
                    Symbol = kvp.Value.Symbol,
                    Quantity = kvp.Value.Quantity,
                    AverageCost = kvp.Value.AverageCost,
                    CurrentPrice = kvp.Value.CurrentPrice,
                    MarketValue = kvp.Value.MarketValue,
                    UnrealizedPnL = kvp.Value.UnrealizedPnL
                };
            }

            _snapshots.Add(snapshot);
            return snapshot;
        }

        /// <summary>
        /// Gets all portfolio snapshots
        /// </summary>
        public IReadOnlyList<PortfolioSnapshot> GetSnapshots()
        {
            return _snapshots.AsReadOnly();
        }

        /// <summary>
        /// Resets the portfolio to initial state
        /// </summary>
        public void Reset(decimal initialCash)
        {
            _positions.Clear();
            _snapshots.Clear();
            _cashBalance = initialCash;
            _initialCash = initialCash;
            _realizedPnL = 0;
        }

        /// <summary>
        /// Restores a position from persistence (for app restart recovery)
        /// </summary>
        /// <param name="position">The position to restore</param>
        public void RestorePosition(TradingPosition position)
        {
            if (position == null || string.IsNullOrEmpty(position.Symbol))
            {
                return;
            }

            _positions[position.Symbol] = position;
        }

        /// <summary>
        /// Sets the realized P&L (for restoration from persistence)
        /// </summary>
        /// <param name="realizedPnL">The realized P&L value to set</param>
        public void SetRealizedPnL(decimal realizedPnL)
        {
            _realizedPnL = realizedPnL;
        }

        /// <summary>
        /// Gets portfolio performance metrics
        /// </summary>
        public PortfolioPerformance GetPerformance()
        {
            decimal totalReturn = _initialCash > 0 ? (TotalValue - _initialCash) / _initialCash : 0;

            return new PortfolioPerformance
            {
                InitialValue = _initialCash,
                CurrentValue = TotalValue,
                TotalReturn = totalReturn,
                TotalReturnPercent = totalReturn * 100,
                RealizedPnL = _realizedPnL,
                UnrealizedPnL = UnrealizedPnL,
                PositionCount = _positions.Count
            };
        }
    }

    /// <summary>
    /// Portfolio performance metrics
    /// </summary>
    public class PortfolioPerformance
    {
        public decimal InitialValue { get; set; }
        public decimal CurrentValue { get; set; }
        public decimal TotalReturn { get; set; }
        public decimal TotalReturnPercent { get; set; }
        public decimal RealizedPnL { get; set; }
        public decimal UnrealizedPnL { get; set; }
        public int PositionCount { get; set; }
    }
}
