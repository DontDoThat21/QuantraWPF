using System;

namespace Quantra.DAL.TradingEngine.Orders
{
    /// <summary>
    /// Trailing stop order - stop price adjusts dynamically as market moves
    /// </summary>
    public class TrailingStopOrder : Order
    {
        private decimal _trailAmount;
        private decimal _trailPercent;
        private bool _usePercentage;
        private decimal _highWaterMark;
        private decimal _lowWaterMark;

        public TrailingStopOrder()
        {
            OrderType = OrderType.TrailingStop;
        }

        /// <summary>
        /// Fixed trail amount in dollars
        /// </summary>
        public decimal TrailAmount
        {
            get => _trailAmount;
            set
            {
                _trailAmount = value;
                _usePercentage = false;
                OnPropertyChanged();
            }
        }

        /// <summary>
        /// Trail amount as percentage (e.g., 0.05 for 5%)
        /// </summary>
        public decimal TrailPercent
        {
            get => _trailPercent;
            set
            {
                _trailPercent = value;
                _usePercentage = true;
                OnPropertyChanged();
            }
        }

        /// <summary>
        /// Whether to use percentage-based trailing
        /// </summary>
        public bool UsePercentage
        {
            get => _usePercentage;
            set
            {
                _usePercentage = value;
                OnPropertyChanged();
            }
        }

        /// <summary>
        /// Highest price reached since order was placed (for sell trailing stops)
        /// </summary>
        public decimal HighWaterMark
        {
            get => _highWaterMark;
            set
            {
                _highWaterMark = value;
                OnPropertyChanged();
            }
        }

        /// <summary>
        /// Lowest price reached since order was placed (for buy trailing stops)
        /// </summary>
        public decimal LowWaterMark
        {
            get => _lowWaterMark;
            set
            {
                _lowWaterMark = value;
                OnPropertyChanged();
            }
        }

        /// <summary>
        /// Updates the stop price based on current market price
        /// </summary>
        /// <param name="currentPrice">Current market price</param>
        /// <returns>True if stop was triggered, false otherwise</returns>
        public bool UpdateStopPrice(decimal currentPrice)
        {
            if (Side == OrderSide.Sell)
            {
                // For sell trailing stop: track high water mark, set stop below it
                if (currentPrice > HighWaterMark)
                {
                    HighWaterMark = currentPrice;
                    decimal trail = UsePercentage ? currentPrice * TrailPercent : TrailAmount;
                    StopPrice = HighWaterMark - trail;
                }

                // Check if stop was triggered
                return currentPrice <= StopPrice && StopPrice > 0;
            }
            else
            {
                // For buy trailing stop: track low water mark, set stop above it
                if (currentPrice < LowWaterMark || LowWaterMark == 0)
                {
                    LowWaterMark = currentPrice;
                    decimal trail = UsePercentage ? currentPrice * TrailPercent : TrailAmount;
                    StopPrice = LowWaterMark + trail;
                }

                // Check if stop was triggered
                return currentPrice >= StopPrice && StopPrice > 0;
            }
        }

        /// <summary>
        /// Creates a sell trailing stop order with fixed dollar amount
        /// </summary>
        public static TrailingStopOrder CreateSellTrailingStop(string symbol, int quantity, decimal trailAmount, decimal currentPrice)
        {
            return new TrailingStopOrder
            {
                Symbol = symbol,
                Side = OrderSide.Sell,
                Quantity = quantity,
                TrailAmount = trailAmount,
                UsePercentage = false,
                HighWaterMark = currentPrice,
                StopPrice = currentPrice - trailAmount
            };
        }

        /// <summary>
        /// Creates a sell trailing stop order with percentage trail
        /// </summary>
        public static TrailingStopOrder CreateSellTrailingStopPercent(string symbol, int quantity, decimal trailPercent, decimal currentPrice)
        {
            return new TrailingStopOrder
            {
                Symbol = symbol,
                Side = OrderSide.Sell,
                Quantity = quantity,
                TrailPercent = trailPercent,
                UsePercentage = true,
                HighWaterMark = currentPrice,
                StopPrice = currentPrice * (1 - trailPercent)
            };
        }

        /// <summary>
        /// Creates a buy trailing stop order with fixed dollar amount
        /// </summary>
        public static TrailingStopOrder CreateBuyTrailingStop(string symbol, int quantity, decimal trailAmount, decimal currentPrice)
        {
            return new TrailingStopOrder
            {
                Symbol = symbol,
                Side = OrderSide.Buy,
                Quantity = quantity,
                TrailAmount = trailAmount,
                UsePercentage = false,
                LowWaterMark = currentPrice,
                StopPrice = currentPrice + trailAmount
            };
        }

        /// <summary>
        /// Creates a buy trailing stop order with percentage trail
        /// </summary>
        public static TrailingStopOrder CreateBuyTrailingStopPercent(string symbol, int quantity, decimal trailPercent, decimal currentPrice)
        {
            return new TrailingStopOrder
            {
                Symbol = symbol,
                Side = OrderSide.Buy,
                Quantity = quantity,
                TrailPercent = trailPercent,
                UsePercentage = true,
                LowWaterMark = currentPrice,
                StopPrice = currentPrice * (1 + trailPercent)
            };
        }
    }
}
