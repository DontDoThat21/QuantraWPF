using System;

namespace Quantra.DAL.TradingEngine.Orders
{
    /// <summary>
    /// Types of orders supported by the trading engine
    /// </summary>
    public enum OrderType
    {
        /// <summary>
        /// Market order - executes immediately at best available price
        /// </summary>
        Market,

        /// <summary>
        /// Limit order - executes at specified price or better
        /// </summary>
        Limit,

        /// <summary>
        /// Stop order - converts to market order when stop price is breached
        /// </summary>
        Stop,

        /// <summary>
        /// Stop-limit order - converts to limit order when stop price is breached
        /// </summary>
        StopLimit,

        /// <summary>
        /// Trailing stop order - stop price adjusts dynamically with market
        /// </summary>
        TrailingStop
    }

    /// <summary>
    /// States an order can be in during its lifecycle
    /// </summary>
    public enum OrderState
    {
        /// <summary>
        /// Order created but not yet submitted
        /// </summary>
        Pending,

        /// <summary>
        /// Order submitted to the exchange/broker
        /// </summary>
        Submitted,

        /// <summary>
        /// Order has been partially filled
        /// </summary>
        PartiallyFilled,

        /// <summary>
        /// Order has been completely filled
        /// </summary>
        Filled,

        /// <summary>
        /// Order has been cancelled
        /// </summary>
        Cancelled,

        /// <summary>
        /// Order has expired (e.g., day order at market close)
        /// </summary>
        Expired,

        /// <summary>
        /// Order was rejected by the exchange/broker
        /// </summary>
        Rejected
    }

    /// <summary>
    /// Time-in-force specifies how long an order remains active
    /// </summary>
    public enum TimeInForce
    {
        /// <summary>
        /// Day order - expires at end of trading day
        /// </summary>
        Day,

        /// <summary>
        /// Good-Til-Cancelled - remains active until filled or cancelled
        /// </summary>
        GTC,

        /// <summary>
        /// Immediate-Or-Cancel - fill immediately or cancel
        /// </summary>
        IOC,

        /// <summary>
        /// Fill-Or-Kill - fill entire order immediately or cancel
        /// </summary>
        FOK
    }

    /// <summary>
    /// Order direction/side
    /// </summary>
    public enum OrderSide
    {
        /// <summary>
        /// Buy order (go long)
        /// </summary>
        Buy,

        /// <summary>
        /// Sell order (close long or go short)
        /// </summary>
        Sell
    }

    /// <summary>
    /// Type of asset being traded
    /// </summary>
    public enum AssetType
    {
        /// <summary>
        /// Common stock
        /// </summary>
        Stock,

        /// <summary>
        /// Exchange-traded fund
        /// </summary>
        ETF,

        /// <summary>
        /// Options contract
        /// </summary>
        Option
    }
}
