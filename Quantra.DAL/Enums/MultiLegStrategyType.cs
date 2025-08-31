using System;

namespace Quantra.Enums
{
    /// <summary>
    /// Defines the types of multi-leg strategies that can be executed
    /// </summary>
    public enum MultiLegStrategyType
    {
        /// <summary>
        /// Custom strategy with user-defined legs
        /// </summary>
        Custom = 0,
        
        /// <summary>
        /// Standard vertical spread (bull call spread or bear put spread)
        /// </summary>
        VerticalSpread = 1,
        
        /// <summary>
        /// Standard horizontal spread (calendar spread)
        /// </summary>
        CalendarSpread = 2,
        
        /// <summary>
        /// Diagonal spread combining elements of vertical and horizontal spreads
        /// </summary>
        DiagonalSpread = 3,
        
        /// <summary>
        /// Straddle - put and call at same strike price and expiration
        /// </summary>
        Straddle = 4,
        
        /// <summary>
        /// Strangle - put and call at different strike prices but same expiration
        /// </summary>
        Strangle = 5,
        
        /// <summary>
        /// Iron Condor - combination of bull put spread and bear call spread
        /// </summary>
        IronCondor = 6,
        
        /// <summary>
        /// Butterfly spread - three strike prices with ratio 1:2:1
        /// </summary>
        ButterflySpread = 7,
        
        /// <summary>
        /// Covered call - long stock with short call
        /// </summary>
        CoveredCall = 8,
        
        /// <summary>
        /// Pairs trade - long one security and short a correlated security
        /// </summary>
        PairsTrade = 9,
        
        /// <summary>
        /// Basket order - multiple related securities bought or sold simultaneously
        /// </summary>
        BasketOrder = 10
    }
}