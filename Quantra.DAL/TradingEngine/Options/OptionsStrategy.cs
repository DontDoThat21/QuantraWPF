using System;
using System.Collections.Generic;
using Quantra.DAL.TradingEngine.Orders;

namespace Quantra.DAL.TradingEngine.Options
{
    /// <summary>
    /// Type of option
    /// </summary>
    public enum OptionType
    {
        Call,
        Put
    }

    /// <summary>
    /// Types of option strategies
    /// </summary>
    public enum OptionStrategyType
    {
        /// <summary>
        /// Single leg option trade
        /// </summary>
        Single,

        /// <summary>
        /// Covered call - long stock + short call
        /// </summary>
        CoveredCall,

        /// <summary>
        /// Protective put - long stock + long put
        /// </summary>
        ProtectivePut,

        /// <summary>
        /// Bull call spread - long call + short higher strike call
        /// </summary>
        BullCallSpread,

        /// <summary>
        /// Bear put spread - long put + short lower strike put
        /// </summary>
        BearPutSpread,

        /// <summary>
        /// Bull put spread - short put + long lower strike put
        /// </summary>
        BullPutSpread,

        /// <summary>
        /// Bear call spread - short call + long higher strike call
        /// </summary>
        BearCallSpread,

        /// <summary>
        /// Long straddle - long call + long put at same strike
        /// </summary>
        LongStraddle,

        /// <summary>
        /// Short straddle - short call + short put at same strike
        /// </summary>
        ShortStraddle,

        /// <summary>
        /// Long strangle - long call + long put at different strikes
        /// </summary>
        LongStrangle,

        /// <summary>
        /// Short strangle - short call + short put at different strikes
        /// </summary>
        ShortStrangle,

        /// <summary>
        /// Iron condor - short put spread + short call spread
        /// </summary>
        IronCondor,

        /// <summary>
        /// Butterfly spread
        /// </summary>
        Butterfly,

        /// <summary>
        /// Calendar spread - same strike, different expirations
        /// </summary>
        CalendarSpread
    }

    /// <summary>
    /// Represents a single leg of an options strategy
    /// </summary>
    public class OptionsLeg
    {
        /// <summary>
        /// Unique ID for this leg
        /// </summary>
        public Guid Id { get; set; } = Guid.NewGuid();

        /// <summary>
        /// Underlying symbol
        /// </summary>
        public string UnderlyingSymbol { get; set; } = string.Empty;

        /// <summary>
        /// Option type (Call or Put)
        /// </summary>
        public OptionType OptionType { get; set; }

        /// <summary>
        /// Strike price
        /// </summary>
        public decimal StrikePrice { get; set; }

        /// <summary>
        /// Expiration date
        /// </summary>
        public DateTime Expiration { get; set; }

        /// <summary>
        /// Buy or Sell
        /// </summary>
        public OrderSide Side { get; set; }

        /// <summary>
        /// Number of contracts (each contract = 100 shares)
        /// </summary>
        public int Quantity { get; set; }

        /// <summary>
        /// Entry price per contract
        /// </summary>
        public decimal EntryPrice { get; set; }

        /// <summary>
        /// Current price per contract
        /// </summary>
        public decimal CurrentPrice { get; set; }

        /// <summary>
        /// Implied volatility
        /// </summary>
        public decimal ImpliedVolatility { get; set; }

        /// <summary>
        /// Delta of this leg
        /// </summary>
        public decimal Delta { get; set; }

        /// <summary>
        /// Gamma of this leg
        /// </summary>
        public decimal Gamma { get; set; }

        /// <summary>
        /// Theta of this leg (daily decay)
        /// </summary>
        public decimal Theta { get; set; }

        /// <summary>
        /// Vega of this leg
        /// </summary>
        public decimal Vega { get; set; }

        /// <summary>
        /// Rho of this leg
        /// </summary>
        public decimal Rho { get; set; }

        /// <summary>
        /// Contract multiplier (usually 100 for equity options)
        /// </summary>
        public int Multiplier { get; set; } = 100;

        /// <summary>
        /// Gets the option symbol
        /// </summary>
        public string OptionSymbol => $"{UnderlyingSymbol}{Expiration:yyMMdd}{(OptionType == OptionType.Call ? "C" : "P")}{StrikePrice:00000000}";

        /// <summary>
        /// Gets the days to expiration
        /// </summary>
        public int DaysToExpiration => Math.Max(0, (int)(Expiration.Date - DateTime.Today).TotalDays);

        /// <summary>
        /// Gets the notional value
        /// </summary>
        public decimal NotionalValue => CurrentPrice * Quantity * Multiplier;

        /// <summary>
        /// Gets unrealized P&L for this leg
        /// </summary>
        public decimal UnrealizedPnL
        {
            get
            {
                decimal priceDiff = CurrentPrice - EntryPrice;
                decimal pnl = priceDiff * Quantity * Multiplier;
                return Side == OrderSide.Buy ? pnl : -pnl;
            }
        }
    }

    /// <summary>
    /// Represents a multi-leg options strategy
    /// </summary>
    public class OptionsStrategy
    {
        /// <summary>
        /// Unique ID for this strategy
        /// </summary>
        public Guid Id { get; set; } = Guid.NewGuid();

        /// <summary>
        /// Name of the strategy
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Underlying symbol
        /// </summary>
        public string UnderlyingSymbol { get; set; } = string.Empty;

        /// <summary>
        /// Type of strategy
        /// </summary>
        public OptionStrategyType StrategyType { get; set; }

        /// <summary>
        /// Legs that make up this strategy
        /// </summary>
        public List<OptionsLeg> Legs { get; set; } = new List<OptionsLeg>();

        /// <summary>
        /// Time the strategy was opened
        /// </summary>
        public DateTime OpenedTime { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Whether this is a paper trade
        /// </summary>
        public bool IsPaperTrade { get; set; } = true;

        /// <summary>
        /// Notes about the strategy
        /// </summary>
        public string Notes { get; set; } = string.Empty;

        /// <summary>
        /// Gets the net delta of the strategy
        /// </summary>
        public decimal NetDelta
        {
            get
            {
                decimal total = 0;
                foreach (var leg in Legs)
                {
                    decimal legDelta = leg.Delta * leg.Quantity * leg.Multiplier;
                    total += leg.Side == OrderSide.Buy ? legDelta : -legDelta;
                }
                return total;
            }
        }

        /// <summary>
        /// Gets the net gamma of the strategy
        /// </summary>
        public decimal NetGamma
        {
            get
            {
                decimal total = 0;
                foreach (var leg in Legs)
                {
                    decimal legGamma = leg.Gamma * leg.Quantity * leg.Multiplier;
                    total += leg.Side == OrderSide.Buy ? legGamma : -legGamma;
                }
                return total;
            }
        }

        /// <summary>
        /// Gets the net theta of the strategy
        /// </summary>
        public decimal NetTheta
        {
            get
            {
                decimal total = 0;
                foreach (var leg in Legs)
                {
                    decimal legTheta = leg.Theta * leg.Quantity * leg.Multiplier;
                    total += leg.Side == OrderSide.Buy ? legTheta : -legTheta;
                }
                return total;
            }
        }

        /// <summary>
        /// Gets the net vega of the strategy
        /// </summary>
        public decimal NetVega
        {
            get
            {
                decimal total = 0;
                foreach (var leg in Legs)
                {
                    decimal legVega = leg.Vega * leg.Quantity * leg.Multiplier;
                    total += leg.Side == OrderSide.Buy ? legVega : -legVega;
                }
                return total;
            }
        }

        /// <summary>
        /// Gets the net premium (positive = debit, negative = credit)
        /// </summary>
        public decimal NetPremium
        {
            get
            {
                decimal total = 0;
                foreach (var leg in Legs)
                {
                    decimal premium = leg.EntryPrice * leg.Quantity * leg.Multiplier;
                    total += leg.Side == OrderSide.Buy ? premium : -premium;
                }
                return total;
            }
        }

        /// <summary>
        /// Gets the total unrealized P&L
        /// </summary>
        public decimal UnrealizedPnL
        {
            get
            {
                decimal total = 0;
                foreach (var leg in Legs)
                {
                    total += leg.UnrealizedPnL;
                }
                return total;
            }
        }

        /// <summary>
        /// Gets the nearest expiration date
        /// </summary>
        public DateTime? NearestExpiration
        {
            get
            {
                if (Legs.Count == 0) return null;
                DateTime nearest = DateTime.MaxValue;
                foreach (var leg in Legs)
                {
                    if (leg.Expiration < nearest) nearest = leg.Expiration;
                }
                return nearest == DateTime.MaxValue ? null : nearest;
            }
        }

        /// <summary>
        /// Checks if any leg has expired
        /// </summary>
        public bool HasExpiredLegs => Legs.Exists(l => l.DaysToExpiration <= 0);

        /// <summary>
        /// Creates a covered call strategy
        /// </summary>
        public static OptionsStrategy CreateCoveredCall(
            string symbol,
            decimal currentPrice,
            decimal strikePrice,
            DateTime expiration,
            decimal callPremium,
            int contracts = 1)
        {
            return new OptionsStrategy
            {
                Name = $"Covered Call {symbol} {strikePrice:F0}C {expiration:MMM dd}",
                UnderlyingSymbol = symbol,
                StrategyType = OptionStrategyType.CoveredCall,
                Legs = new List<OptionsLeg>
                {
                    new OptionsLeg
                    {
                        UnderlyingSymbol = symbol,
                        OptionType = OptionType.Call,
                        StrikePrice = strikePrice,
                        Expiration = expiration,
                        Side = OrderSide.Sell,
                        Quantity = contracts,
                        EntryPrice = callPremium,
                        CurrentPrice = callPremium
                    }
                }
            };
        }

        /// <summary>
        /// Creates a bull call spread
        /// </summary>
        public static OptionsStrategy CreateBullCallSpread(
            string symbol,
            decimal longStrike,
            decimal shortStrike,
            DateTime expiration,
            decimal longPremium,
            decimal shortPremium,
            int contracts = 1)
        {
            return new OptionsStrategy
            {
                Name = $"Bull Call Spread {symbol} {longStrike:F0}/{shortStrike:F0}C {expiration:MMM dd}",
                UnderlyingSymbol = symbol,
                StrategyType = OptionStrategyType.BullCallSpread,
                Legs = new List<OptionsLeg>
                {
                    new OptionsLeg
                    {
                        UnderlyingSymbol = symbol,
                        OptionType = OptionType.Call,
                        StrikePrice = longStrike,
                        Expiration = expiration,
                        Side = OrderSide.Buy,
                        Quantity = contracts,
                        EntryPrice = longPremium,
                        CurrentPrice = longPremium
                    },
                    new OptionsLeg
                    {
                        UnderlyingSymbol = symbol,
                        OptionType = OptionType.Call,
                        StrikePrice = shortStrike,
                        Expiration = expiration,
                        Side = OrderSide.Sell,
                        Quantity = contracts,
                        EntryPrice = shortPremium,
                        CurrentPrice = shortPremium
                    }
                }
            };
        }

        /// <summary>
        /// Creates an iron condor
        /// </summary>
        public static OptionsStrategy CreateIronCondor(
            string symbol,
            decimal putLongStrike,
            decimal putShortStrike,
            decimal callShortStrike,
            decimal callLongStrike,
            DateTime expiration,
            decimal putLongPremium,
            decimal putShortPremium,
            decimal callShortPremium,
            decimal callLongPremium,
            int contracts = 1)
        {
            return new OptionsStrategy
            {
                Name = $"Iron Condor {symbol} {putShortStrike:F0}/{callShortStrike:F0} {expiration:MMM dd}",
                UnderlyingSymbol = symbol,
                StrategyType = OptionStrategyType.IronCondor,
                Legs = new List<OptionsLeg>
                {
                    new OptionsLeg
                    {
                        UnderlyingSymbol = symbol,
                        OptionType = OptionType.Put,
                        StrikePrice = putLongStrike,
                        Expiration = expiration,
                        Side = OrderSide.Buy,
                        Quantity = contracts,
                        EntryPrice = putLongPremium,
                        CurrentPrice = putLongPremium
                    },
                    new OptionsLeg
                    {
                        UnderlyingSymbol = symbol,
                        OptionType = OptionType.Put,
                        StrikePrice = putShortStrike,
                        Expiration = expiration,
                        Side = OrderSide.Sell,
                        Quantity = contracts,
                        EntryPrice = putShortPremium,
                        CurrentPrice = putShortPremium
                    },
                    new OptionsLeg
                    {
                        UnderlyingSymbol = symbol,
                        OptionType = OptionType.Call,
                        StrikePrice = callShortStrike,
                        Expiration = expiration,
                        Side = OrderSide.Sell,
                        Quantity = contracts,
                        EntryPrice = callShortPremium,
                        CurrentPrice = callShortPremium
                    },
                    new OptionsLeg
                    {
                        UnderlyingSymbol = symbol,
                        OptionType = OptionType.Call,
                        StrikePrice = callLongStrike,
                        Expiration = expiration,
                        Side = OrderSide.Buy,
                        Quantity = contracts,
                        EntryPrice = callLongPremium,
                        CurrentPrice = callLongPremium
                    }
                }
            };
        }

        /// <summary>
        /// Creates a long straddle
        /// </summary>
        public static OptionsStrategy CreateLongStraddle(
            string symbol,
            decimal strike,
            DateTime expiration,
            decimal callPremium,
            decimal putPremium,
            int contracts = 1)
        {
            return new OptionsStrategy
            {
                Name = $"Long Straddle {symbol} {strike:F0} {expiration:MMM dd}",
                UnderlyingSymbol = symbol,
                StrategyType = OptionStrategyType.LongStraddle,
                Legs = new List<OptionsLeg>
                {
                    new OptionsLeg
                    {
                        UnderlyingSymbol = symbol,
                        OptionType = OptionType.Call,
                        StrikePrice = strike,
                        Expiration = expiration,
                        Side = OrderSide.Buy,
                        Quantity = contracts,
                        EntryPrice = callPremium,
                        CurrentPrice = callPremium
                    },
                    new OptionsLeg
                    {
                        UnderlyingSymbol = symbol,
                        OptionType = OptionType.Put,
                        StrikePrice = strike,
                        Expiration = expiration,
                        Side = OrderSide.Buy,
                        Quantity = contracts,
                        EntryPrice = putPremium,
                        CurrentPrice = putPremium
                    }
                }
            };
        }
    }
}
