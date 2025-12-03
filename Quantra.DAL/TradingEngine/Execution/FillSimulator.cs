using System;
using Quantra.DAL.TradingEngine.Data;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.Models;

namespace Quantra.DAL.TradingEngine.Execution
{
    /// <summary>
    /// Realistic fill simulator with configurable slippage and commission models
    /// </summary>
    public class FillSimulator : IFillSimulator
    {
        private readonly TransactionCostModel _costModel;
        private readonly Random _random;

        /// <summary>
        /// Creates a new fill simulator with the specified cost model
        /// </summary>
        public FillSimulator(TransactionCostModel? costModel = null)
        {
            _costModel = costModel ?? TransactionCostModel.CreateZeroCostModel();
            _random = new Random();
        }

        /// <summary>
        /// Attempts to fill an order given the current market quote
        /// </summary>
        public FillResult TryFill(Order order, Quote quote, DateTime time)
        {
            if (order == null || quote == null)
            {
                return FillResult.Rejected("Invalid order or quote");
            }

            // Check if order is in a fillable state
            if (order.IsTerminal)
            {
                return FillResult.Rejected("Order is in terminal state");
            }

            int remainingQuantity = order.RemainingQuantity;
            if (remainingQuantity <= 0)
            {
                return FillResult.Rejected("No remaining quantity to fill");
            }

            // Determine if order can be filled based on order type
            bool canFill = false;
            decimal basePrice = 0;

            switch (order.OrderType)
            {
                case OrderType.Market:
                    canFill = true;
                    basePrice = order.Side == OrderSide.Buy ? quote.Ask : quote.Bid;
                    break;

                case OrderType.Limit:
                    if (order.Side == OrderSide.Buy)
                    {
                        canFill = quote.Ask <= order.LimitPrice;
                        basePrice = Math.Min(quote.Ask, order.LimitPrice);
                    }
                    else
                    {
                        canFill = quote.Bid >= order.LimitPrice;
                        basePrice = Math.Max(quote.Bid, order.LimitPrice);
                    }
                    break;

                case OrderType.Stop:
                    if (order.Side == OrderSide.Sell)
                    {
                        // Sell stop triggers when price falls to stop price
                        canFill = quote.Bid <= order.StopPrice;
                        basePrice = quote.Bid;
                    }
                    else
                    {
                        // Buy stop triggers when price rises to stop price
                        canFill = quote.Ask >= order.StopPrice;
                        basePrice = quote.Ask;
                    }
                    break;

                case OrderType.StopLimit:
                    if (order.Side == OrderSide.Sell)
                    {
                        // Stop triggered and limit price met
                        bool stopTriggered = quote.Bid <= order.StopPrice;
                        bool limitMet = quote.Bid >= order.LimitPrice;
                        canFill = stopTriggered && limitMet;
                        basePrice = order.LimitPrice;
                    }
                    else
                    {
                        bool stopTriggered = quote.Ask >= order.StopPrice;
                        bool limitMet = quote.Ask <= order.LimitPrice;
                        canFill = stopTriggered && limitMet;
                        basePrice = order.LimitPrice;
                    }
                    break;

                case OrderType.TrailingStop:
                    if (order is TrailingStopOrder trailingOrder)
                    {
                        canFill = trailingOrder.UpdateStopPrice(order.Side == OrderSide.Buy ? quote.Ask : quote.Bid);
                        basePrice = order.Side == OrderSide.Buy ? quote.Ask : quote.Bid;
                    }
                    break;
            }

            if (!canFill)
            {
                return FillResult.Rejected("Order conditions not met");
            }

            // Calculate slippage and commission
            bool isBuy = order.Side == OrderSide.Buy;
            var (totalCost, effectivePrice) = _costModel.CalculateAllCosts(
                remainingQuantity, 
                (double)basePrice, 
                isBuy, 
                (double)quote.LastSize * 1000); // Estimate daily volume

            decimal slippage = (decimal)effectivePrice - basePrice;
            if (!isBuy) slippage = -slippage;

            // Simulate partial fills for large orders (over 10% of quote size)
            int fillQuantity = remainingQuantity;
            if (remainingQuantity > quote.AskSize * 10)
            {
                // Simulate partial fill based on available liquidity
                fillQuantity = Math.Max(1, (int)(remainingQuantity * (0.5 + _random.NextDouble() * 0.5)));
            }

            // Handle IOC and FOK time-in-force
            if (order.TimeInForce == TimeInForce.FOK && fillQuantity < remainingQuantity)
            {
                return FillResult.Rejected("FOK order cannot be fully filled");
            }

            decimal commission = (decimal)_costModel.CalculateCommission(fillQuantity, (double)effectivePrice);

            return FillResult.Success(
                fillQuantity,
                (decimal)effectivePrice,
                slippage,
                commission,
                time);
        }

        /// <summary>
        /// Calculates the expected execution price for an order
        /// </summary>
        public decimal GetExpectedPrice(Order order, Quote quote)
        {
            if (order == null || quote == null)
            {
                return 0;
            }

            decimal basePrice = order.Side == OrderSide.Buy ? quote.Ask : quote.Bid;

            // Apply expected slippage
            var (_, effectivePrice) = _costModel.CalculateAllCosts(
                order.RemainingQuantity,
                (double)basePrice,
                order.Side == OrderSide.Buy,
                (double)quote.LastSize * 1000);

            return (decimal)effectivePrice;
        }
    }
}
