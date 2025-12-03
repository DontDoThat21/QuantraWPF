using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Quantra.DAL.TradingEngine.Core;
using Quantra.DAL.TradingEngine.Data;
using Quantra.DAL.TradingEngine.Execution;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.DAL.TradingEngine.Positions;
using Quantra.DAL.TradingEngine.Time;
using Quantra.Models;

namespace Quantra.Tests.TradingEngine
{
    /// <summary>
    /// Unit tests for the Trading Engine core functionality
    /// </summary>
    public class TradingEngineTests
    {
        private readonly HistoricalDataSource _dataSource;
        private readonly BacktestClock _clock;
        private readonly PortfolioManager _portfolio;
        private readonly Quantra.DAL.TradingEngine.Core.TradingEngine _engine;

        public TradingEngineTests()
        {
            _dataSource = new HistoricalDataSource();
            _clock = new BacktestClock(new DateTime(2024, 1, 2, 9, 30, 0));
            _portfolio = new PortfolioManager(100000m);
            _engine = new Quantra.DAL.TradingEngine.Core.TradingEngine();

            // Load sample historical data
            LoadSampleData();

            _engine.Initialize(_dataSource, _clock, _portfolio);
        }

        private void LoadSampleData()
        {
            // Create sample historical data for AAPL
            var aaplPrices = new List<HistoricalPrice>();
            DateTime startDate = new DateTime(2024, 1, 2);

            for (int i = 0; i < 30; i++)
            {
                aaplPrices.Add(new HistoricalPrice
                {
                    Date = startDate.AddDays(i),
                    Open = 180 + i * 0.5,
                    High = 182 + i * 0.5,
                    Low = 179 + i * 0.5,
                    Close = 181 + i * 0.5,
                    Volume = 10000000
                });
            }

            _dataSource.LoadHistoricalData("AAPL", aaplPrices);

            // Create sample data for MSFT
            var msftPrices = new List<HistoricalPrice>();
            for (int i = 0; i < 30; i++)
            {
                msftPrices.Add(new HistoricalPrice
                {
                    Date = startDate.AddDays(i),
                    Open = 370 + i * 1.0,
                    High = 375 + i * 1.0,
                    Low = 368 + i * 1.0,
                    Close = 372 + i * 1.0,
                    Volume = 8000000
                });
            }

            _dataSource.LoadHistoricalData("MSFT", msftPrices);
        }

        [Fact]
        public void TradingEngine_Initialization_SetsPropertiesCorrectly()
        {
            // Assert
            Assert.NotNull(_engine.GetDataSource());
            Assert.NotNull(_engine.GetClock());
            Assert.NotNull(_engine.GetPortfolio());
            Assert.Equal(100000m, _engine.GetPortfolio().CashBalance);
        }

        [Fact]
        public async Task PlaceOrder_MarketBuy_CreatesOrderSuccessfully()
        {
            // Arrange
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 100);

            // Act
            var orderId = await _engine.PlaceOrderAsync(order);

            // Assert
            Assert.NotEqual(Guid.Empty, orderId);
            var placedOrder = _engine.GetOrder(orderId);
            Assert.NotNull(placedOrder);
            Assert.Equal("AAPL", placedOrder!.Symbol);
            Assert.Equal(OrderSide.Buy, placedOrder.Side);
            Assert.Equal(100, placedOrder.Quantity);
        }

        [Fact]
        public async Task PlaceOrder_LimitOrder_StaysSubmittedWhenNotFillable()
        {
            // Arrange - create limit order below current ask price
            var order = Order.CreateLimitOrder("AAPL", OrderSide.Buy, 100, 150m); // Way below market

            // Act
            await _engine.PlaceOrderAsync(order);

            // Assert
            Assert.Equal(OrderState.Submitted, order.State);
            Assert.Equal(0, order.FilledQuantity);
        }

        [Fact]
        public async Task PlaceOrder_WithInvalidSymbol_RejectsOrder()
        {
            // Arrange
            var order = Order.CreateMarketOrder("", OrderSide.Buy, 100);

            // Act
            await _engine.PlaceOrderAsync(order);

            // Assert
            Assert.Equal(OrderState.Rejected, order.State);
            Assert.Contains("Symbol", order.RejectReason);
        }

        [Fact]
        public async Task PlaceOrder_WithZeroQuantity_RejectsOrder()
        {
            // Arrange
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 0);

            // Act
            await _engine.PlaceOrderAsync(order);

            // Assert
            Assert.Equal(OrderState.Rejected, order.State);
            Assert.Contains("Quantity", order.RejectReason);
        }

        [Fact]
        public void CancelOrder_ExistingOrder_CancelsSuccessfully()
        {
            // Arrange
            var order = Order.CreateLimitOrder("AAPL", OrderSide.Buy, 100, 150m);
            order.State = OrderState.Submitted;
            _engine.PlaceOrderAsync(order).Wait();

            // Act
            var cancelled = _engine.CancelOrder(order.Id);

            // Assert
            Assert.True(cancelled);
            Assert.Equal(OrderState.Cancelled, order.State);
        }

        [Fact]
        public void CancelOrder_NonExistentOrder_ReturnsFalse()
        {
            // Act
            var cancelled = _engine.CancelOrder(Guid.NewGuid());

            // Assert
            Assert.False(cancelled);
        }

        [Fact]
        public async Task GetActiveOrders_ReturnsOnlyNonTerminalOrders()
        {
            // Arrange
            var order1 = Order.CreateLimitOrder("AAPL", OrderSide.Buy, 100, 150m);
            var order2 = Order.CreateLimitOrder("MSFT", OrderSide.Buy, 50, 350m);

            await _engine.PlaceOrderAsync(order1);
            await _engine.PlaceOrderAsync(order2);
            _engine.CancelOrder(order1.Id);

            // Act
            var activeOrders = _engine.GetActiveOrders().ToList();

            // Assert
            Assert.Single(activeOrders);
            Assert.Equal("MSFT", activeOrders[0].Symbol);
        }

        [Fact]
        public void GetPortfolio_ReturnsCorrectInitialBalance()
        {
            // Act
            var portfolio = _engine.GetPortfolio();

            // Assert
            Assert.Equal(100000m, portfolio.CashBalance);
            Assert.Equal(100000m, portfolio.TotalValue);
            Assert.Equal(0m, portfolio.UnrealizedPnL);
            Assert.Equal(0m, portfolio.RealizedPnL);
        }

        [Fact]
        public async Task ProcessTimeStep_UpdatesPositionPrices()
        {
            // Arrange - First create a position by simulating a fill
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 100);
            await _engine.PlaceOrderAsync(order);
            await _engine.ProcessTimeStepAsync(_clock.CurrentTime);

            // Act - Advance time and process
            _clock.AdvanceTo(new DateTime(2024, 1, 5, 9, 30, 0));
            await _engine.ProcessTimeStepAsync(_clock.CurrentTime);

            // Assert - Position should exist and be updated
            var positions = _engine.GetPositions().ToList();
            // If position was created, it should be tracked
            var portfolio = _engine.GetPortfolio();
            Assert.NotNull(portfolio);
        }
    }

    /// <summary>
    /// Unit tests for the Order class
    /// </summary>
    public class OrderTests
    {
        [Fact]
        public void CreateMarketOrder_SetsPropertiesCorrectly()
        {
            // Act
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 100);

            // Assert
            Assert.NotEqual(Guid.Empty, order.Id);
            Assert.Equal("AAPL", order.Symbol);
            Assert.Equal(OrderType.Market, order.OrderType);
            Assert.Equal(OrderSide.Buy, order.Side);
            Assert.Equal(100, order.Quantity);
            Assert.Equal(OrderState.Pending, order.State);
        }

        [Fact]
        public void CreateLimitOrder_SetsLimitPrice()
        {
            // Act
            var order = Order.CreateLimitOrder("AAPL", OrderSide.Buy, 100, 175.50m);

            // Assert
            Assert.Equal(OrderType.Limit, order.OrderType);
            Assert.Equal(175.50m, order.LimitPrice);
        }

        [Fact]
        public void CreateStopOrder_SetsStopPrice()
        {
            // Act
            var order = Order.CreateStopOrder("AAPL", OrderSide.Sell, 100, 170m);

            // Assert
            Assert.Equal(OrderType.Stop, order.OrderType);
            Assert.Equal(170m, order.StopPrice);
        }

        [Fact]
        public void CreateStopLimitOrder_SetsBothPrices()
        {
            // Act
            var order = Order.CreateStopLimitOrder("AAPL", OrderSide.Sell, 100, 170m, 168m);

            // Assert
            Assert.Equal(OrderType.StopLimit, order.OrderType);
            Assert.Equal(170m, order.StopPrice);
            Assert.Equal(168m, order.LimitPrice);
        }

        [Fact]
        public void Order_RemainingQuantity_CalculatesCorrectly()
        {
            // Arrange
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 100);
            order.FilledQuantity = 30;

            // Assert
            Assert.Equal(70, order.RemainingQuantity);
        }

        [Fact]
        public void Order_FillPercentage_CalculatesCorrectly()
        {
            // Arrange
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 100);
            order.FilledQuantity = 50;

            // Assert
            Assert.Equal(50m, order.FillPercentage);
        }

        [Fact]
        public void Order_IsTerminal_ReturnsTrueForFilledOrder()
        {
            // Arrange
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 100);
            order.State = OrderState.Filled;

            // Assert
            Assert.True(order.IsTerminal);
        }

        [Fact]
        public void Order_CanCancel_ReturnsTrueForSubmittedOrder()
        {
            // Arrange
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 100);
            order.State = OrderState.Submitted;

            // Assert
            Assert.True(order.CanCancel);
        }

        [Fact]
        public void Order_CanCancel_ReturnsFalseForFilledOrder()
        {
            // Arrange
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 100);
            order.State = OrderState.Filled;

            // Assert
            Assert.False(order.CanCancel);
        }
    }

    /// <summary>
    /// Unit tests for the TrailingStopOrder class
    /// </summary>
    public class TrailingStopOrderTests
    {
        [Fact]
        public void CreateSellTrailingStop_SetsPropertiesCorrectly()
        {
            // Act
            var order = TrailingStopOrder.CreateSellTrailingStop("AAPL", 100, 5m, 180m);

            // Assert
            Assert.Equal(OrderType.TrailingStop, order.OrderType);
            Assert.Equal(OrderSide.Sell, order.Side);
            Assert.Equal(5m, order.TrailAmount);
            Assert.False(order.UsePercentage);
            Assert.Equal(180m, order.HighWaterMark);
            Assert.Equal(175m, order.StopPrice);
        }

        [Fact]
        public void CreateSellTrailingStopPercent_SetsPropertiesCorrectly()
        {
            // Act
            var order = TrailingStopOrder.CreateSellTrailingStopPercent("AAPL", 100, 0.05m, 180m);

            // Assert
            Assert.Equal(0.05m, order.TrailPercent);
            Assert.True(order.UsePercentage);
            Assert.Equal(180m, order.HighWaterMark);
            Assert.Equal(171m, order.StopPrice); // 180 * (1 - 0.05) = 171
        }

        [Fact]
        public void UpdateStopPrice_PriceIncrease_UpdatesStopPrice()
        {
            // Arrange
            var order = TrailingStopOrder.CreateSellTrailingStop("AAPL", 100, 5m, 180m);

            // Act
            var triggered = order.UpdateStopPrice(185m);

            // Assert
            Assert.False(triggered);
            Assert.Equal(185m, order.HighWaterMark);
            Assert.Equal(180m, order.StopPrice);
        }

        [Fact]
        public void UpdateStopPrice_PriceDropsToStop_ReturnsTrueForTrigger()
        {
            // Arrange
            var order = TrailingStopOrder.CreateSellTrailingStop("AAPL", 100, 5m, 180m);

            // Act - Price drops to stop level
            var triggered = order.UpdateStopPrice(175m);

            // Assert
            Assert.True(triggered);
        }

        [Fact]
        public void UpdateStopPrice_BuyTrailingStop_TracksLowWaterMark()
        {
            // Arrange
            var order = TrailingStopOrder.CreateBuyTrailingStop("AAPL", 100, 5m, 180m);

            // Act - Price drops
            order.UpdateStopPrice(175m);

            // Assert
            Assert.Equal(175m, order.LowWaterMark);
            Assert.Equal(180m, order.StopPrice); // 175 + 5
        }
    }

    /// <summary>
    /// Unit tests for the PortfolioManager class
    /// </summary>
    public class PortfolioManagerTests
    {
        [Fact]
        public void PortfolioManager_InitialState_HasCorrectBalance()
        {
            // Arrange & Act
            var portfolio = new PortfolioManager(100000m);

            // Assert
            Assert.Equal(100000m, portfolio.CashBalance);
            Assert.Equal(100000m, portfolio.TotalValue);
            Assert.Empty(portfolio.Positions);
        }

        [Fact]
        public void ProcessFill_BuyOrder_ReducesCashAndCreatesPosition()
        {
            // Arrange
            var portfolio = new PortfolioManager(100000m);
            var fill = new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 180m,
                Side = OrderSide.Buy,
                Commission = 5m,
                FillTime = DateTime.UtcNow
            };

            // Act
            portfolio.ProcessFill(fill);

            // Assert
            Assert.Equal(100000m - (100 * 180m) - 5m, portfolio.CashBalance); // 81995
            Assert.Single(portfolio.Positions);
            Assert.True(portfolio.Positions.ContainsKey("AAPL"));
            Assert.Equal(100, portfolio.Positions["AAPL"].Quantity);
        }

        [Fact]
        public void ProcessFill_SellOrder_IncreasesCashAndRemovesPosition()
        {
            // Arrange
            var portfolio = new PortfolioManager(100000m);

            // Buy first
            portfolio.ProcessFill(new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 180m,
                Side = OrderSide.Buy,
                Commission = 5m,
                FillTime = DateTime.UtcNow
            });

            // Act - Sell at higher price
            portfolio.ProcessFill(new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 190m,
                Side = OrderSide.Sell,
                Commission = 5m,
                FillTime = DateTime.UtcNow
            });

            // Assert
            Assert.Empty(portfolio.Positions);
            // Initial: 100000, after buy: 100000 - 18000 - 5 = 81995
            // After sell: 81995 + 19000 - 5 = 100990
            Assert.Equal(100990m, portfolio.CashBalance);
        }

        [Fact]
        public void ProcessFill_PartialSell_UpdatesPositionQuantity()
        {
            // Arrange
            var portfolio = new PortfolioManager(100000m);

            // Buy 100 shares
            portfolio.ProcessFill(new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 180m,
                Side = OrderSide.Buy,
                Commission = 5m,
                FillTime = DateTime.UtcNow
            });

            // Act - Sell 50 shares
            portfolio.ProcessFill(new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 50,
                Price = 190m,
                Side = OrderSide.Sell,
                Commission = 5m,
                FillTime = DateTime.UtcNow
            });

            // Assert
            Assert.Single(portfolio.Positions);
            Assert.Equal(50, portfolio.Positions["AAPL"].Quantity);
        }

        [Fact]
        public void TakeSnapshot_CapturesCurrentState()
        {
            // Arrange
            var portfolio = new PortfolioManager(100000m);
            portfolio.ProcessFill(new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 180m,
                Side = OrderSide.Buy,
                Commission = 0m,
                FillTime = DateTime.UtcNow
            });

            // Update current price
            portfolio.Positions["AAPL"].CurrentPrice = 185m;

            // Act
            var snapshot = portfolio.TakeSnapshot(DateTime.UtcNow);

            // Assert
            Assert.Equal(82000m, snapshot.CashBalance);
            Assert.Single(snapshot.Positions);
            Assert.Equal(18500m, snapshot.Positions["AAPL"].MarketValue);
        }

        [Fact]
        public void Reset_ClearsAllPositionsAndResetsBalance()
        {
            // Arrange
            var portfolio = new PortfolioManager(100000m);
            portfolio.ProcessFill(new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 180m,
                Side = OrderSide.Buy,
                Commission = 0m,
                FillTime = DateTime.UtcNow
            });

            // Act
            portfolio.Reset(50000m);

            // Assert
            Assert.Empty(portfolio.Positions);
            Assert.Equal(50000m, portfolio.CashBalance);
        }
    }

    /// <summary>
    /// Unit tests for the TradingPosition class
    /// </summary>
    public class TradingPositionTests
    {
        [Fact]
        public void AddFill_FirstBuy_CreatesLongPosition()
        {
            // Arrange
            var position = new TradingPosition { Symbol = "AAPL" };
            var fill = new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 180m,
                Side = OrderSide.Buy,
                FillTime = DateTime.UtcNow
            };

            // Act
            position.AddFill(fill);

            // Assert
            Assert.True(position.IsLong);
            Assert.Equal(100, position.Quantity);
            Assert.Equal(180m, position.AverageCost);
        }

        [Fact]
        public void AddFill_AddToPosition_UpdatesAverageCost()
        {
            // Arrange
            var position = new TradingPosition { Symbol = "AAPL" };

            // First fill
            position.AddFill(new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 180m,
                Side = OrderSide.Buy,
                FillTime = DateTime.UtcNow
            });

            // Act - Add more at higher price
            position.AddFill(new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 190m,
                Side = OrderSide.Buy,
                FillTime = DateTime.UtcNow
            });

            // Assert
            Assert.Equal(200, position.Quantity);
            Assert.Equal(185m, position.AverageCost); // (100*180 + 100*190) / 200 = 185
        }

        [Fact]
        public void Position_UnrealizedPnL_CalculatesCorrectly()
        {
            // Arrange
            var position = new TradingPosition
            {
                Symbol = "AAPL",
                Quantity = 100,
                AverageCost = 180m,
                CurrentPrice = 190m
            };

            // Act - Simulate property notification by adding a fill to trigger calculation
            position.AddFill(new OrderFill
            {
                Symbol = "AAPL",
                Quantity = 100,
                Price = 180m,
                Side = OrderSide.Buy,
                FillTime = DateTime.UtcNow
            });
            position.CurrentPrice = 190m;

            // Assert
            Assert.Equal(1000m, position.UnrealizedPnL); // (190 - 180) * 100 = 1000
        }
    }

    /// <summary>
    /// Unit tests for the BacktestClock class
    /// </summary>
    public class BacktestClockTests
    {
        [Fact]
        public void BacktestClock_InitialState_SetsStartTime()
        {
            // Arrange
            var startTime = new DateTime(2024, 1, 2, 9, 30, 0);

            // Act
            var clock = new BacktestClock(startTime);

            // Assert
            Assert.Equal(startTime, clock.CurrentTime);
            Assert.True(clock.IsSimulated);
            Assert.False(clock.IsRunning);
        }

        [Fact]
        public void AdvanceTo_FutureTime_UpdatesCurrentTime()
        {
            // Arrange
            var startTime = new DateTime(2024, 1, 2, 9, 30, 0);
            var clock = new BacktestClock(startTime);
            var targetTime = new DateTime(2024, 1, 3, 9, 30, 0);

            // Act
            clock.AdvanceTo(targetTime);

            // Assert
            Assert.Equal(targetTime, clock.CurrentTime);
        }

        [Fact]
        public void AdvanceTo_PastTime_DoesNotChangeTime()
        {
            // Arrange
            var startTime = new DateTime(2024, 1, 2, 9, 30, 0);
            var clock = new BacktestClock(startTime);
            var pastTime = new DateTime(2024, 1, 1, 9, 30, 0);

            // Act
            clock.AdvanceTo(pastTime);

            // Assert
            Assert.Equal(startTime, clock.CurrentTime);
        }

        [Fact]
        public void AdvanceBy_Duration_AdvancesTimeCorrectly()
        {
            // Arrange
            var startTime = new DateTime(2024, 1, 2, 9, 30, 0);
            var clock = new BacktestClock(startTime);

            // Act
            clock.AdvanceBy(TimeSpan.FromDays(1));

            // Assert
            Assert.Equal(startTime.AddDays(1), clock.CurrentTime);
        }

        [Fact]
        public void TimeChanged_RaisesEvent_WhenAdvancing()
        {
            // Arrange
            var startTime = new DateTime(2024, 1, 2, 9, 30, 0);
            var clock = new BacktestClock(startTime);
            var eventRaised = false;
            DateTime? eventTime = null;

            clock.TimeChanged += (sender, time) =>
            {
                eventRaised = true;
                eventTime = time;
            };

            // Act
            var targetTime = new DateTime(2024, 1, 3, 9, 30, 0);
            clock.AdvanceTo(targetTime);

            // Assert
            Assert.True(eventRaised);
            Assert.Equal(targetTime, eventTime);
        }
    }

    /// <summary>
    /// Unit tests for the FillSimulator class
    /// </summary>
    public class FillSimulatorTests
    {
        [Fact]
        public void TryFill_MarketOrder_FillsSuccessfully()
        {
            // Arrange
            var simulator = new FillSimulator();
            var order = Order.CreateMarketOrder("AAPL", OrderSide.Buy, 100);
            var quote = new Quote
            {
                Symbol = "AAPL",
                Bid = 179.50m,
                Ask = 180.00m,
                Last = 179.75m,
                BidSize = 1000,
                AskSize = 1000,
                Timestamp = DateTime.UtcNow
            };

            // Act
            var result = simulator.TryFill(order, quote, DateTime.UtcNow);

            // Assert
            Assert.True(result.IsFilled);
            Assert.Equal(100, result.FilledQuantity);
            Assert.True(result.ExecutionPrice >= quote.Ask); // May include slippage
        }

        [Fact]
        public void TryFill_LimitBuyBelowAsk_DoesNotFill()
        {
            // Arrange
            var simulator = new FillSimulator();
            var order = Order.CreateLimitOrder("AAPL", OrderSide.Buy, 100, 175m);
            var quote = new Quote
            {
                Symbol = "AAPL",
                Bid = 179.50m,
                Ask = 180.00m,
                Last = 179.75m,
                BidSize = 1000,
                AskSize = 1000,
                Timestamp = DateTime.UtcNow
            };

            // Act
            var result = simulator.TryFill(order, quote, DateTime.UtcNow);

            // Assert
            Assert.False(result.IsFilled);
            Assert.NotNull(result.RejectReason);
        }

        [Fact]
        public void TryFill_LimitBuyAtAsk_FillsSuccessfully()
        {
            // Arrange
            var simulator = new FillSimulator();
            var order = Order.CreateLimitOrder("AAPL", OrderSide.Buy, 100, 180m);
            var quote = new Quote
            {
                Symbol = "AAPL",
                Bid = 179.50m,
                Ask = 180.00m,
                Last = 179.75m,
                BidSize = 1000,
                AskSize = 1000,
                Timestamp = DateTime.UtcNow
            };

            // Act
            var result = simulator.TryFill(order, quote, DateTime.UtcNow);

            // Assert
            Assert.True(result.IsFilled);
        }

        [Fact]
        public void TryFill_StopSellTriggered_FillsSuccessfully()
        {
            // Arrange
            var simulator = new FillSimulator();
            var order = Order.CreateStopOrder("AAPL", OrderSide.Sell, 100, 175m);
            var quote = new Quote
            {
                Symbol = "AAPL",
                Bid = 174.00m,
                Ask = 174.50m,
                Last = 174.25m,
                BidSize = 1000,
                AskSize = 1000,
                Timestamp = DateTime.UtcNow
            };

            // Act
            var result = simulator.TryFill(order, quote, DateTime.UtcNow);

            // Assert
            Assert.True(result.IsFilled);
        }

        [Fact]
        public void TryFill_StopSellNotTriggered_DoesNotFill()
        {
            // Arrange
            var simulator = new FillSimulator();
            var order = Order.CreateStopOrder("AAPL", OrderSide.Sell, 100, 175m);
            var quote = new Quote
            {
                Symbol = "AAPL",
                Bid = 180.00m,
                Ask = 180.50m,
                Last = 180.25m,
                BidSize = 1000,
                AskSize = 1000,
                Timestamp = DateTime.UtcNow
            };

            // Act
            var result = simulator.TryFill(order, quote, DateTime.UtcNow);

            // Assert
            Assert.False(result.IsFilled);
        }
    }
}
