using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using Quantra.Enums;
using Quantra.Models;
using Microsoft.Extensions.Configuration;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    public class OrderService : IOrderService
    {
        private readonly WebullTradingBot _tradingBot;

        // Accept configuration in constructor
        public OrderService(UserSettingsService userSettingsService,
                HistoricalDataService historicalDataService,
                AlphaVantageService alphaVantageService,
                TechnicalIndicatorService technicalIndicatorService)
        {
            _tradingBot = new WebullTradingBot(userSettingsService,
                historicalDataService,
                alphaVantageService,
                technicalIndicatorService);
        }

        public async Task<bool> PlaceLimitOrder(string symbol, int quantity, string orderType, double price)
        {
            try
            {
                // This will now place a paper trade in the Webull account (if in paper mode)
                await _tradingBot.PlaceLimitOrder(symbol, quantity, orderType, price);
                return true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to place order", ex.ToString());
                return false;
            }
        }

        public async Task<double> GetMarketPrice(string symbol)
        {
            try
            {
                return await _tradingBot.GetMarketPrice(symbol);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to get market price", ex.ToString());
                throw;
            }
        }

        public void SetTradingMode(TradingMode mode)
        {
            _tradingBot.SetTradingMode(mode);
        }

        public bool GetApiModalCheckSetting()
        {
            var settings = DatabaseMonolith.GetUserSettings();
            return settings.EnableApiModalChecks;
        }

        public void SaveApiModalCheckSetting(bool enableChecks)
        {
            var settings = DatabaseMonolith.GetUserSettings();
            settings.EnableApiModalChecks = enableChecks;
            DatabaseMonolith.SaveUserSettings(settings);
        }

        public ObservableCollection<OrderModel> LoadOrderHistory()
        {
            // In a real app, this would load from database
            // For now, we'll just use some sample data
            var orderHistory = new ObservableCollection<OrderModel>();
            
            // Add some sample orders for demonstration
            orderHistory.Add(new OrderModel 
            { 
                Symbol = "AAPL", 
                OrderType = "BUY", 
                Quantity = 100, 
                Price = 182.50, 
                IsPaperTrade = true, 
                Status = "Executed", 
                Timestamp = DateTime.Now.AddDays(-3),
                PredictionSource = ""
            });
            
            orderHistory.Add(new OrderModel 
            { 
                Symbol = "MSFT", 
                OrderType = "SELL", 
                Quantity = 50, 
                Price = 326.75, 
                IsPaperTrade = false, 
                Status = "Failed", 
                Timestamp = DateTime.Now.AddDays(-2),
                PredictionSource = ""
            });
            
            orderHistory.Add(new OrderModel 
            { 
                Symbol = "TSLA", 
                OrderType = "BUY", 
                Quantity = 25, 
                Price = 215.30, 
                IsPaperTrade = true, 
                Status = "Executed", 
                Timestamp = DateTime.Now.AddDays(-1),
                PredictionSource = ""
            });

            return orderHistory;
        }

        public OrderModel CreateDefaultOrder()
        {
            return new OrderModel
            {
                Symbol = "",
                Quantity = 100,
                Price = 0,
                OrderType = "BUY",
                IsPaperTrade = true,
                Status = "New",
                Timestamp = DateTime.Now,
                StopLoss = 0,
                TakeProfit = 0,
                PredictionSource = ""
            };
        }

        // Implement new interface methods by delegating to the trading bot
        public async Task<bool> PlaceBracketOrder(string symbol, int quantity, string orderType, double price, double stopLossPrice, double takeProfitPrice)
        {
            return await _tradingBot.PlaceBracketOrder(symbol, quantity, orderType, price, stopLossPrice, takeProfitPrice);
        }

        public bool SetTrailingStop(string symbol, double initialPrice, double trailingDistance)
        {
            return _tradingBot.SetTrailingStop(symbol, initialPrice, trailingDistance);
        }

        public bool SetTimeBasedExit(string symbol, DateTime exitTime)
        {
            return _tradingBot.SetTimeBasedExit(symbol, exitTime);
        }

        public int CalculatePositionSizeByRisk(string symbol, double price, double stopLossPrice, double riskPercentage, double accountSize)
        {
            return _tradingBot.CalculatePositionSizeByRisk(symbol, price, stopLossPrice, riskPercentage, accountSize);
        }

        public bool SetupDollarCostAveraging(string symbol, int totalShares, int numberOfOrders, int intervalDays)
        {
            string strategyId = _tradingBot.SetupDollarCostAveraging(symbol, totalShares, numberOfOrders, intervalDays);
            return strategyId != null;
        }

        public bool SetPortfolioAllocations(Dictionary<string, double> allocations)
        {
            return _tradingBot.SetPortfolioAllocations(allocations);
        }

        public async Task<bool> RebalancePortfolio(double tolerancePercentage = 0.02)
        {
            return await _tradingBot.RebalancePortfolio(tolerancePercentage);
        }

        public async Task<bool> PlaceMultiLegOrder(List<ScheduledOrder> orders)
        {
            return await _tradingBot.PlaceMultiLegOrder(orders);
        }

        public bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes)
        {
            return _tradingBot.SplitLargeOrder(symbol, quantity, orderType, price, chunks, intervalMinutes);
        }
        
        public bool SplitLargeOrder(string symbol, int quantity, string orderType, double price, int chunks, int intervalMinutes,
            double priceVariancePercent, bool randomizeIntervals, OrderDistributionType distribution)
        {
            return _tradingBot.SplitLargeOrder(symbol, quantity, orderType, price, chunks, intervalMinutes, 
                priceVariancePercent, randomizeIntervals, distribution);
        }
        
        public int CancelSplitOrderGroup(string splitOrderGroupId)
        {
            return _tradingBot.CancelSplitOrderGroup(splitOrderGroupId);
        }

        public bool ActivateEmergencyStop()
        {
            return _tradingBot.ActivateEmergencyStop();
        }

        public bool DeactivateEmergencyStop()
        {
            return _tradingBot.DeactivateEmergencyStop();
        }

        public bool SetTradingHourRestrictions(TimeOnly marketOpen, TimeOnly marketClose)
        {
            return _tradingBot.SetTradingHourRestrictions(marketOpen, marketClose);
        }

        public bool SetEnabledMarketSessions(MarketSession sessions)
        {
            return _tradingBot.SetEnabledMarketSessions(sessions);
        }

        public MarketSession GetEnabledMarketSessions()
        {
            return _tradingBot.GetEnabledMarketSessions();
        }

        public bool IsTradingAllowed()
        {
            return _tradingBot.IsTradingAllowed();
        }
        
        public bool IsEmergencyStopActive()
        {
            return _tradingBot.IsEmergencyStopActive();
        }
        
        public bool SetMarketSessionTimes(TimeOnly preMarketOpenTime, TimeOnly regularMarketOpenTime, 
            TimeOnly regularMarketCloseTime, TimeOnly afterHoursCloseTime)
        {
            return _tradingBot.SetMarketSessionTimes(preMarketOpenTime, regularMarketOpenTime, 
                regularMarketCloseTime, afterHoursCloseTime);
        }
        
        public (TimeOnly preMarketOpen, TimeOnly regularMarketOpen, TimeOnly regularMarketClose, TimeOnly afterHoursClose) GetMarketSessionTimes()
        {
            return _tradingBot.GetMarketSessionTimes();
        }
    }
}
