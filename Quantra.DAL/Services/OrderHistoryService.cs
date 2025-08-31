using System;
using Quantra.Models;
using System.Data.SQLite;

namespace Quantra.DAL.Services
{
    public static class OrderHistoryService
    {
        public static void AddOrderToHistory(OrderModel order)
        {
            DatabaseMonolith.AddOrderToHistory(order);
        }
    }
}
