using System;
using Quantra.Models;
//using System.Data.SQLite;

namespace Quantra.DAL.Services
{
    public class OrderHistoryService
    {
        DatabaseMonolith _databaseMonolith;

        public OrderHistoryService(DatabaseMonolith databaseMonolith)
        {
            _databaseMonolith = databaseMonolith;
        }   

        public void AddOrderToHistory(OrderModel order)
        {
            _databaseMonolith.AddOrderToHistory(order);
        }
    }
}
