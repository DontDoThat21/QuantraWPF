using Quantra.Models;
using System;
using System.Collections.Generic;

namespace Quantra.Services.Interfaces
{
    public interface ITransactionService
    {
        List<TransactionModel> GetTransactions();
        TransactionModel GetTransaction(int id);
        List<TransactionModel> GetTransactionsByDateRange(DateTime startDate, DateTime endDate);
        List<TransactionModel> GetTransactionsBySymbol(string symbol);
        void SaveTransaction(TransactionModel transaction);
        void DeleteTransaction(int id);
    }
}
