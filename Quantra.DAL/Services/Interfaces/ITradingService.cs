using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    public interface ITradingService
    {
        Task<bool> ExecuteTradeAsync(string symbol, string action, double price, double targetPrice);
    }
}