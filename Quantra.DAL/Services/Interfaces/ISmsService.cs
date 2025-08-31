using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    public interface ISmsService
    {
        Task SendSmsAsync(string to, string message);
    }
}