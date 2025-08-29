using System.Threading.Tasks;

namespace Quantra.Services.Interfaces
{
    public interface ISmsService
    {
        Task SendSmsAsync(string to, string message);
    }
}