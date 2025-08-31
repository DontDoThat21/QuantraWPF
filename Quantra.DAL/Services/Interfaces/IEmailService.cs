using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    public interface IEmailService
    {
        Task SendEmailAsync(string to, string subject, string body);
    }
}