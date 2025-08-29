using System;
using System.Net.Mail;
using System.Threading.Tasks;
using Quantra.Data;
using Quantra.Services.Interfaces;

namespace Quantra.Services
{
    public class EmailService : IEmailService
    {
        // These should be moved to configuration
        private const string SmtpServer = "smtp.gmail.com";
        private const int SmtpPort = 587;
        private const string FromEmail = "your-email@gmail.com";
        
        private string GetEmailPassword()
        {
            // Get this from secure configuration or environment variables
            return Environment.GetEnvironmentVariable("EMAIL_PASSWORD") 
                ?? "your-app-specific-password";
        }

        public async Task SendEmailAsync(string to, string subject, string body)
        {
            try
            {
                using var client = new SmtpClient(SmtpServer, SmtpPort)
                {
                    EnableSsl = true,
                    Credentials = new System.Net.NetworkCredential(FromEmail, GetEmailPassword())
                };

                using var mailMessage = new MailMessage
                {
                    From = new MailAddress(FromEmail),
                    Subject = subject,
                    Body = body,
                    IsBodyHtml = false
                };
                
                mailMessage.To.Add(to);
                
                await client.SendMailAsync(mailMessage);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error sending email: {ex.Message}", ex.ToString());
                throw;
            }
        }
    }
}