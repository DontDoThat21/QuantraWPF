using Quantra.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    public static class ServiceCollectionExtensions
    {
        public static void AddQuantraServices(this IServiceCollection services)
        {
            // Register services
            services.AddSingleton<ITransactionService, TransactionService>();

            // Register the NotificationService as a singleton
            services.AddSingleton<INotificationService, NotificationService>();
            
            // Register the AnalystRatingService
            services.AddSingleton<IAnalystRatingService, AnalystRatingService>();
            
            // Register the SentimentShiftAlertService
            services.AddSingleton<ISentimentShiftAlertService, SentimentShiftAlertService>();
            
            // Register email and SMS services
            services.AddSingleton<IEmailService, EmailService>();
            services.AddSingleton<ISmsService, SmsService>();
            
            // Register TradingService
            services.AddSingleton<ITradingService, TradingService>();

            // Register ViewModels
            services.AddTransient<TransactionsViewModel>();

            // Add more services and ViewModels as needed
        }
    }
}
