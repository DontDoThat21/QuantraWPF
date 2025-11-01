using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.Utilities
{
    // Simple adapter that bridges UI AlertManager to the non-UI IAlertPublisher
    public class AlertPublisher : IAlertPublisher
    {
        public void EmitGlobalAlert(AlertModel alert)
        {
            AlertManager.EmitGlobalAlert(alert);
        }
    }
}
