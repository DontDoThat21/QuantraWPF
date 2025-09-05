using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.Utilities;

namespace Quantra.Services
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
