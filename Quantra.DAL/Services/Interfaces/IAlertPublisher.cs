using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Abstraction for publishing alerts from non-UI layers.
    /// Implemented in the UI project and injected via DI.
    /// </summary>
    public interface IAlertPublisher
    {
        void EmitGlobalAlert(AlertModel alert);
    }
}
