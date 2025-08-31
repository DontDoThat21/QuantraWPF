using System;
using System.Threading.Tasks;

namespace Quantra.DAL.Interfaces
{
    /// <summary>
    /// Interface for managing global application loading state
    /// </summary>
    public interface IGlobalLoadingStateService
    {
        /// <summary>
        /// Event raised when the global loading state changes
        /// </summary>
        event Action<bool> LoadingStateChanged;

        /// <summary>
        /// Gets the current global loading state
        /// </summary>
        bool IsLoading { get; }

        /// <summary>
        /// Sets the global loading state and notifies subscribers
        /// </summary>
        /// <param name="isLoading">True to indicate loading/busy, false for idle</param>
        void SetLoadingState(bool isLoading);

        /// <summary>
        /// Convenience method to temporarily set loading state during an operation
        /// </summary>
        /// <param name="operation">The operation to perform while loading</param>
        /// <returns>The async operation</returns>
        Task WithLoadingState(Task operation);

        /// <summary>
        /// Convenience method to temporarily set loading state during an operation with return value
        /// </summary>
        /// <typeparam name="T">Return type of the operation</typeparam>
        /// <param name="operation">The operation to perform while loading</param>
        /// <returns>The result of the operation</returns>
        Task<T> WithLoadingState<T>(Task<T> operation);
    }
}
