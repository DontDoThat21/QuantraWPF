using System;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Static service for managing global application loading state
    /// </summary>
    public static class GlobalLoadingStateService
    {
        /// <summary>
        /// Event raised when the global loading state changes
        /// </summary>
        public static event Action<bool> LoadingStateChanged;

        private static bool _isLoading = false;

        /// <summary>
        /// Gets the current global loading state
        /// </summary>
        public static bool IsLoading => _isLoading;

        /// <summary>
        /// Sets the global loading state and notifies subscribers
        /// </summary>
        /// <param name="isLoading">True to indicate loading/busy, false for idle</param>
        public static void SetLoadingState(bool isLoading)
        {
            if (_isLoading != isLoading)
            {
                _isLoading = isLoading;
                LoadingStateChanged?.Invoke(isLoading);
            }
        }

        /// <summary>
        /// Convenience method to temporarily set loading state during an operation
        /// </summary>
        /// <param name="operation">The operation to perform while loading</param>
        /// <returns>The async operation</returns>
        public static async Task WithLoadingState(Task operation)
        {
            try
            {
                SetLoadingState(true);
                await operation;
            }
            finally
            {
                SetLoadingState(false);
            }
        }

        /// <summary>
        /// Convenience method to temporarily set loading state during an operation with return value
        /// </summary>
        /// <typeparam name="T">Return type of the operation</typeparam>
        /// <param name="operation">The operation to perform while loading</param>
        /// <returns>The result of the operation</returns>
        public static async Task<T> WithLoadingState<T>(Task<T> operation)
        {
            try
            {
                SetLoadingState(true);
                return await operation;
            }
            finally
            {
                SetLoadingState(false);
            }
        }
    }
}