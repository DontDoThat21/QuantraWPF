using Quantra.Models;

namespace Quantra.Services.Interfaces
{
    /// <summary>
    /// Interface for audio playback service used for alerts and notifications
    /// </summary>
    public interface IAudioService
    {
        /// <summary>
        /// Gets whether sound is enabled globally in the application
        /// </summary>
        bool IsEnabled { get; }

        /// <summary>
        /// Plays a sound file by name from the Sounds folder
        /// </summary>
        /// <param name="fileName">Name of the sound file to play</param>
        void PlaySound(string fileName);

        /// <summary>
        /// Plays the sound associated with an alert if sound is enabled
        /// </summary>
        /// <param name="alert">The alert model containing sound preferences</param>
        void PlayAlertSound(AlertModel alert);

        /// <summary>
        /// Stops any currently playing sounds
        /// </summary>
        void StopSound();
    }
}