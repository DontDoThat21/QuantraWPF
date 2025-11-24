using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using System;
using System.IO;
using System.Media;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for playing audio alerts and notifications
    /// </summary>
    public class AudioService : IAudioService
    {
        private readonly SoundPlayer _soundPlayer;
        private readonly UserSettings _userSettings;
        private readonly string _soundsFolder;

        public AudioService(UserSettings userSettings)
        {
            _userSettings = userSettings ?? throw new ArgumentNullException(nameof(userSettings));
            _soundPlayer = new SoundPlayer();

            // Set the path to the sounds folder
            _soundsFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Sounds");

            // Create the sounds directory if it doesn't exist
            if (!Directory.Exists(_soundsFolder))
            {
                Directory.CreateDirectory(_soundsFolder);
            }
        }

        /// <summary>
        /// Gets whether sound is enabled globally based on user settings
        /// </summary>
        public bool IsEnabled => _userSettings.EnableAlertSounds;

        /// <summary>
        /// Plays a sound file by name from the Sounds folder
        /// </summary>
        /// <param name="fileName">Name of the sound file to play</param>
        public void PlaySound(string fileName)
        {
            if (!IsEnabled || string.IsNullOrEmpty(fileName))
                return;

            try
            {
                string soundPath = Path.Combine(_soundsFolder, fileName);

                if (File.Exists(soundPath))
                {
                    _soundPlayer.SoundLocation = soundPath;
                    _soundPlayer.Play();
                }
            }
            catch (Exception ex)
            {
                // Log the exception but don't throw it to prevent UI disruption
                Console.WriteLine($"Error playing sound: {ex.Message}");
            }
        }

        /// <summary>
        /// Plays the sound associated with an alert if sound is enabled
        /// </summary>
        /// <param name="alert">The alert model containing sound preferences</param>
        public void PlayAlertSound(AlertModel alert)
        {
            if (!IsEnabled || alert == null || !alert.EnableSound)
                return;

            string soundFile = alert.SoundFileName;

            // If no specific sound is set for this alert, use the default based on category
            if (string.IsNullOrEmpty(soundFile))
            {
                soundFile = alert.Category switch
                {
                    AlertCategory.Opportunity => _userSettings.DefaultOpportunitySound,
                    AlertCategory.Prediction => _userSettings.DefaultPredictionSound,
                    AlertCategory.TechnicalIndicator => _userSettings.DefaultTechnicalIndicatorSound,
                    _ => _userSettings.DefaultAlertSound
                };
            }

            PlaySound(soundFile);
        }

        /// <summary>
        /// Stops any currently playing sounds
        /// </summary>
        public void StopSound()
        {
            try
            {
                _soundPlayer.Stop();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error stopping sound: {ex.Message}");
            }
        }
    }
}