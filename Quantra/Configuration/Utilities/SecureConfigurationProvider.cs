using System;
using System.Security.Cryptography;
#if WINDOWS
using static System.Security.Cryptography.ProtectedData;
#endif
using System.Text;
using System.IO;

namespace Quantra.Configuration.Utilities
{
    /// <summary>
    /// Secure configuration provider for sensitive information
    /// </summary>
    public static class SecureConfigurationProvider
    {
        private const string ENTROPY_FILE = "config.entropy";
        private static readonly byte[] _defaultEntropy = Encoding.UTF8.GetBytes("Quantra_Configuration_Protection");
        private static string _entropyFilePath;

        /// <summary>
        /// Static constructor
        /// </summary>
        static SecureConfigurationProvider()
        {
            // Set up entropy file path
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var appFolder = Path.Combine(appDataPath, "Quantra");

            if (!Directory.Exists(appFolder))
                Directory.CreateDirectory(appFolder);

            _entropyFilePath = Path.Combine(appFolder, ENTROPY_FILE);

            // Create entropy file if it doesn't exist
            if (!File.Exists(_entropyFilePath))
            {
                var entropy = new byte[32];
                using (var rng = RandomNumberGenerator.Create())
                {
                    rng.GetBytes(entropy);
                }

                File.WriteAllBytes(_entropyFilePath, entropy);
            }
        }

        /// <summary>
        /// Encrypt sensitive data
        /// </summary>
        /// <param name="plainText">Plain text to encrypt</param>
        /// <param name="useCustomEntropy">Whether to use custom entropy</param>
        /// <returns>Base64-encoded encrypted string</returns>
        public static string ProtectData(string plainText, bool useCustomEntropy = true)
        {
            if (string.IsNullOrEmpty(plainText))
                return plainText;

            try
            {
#if WINDOWS
                byte[] entropy = useCustomEntropy ? GetEntropy() : _defaultEntropy;
                byte[] data = Encoding.UTF8.GetBytes(plainText);
                byte[] encryptedData = ProtectedData.Protect(data, entropy, DataProtectionScope.CurrentUser);

                return Convert.ToBase64String(encryptedData);
#else
                // On non-Windows platforms, we can't use DPAPI, so we'll just return the plain text
                // In a real application, we would use a platform-specific encryption solution
                return plainText;
#endif
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to protect sensitive configuration data", ex.ToString());
                return plainText; // Return unencrypted if encryption fails
            }
        }

        /// <summary>
        /// Decrypt sensitive data
        /// </summary>
        /// <param name="encryptedText">Base64-encoded encrypted text</param>
        /// <param name="useCustomEntropy">Whether custom entropy was used</param>
        /// <returns>Decrypted plain text</returns>
        public static string UnprotectData(string encryptedText, bool useCustomEntropy = true)
        {
            if (string.IsNullOrEmpty(encryptedText))
                return encryptedText;

            try
            {
#if WINDOWS
                byte[] entropy = useCustomEntropy ? GetEntropy() : _defaultEntropy;
                byte[] encryptedData = Convert.FromBase64String(encryptedText);
                byte[] data = ProtectedData.Unprotect(encryptedData, entropy, DataProtectionScope.CurrentUser);

                return Encoding.UTF8.GetString(data);
#else
                // On non-Windows platforms, we just return the encrypted text as is
                // In a real application, we would use a platform-specific decryption solution
                return encryptedText;
#endif
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to unprotect sensitive configuration data", ex.ToString());
                return encryptedText; // Return encrypted if decryption fails
            }
        }

        /// <summary>
        /// Get entropy for encryption
        /// </summary>
        /// <returns>Entropy byte array</returns>
        private static byte[] GetEntropy()
        {
            try
            {
                if (File.Exists(_entropyFilePath))
                {
                    return File.ReadAllBytes(_entropyFilePath);
                }
            }
            catch
            {
                // Fall back to default entropy if file can't be read
            }

            return _defaultEntropy;
        }
    }
}