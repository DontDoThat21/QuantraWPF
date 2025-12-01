using System;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for user authentication and registration
    /// </summary>
    public class AuthenticationService
    {
        private readonly QuantraDbContext _dbContext;
        private readonly LoggingService _loggingService;

        // Current logged-in user ID (used to scope settings profiles)
        private static int? _currentUserId;

        /// <summary>
        /// Gets the currently logged-in user ID, or null if no user is logged in
        /// </summary>
        public static int? CurrentUserId => _currentUserId;

        public AuthenticationService(QuantraDbContext dbContext, LoggingService loggingService)
        {
            _dbContext = dbContext ?? throw new ArgumentNullException(nameof(dbContext));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        /// <summary>
        /// Registers a new user with the provided credentials
        /// </summary>
        /// <param name="username">Unique username</param>
        /// <param name="password">User's password (will be hashed)</param>
        /// <param name="email">Optional email address</param>
        /// <returns>Registration result containing success status and user ID</returns>
        public async Task<RegistrationResult> RegisterUserAsync(string username, string password, string email = null)
        {
            if (string.IsNullOrWhiteSpace(username))
            {
                return new RegistrationResult { Success = false, ErrorMessage = "Username is required." };
            }

            if (string.IsNullOrWhiteSpace(password))
            {
                return new RegistrationResult { Success = false, ErrorMessage = "Password is required." };
            }

            if (password.Length < 6)
            {
                return new RegistrationResult { Success = false, ErrorMessage = "Password must be at least 6 characters long." };
            }

            try
            {
                // Check if username already exists
                var existingUser = await _dbContext.UserCredentials
                    .FirstOrDefaultAsync(u => u.Username.ToLower() == username.ToLower());

                if (existingUser != null)
                {
                    return new RegistrationResult { Success = false, ErrorMessage = "Username already exists." };
                }

                // Generate password hash and salt
                var (hash, salt) = HashPassword(password);

                // Create new user credential
                var newUser = new UserCredential
                {
                    Username = username,
                    Password = string.Empty, // Keep empty for backward compatibility
                    PasswordHash = hash,
                    PasswordSalt = salt,
                    Email = email,
                    Pin = string.Empty,
                    CreatedDate = DateTime.Now,
                    IsActive = true
                };

                _dbContext.UserCredentials.Add(newUser);
                await _dbContext.SaveChangesAsync();

                _loggingService.Log("Info", $"New user registered: {username}");

                // Create a default settings profile for the user
                await CreateDefaultSettingsProfileForUserAsync(newUser.Id);

                return new RegistrationResult 
                { 
                    Success = true, 
                    UserId = newUser.Id,
                    Message = "Registration successful."
                };
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to register user: {username}", ex.ToString());
                return new RegistrationResult { Success = false, ErrorMessage = $"Registration failed: {ex.Message}" };
            }
        }

        /// <summary>
        /// Authenticates a user with username and password
        /// </summary>
        /// <param name="username">Username</param>
        /// <param name="password">Password</param>
        /// <returns>Authentication result with user information if successful</returns>
        public async Task<AuthenticationResult> AuthenticateAsync(string username, string password)
        {
            if (string.IsNullOrWhiteSpace(username) || string.IsNullOrWhiteSpace(password))
            {
                return new AuthenticationResult { Success = false, ErrorMessage = "Username and password are required." };
            }

            try
            {
                var user = await _dbContext.UserCredentials
                    .FirstOrDefaultAsync(u => u.Username.ToLower() == username.ToLower());

                if (user == null)
                {
                    return new AuthenticationResult { Success = false, ErrorMessage = "Invalid username or password." };
                }

                if (!user.IsActive)
                {
                    return new AuthenticationResult { Success = false, ErrorMessage = "Account is inactive." };
                }

                // Try to verify with hashed password first
                bool passwordValid = false;
                if (!string.IsNullOrEmpty(user.PasswordHash) && !string.IsNullOrEmpty(user.PasswordSalt))
                {
                    passwordValid = VerifyPassword(password, user.PasswordHash, user.PasswordSalt);
                }
                else if (!string.IsNullOrEmpty(user.Password))
                {
                    // Backward compatibility: plain text password check
                    // Also migrate to hashed password on successful login
                    passwordValid = user.Password == password;
                    if (passwordValid)
                    {
                        // Migrate to hashed password
                        var (hash, salt) = HashPassword(password);
                        user.PasswordHash = hash;
                        user.PasswordSalt = salt;
                        user.Password = string.Empty;
                        await _dbContext.SaveChangesAsync();
                        _loggingService.Log("Info", $"Migrated user {username} to hashed password");
                    }
                }

                if (!passwordValid)
                {
                    return new AuthenticationResult { Success = false, ErrorMessage = "Invalid username or password." };
                }

                // Update last login date
                user.LastLoginDate = DateTime.Now;
                await _dbContext.SaveChangesAsync();

                // Set current user ID
                _currentUserId = user.Id;

                _loggingService.Log("Info", $"User logged in: {username}");

                return new AuthenticationResult
                {
                    Success = true,
                    UserId = user.Id,
                    Username = user.Username,
                    Email = user.Email
                };
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Authentication failed for user: {username}", ex.ToString());
                return new AuthenticationResult { Success = false, ErrorMessage = $"Authentication failed: {ex.Message}" };
            }
        }

        /// <summary>
        /// Logs out the current user
        /// </summary>
        public void Logout()
        {
            _currentUserId = null;
            _loggingService.Log("Info", "User logged out");
        }

        /// <summary>
        /// Checks if a username is available for registration
        /// </summary>
        public async Task<bool> IsUsernameAvailableAsync(string username)
        {
            if (string.IsNullOrWhiteSpace(username))
            {
                return false;
            }

            return !await _dbContext.UserCredentials
                .AnyAsync(u => u.Username.ToLower() == username.ToLower());
        }

        /// <summary>
        /// Creates a default settings profile for a new user
        /// </summary>
        private async Task CreateDefaultSettingsProfileForUserAsync(int userId)
        {
            try
            {
                var defaultProfile = new SettingsProfile
                {
                    UserId = userId,
                    Name = "Default",
                    Description = "Default user settings",
                    IsDefault = true,
                    EnableApiModalChecks = true,
                    ApiTimeoutSeconds = 30,
                    CacheDurationMinutes = 15,
                    EnableHistoricalDataCache = true,
                    EnableDarkMode = true,
                    ChartUpdateIntervalSeconds = 2,
                    EnablePriceAlerts = true,
                    EnableTradeNotifications = true,
                    EnablePaperTrading = true,
                    RiskLevel = "Low",
                    DefaultGridRows = 4,
                    DefaultGridColumns = 4,
                    GridBorderColor = "#FF00FFFF",
                    AlertEmail = string.Empty,
                    EnableEmailAlerts = false,
                    EnableVixMonitoring = true,
                    CreatedAt = DateTime.Now,
                    LastModified = DateTime.Now
                };

                _dbContext.SettingsProfiles.Add(defaultProfile);
                await _dbContext.SaveChangesAsync();

                _loggingService.Log("Info", $"Created default settings profile for user ID: {userId}");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to create default settings profile for user ID: {userId}", ex.ToString());
            }
        }

        #region Password Hashing

        /// <summary>
        /// Generates a hash and salt for a password
        /// </summary>
        private (string Hash, string Salt) HashPassword(string password)
        {
            // Generate a random salt
            byte[] saltBytes = new byte[32];
            using (var rng = RandomNumberGenerator.Create())
            {
                rng.GetBytes(saltBytes);
            }
            string salt = Convert.ToBase64String(saltBytes);

            // Hash the password with the salt
            string hash = ComputeHash(password, saltBytes);

            return (hash, salt);
        }

        /// <summary>
        /// Verifies a password against a stored hash and salt
        /// </summary>
        private bool VerifyPassword(string password, string storedHash, string storedSalt)
        {
            try
            {
                byte[] saltBytes = Convert.FromBase64String(storedSalt);
                string computedHash = ComputeHash(password, saltBytes);
                return computedHash == storedHash;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Computes a hash for a password with the given salt using PBKDF2
        /// </summary>
        private string ComputeHash(string password, byte[] salt)
        {
            using (var pbkdf2 = new Rfc2898DeriveBytes(password, salt, 10000, HashAlgorithmName.SHA256))
            {
                byte[] hash = pbkdf2.GetBytes(32);
                return Convert.ToBase64String(hash);
            }
        }

        #endregion
    }

    /// <summary>
    /// Result of a user registration attempt
    /// </summary>
    public class RegistrationResult
    {
        public bool Success { get; set; }
        public int? UserId { get; set; }
        public string Message { get; set; }
        public string ErrorMessage { get; set; }
    }

    /// <summary>
    /// Result of a user authentication attempt
    /// </summary>
    public class AuthenticationResult
    {
        public bool Success { get; set; }
        public int? UserId { get; set; }
        public string Username { get; set; }
        public string Email { get; set; }
        public string ErrorMessage { get; set; }
    }
}
