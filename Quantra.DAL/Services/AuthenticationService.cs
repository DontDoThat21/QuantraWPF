using System;
using System.Collections.Generic;
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
        // Thread-safe implementation using lock for single-threaded WPF app context
        private static int? _currentUserId;
        private static readonly object _userIdLock = new object();

        /// <summary>
        /// Gets the currently logged-in user ID, or null if no user is logged in.
        /// This is a static property for application-wide user context in this single-user WPF application.
        /// </summary>
        public static int? CurrentUserId
        {
            get
            {
                lock (_userIdLock)
                {
                    return _currentUserId;
                }
            }
        }

        /// <summary>
        /// Sets the current user ID. Should only be called by AuthenticationService during login/logout.
        /// </summary>
        internal static void SetCurrentUserId(int? userId)
        {
            lock (_userIdLock)
            {
                _currentUserId = userId;
            }
        }

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
                    .AsNoTracking()  // Read-only check
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
                // Query user credentials - tracking is needed here because we'll update LastLoginDate
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

                // Set current user ID using thread-safe method
                SetCurrentUserId(user.Id);

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
            SetCurrentUserId(null);
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
                .AsNoTracking()  // Read-only query - no tracking needed
                .AnyAsync(u => u.Username.ToLower() == username.ToLower());
        }

        /// <summary>
        /// Gets a list of all previously logged-in users (users with a LastLoginDate)
        /// </summary>
        /// <returns>List of usernames that have previously logged in</returns>
        public async Task<List<string>> GetPreviouslyLoggedInUsersAsync()
        {
            try
            {
                return await _dbContext.UserCredentials
                    .AsNoTracking()  // Read-only query - no tracking needed
                    .Where(u => u.IsActive && u.LastLoginDate != null)
                    .OrderByDescending(u => u.LastLoginDate)
                    .Select(u => u.Username)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to get previously logged-in users", ex.ToString());
                return new List<string>();
            }
        }

        /// <summary>
        /// Gets the username for the currently logged-in user
        /// </summary>
        /// <returns>Username of the current user, or null if not logged in</returns>
        public async Task<string> GetCurrentUsernameAsync()
        {
            if (!CurrentUserId.HasValue)
            {
                return null;
            }

            try
            {
                var user = await _dbContext.UserCredentials
                    .AsNoTracking()  // Read-only query - no tracking needed
                    .FirstOrDefaultAsync(u => u.Id == CurrentUserId.Value);
                return user?.Username;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get username for user ID: {CurrentUserId.Value}", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Gets the username for the currently logged-in user (synchronous version)
        /// </summary>
        /// <returns>Username of the current user, or null if not logged in</returns>
        public string GetCurrentUsername()
        {
            if (!CurrentUserId.HasValue)
            {
                return null;
            }

            try
            {
                var user = _dbContext.UserCredentials
                    .AsNoTracking()  // Read-only query - no tracking needed
                    .FirstOrDefault(u => u.Id == CurrentUserId.Value);
                return user?.Username;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get username for user ID: {CurrentUserId.Value}", ex.ToString());
                return null;
            }
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
                    
                    // Risk management settings with valid defaults
                    AccountSize = 100000m,
                    BaseRiskPercentage = 0.01m, // 1%
                    PositionSizingMethod = "FixedRisk",
                    MaxPositionSizePercent = 0.1m, // 10%
                    FixedTradeAmount = 5000m,
                    UseVolatilityBasedSizing = false,
                    ATRMultiple = 2m,
                    UseKellyCriterion = false,
                    HistoricalWinRate = 0.55m,
                    HistoricalRewardRiskRatio = 2m,
                    KellyFractionMultiplier = 0.5m,
                    
                    // Alert sound settings
                    EnableAlertSounds = true,
                    DefaultAlertSound = "alert.wav",
                    DefaultOpportunitySound = "opportunity.wav",
                    DefaultPredictionSound = "prediction.wav",
                    DefaultTechnicalIndicatorSound = "indicator.wav",
                    AlertVolume = 80,
                    
                    // Visual indicator settings
                    EnableVisualIndicators = true,
                    DefaultVisualIndicatorType = "Toast",
                    DefaultVisualIndicatorColor = "#FFFF00",
                    VisualIndicatorDuration = 5,
                    
                    // News sentiment settings
                    EnableNewsSentimentAnalysis = true,
                    NewsArticleRefreshIntervalMinutes = 30,
                    MaxNewsArticlesPerSymbol = 15,
                    EnableNewsSourceFiltering = true,
                    
                    // Analyst ratings settings
                    EnableAnalystRatings = true,
                    RatingsCacheExpiryHours = 24,
                    EnableRatingChangeAlerts = true,
                    EnableConsensusChangeAlerts = true,
                    AnalystRatingSentimentWeight = 2m,
                    
                    // Insider trading settings
                    EnableInsiderTradingAnalysis = true,
                    InsiderDataRefreshIntervalMinutes = 120,
                    EnableInsiderTradingAlerts = true,
                    TrackNotableInsiders = true,
                    InsiderTradingSentimentWeight = 2.5m,
                    HighlightCEOTransactions = true,
                    HighlightOptionsActivity = true,
                    EnableInsiderTransactionNotifications = true,
                    
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
        /// OWASP recommends at least 120,000 iterations for PBKDF2 with SHA-256
        /// </summary>
        private string ComputeHash(string password, byte[] salt)
        {
            using (var pbkdf2 = new Rfc2898DeriveBytes(password, salt, 120000, HashAlgorithmName.SHA256))
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
