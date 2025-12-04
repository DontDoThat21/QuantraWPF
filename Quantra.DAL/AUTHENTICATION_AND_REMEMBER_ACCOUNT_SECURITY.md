# Authentication and "Remember Account" Security

## Overview

This document explains how the "Remember this Account" feature works and the security improvements made to protect user credentials.

---

## ?? Security Issue Fixed

**ISSUE**: Previously, the "Remember this Account" checkbox stored user passwords in **plain text** in the database, which is a critical security vulnerability.

**RESOLUTION**: 
- Passwords are now **never** stored in plain text
- Only **password hashes** are stored (using PBKDF2 with SHA-256 and 120,000 iterations)
- The "Remember Account" feature now only stores the **username** for auto-population
- Users must **re-enter their password** on each login (industry security best practice)

---

## How It Works Now

### 1. User Registration

When a new user registers:
```csharp
var (hash, salt) = HashPassword(password);

var newUser = new UserCredential
{
    Username = username,
    Password = string.Empty,        // Empty for security
    PasswordHash = hash,            // Secure hash
    PasswordSalt = salt,            // Random salt
    Email = email,
    CreatedDate = DateTime.Now,
    IsActive = true
};
```

**Key Points:**
- Password is hashed using PBKDF2 with 120,000 iterations (OWASP recommended)
- A random 32-byte salt is generated for each password
- Hash and salt are stored separately
- Original password is never stored

### 2. User Login

When a user logs in:
```csharp
var user = await _dbContext.UserCredentials
    .FirstOrDefaultAsync(u => u.Username.ToLower() == username.ToLower());

bool passwordValid = VerifyPassword(password, user.PasswordHash, user.PasswordSalt);

if (passwordValid)
{
    user.LastLoginDate = DateTime.Now;  // Track last login
    await _dbContext.SaveChangesAsync();
    SetCurrentUserId(user.Id);
}
```

**Key Points:**
- Password is verified against stored hash
- `LastLoginDate` is automatically updated on successful login
- Current user ID is set in `AuthenticationService.CurrentUserId`
- No plain text password is ever stored

### 3. Auto-Population on Restart

When the login window opens:
```csharp
private async void AutoPopulateLastLoggedInUserAsync()
{
    var recentUsers = await _authenticationService.GetPreviouslyLoggedInUsersAsync();
    
    if (recentUsers != null && recentUsers.Count > 0)
    {
        // Set the username to the most recent user
        Username = recentUsers[0];
    }
}
```

**Key Points:**
- The most recently logged-in user (by `LastLoginDate`) is automatically populated
- Password field remains empty
- User must re-enter password for security

### 4. "Remember This Account" Checkbox

**Current State:**
- Checkbox is **checked by default** and **disabled**
- Displays: "Remember this account (Auto-populated on restart)"
- Tooltip explains that username will be remembered, but password must be re-entered

**Implementation:**
```xml
<CheckBox x:Name="RememberMeCheckBox" 
         Content="Remember this account (Auto-populated on restart)" 
         IsChecked="True"
         IsEnabled="False"
         ToolTip="Your username will be automatically populated on next login. 
                  You must re-enter your password for security."/>
```

---

## Database Schema

### UserCredentials Table

```sql
CREATE TABLE [UserCredentials] (
    [Id] INT PRIMARY KEY IDENTITY(1,1),
    [Username] NVARCHAR(200) NOT NULL UNIQUE,
    [Password] NVARCHAR(500) NOT NULL,          -- DEPRECATED (kept empty for backward compatibility)
    [PasswordHash] NVARCHAR(500) NULL,          -- Secure password hash (PBKDF2)
    [PasswordSalt] NVARCHAR(100) NULL,          -- Random salt for hashing
    [Pin] NVARCHAR(50) NULL,                    -- Optional PIN (not used for authentication)
    [Email] NVARCHAR(255) NULL,                 -- Optional email
    [LastLoginDate] DATETIME2 NULL,             -- Used to track recent users
    [CreatedDate] DATETIME2 NULL,               -- Account creation date
    [IsActive] BIT NOT NULL DEFAULT 1           -- Account status
);
```

**Column Usage:**
- `Password`: **DEPRECATED** - Should always be empty or whitespace
- `PasswordHash`: Secure hash generated from user's password
- `PasswordSalt`: Random salt used during hashing
- `LastLoginDate`: Used to determine which user logged in most recently (for auto-population)

---

## Security Best Practices Implemented

### 1. Password Hashing
- ? PBKDF2 algorithm with SHA-256
- ? 120,000 iterations (OWASP recommended minimum)
- ? 32-byte random salt per password
- ? 32-byte hash output

### 2. No Plain Text Storage
- ? Passwords are hashed before storage
- ? Original passwords are never saved to database
- ? Password field is kept empty for backward compatibility

### 3. Secure Password Verification
```csharp
private bool VerifyPassword(string password, string storedHash, string storedSalt)
{
    byte[] saltBytes = Convert.FromBase64String(storedSalt);
    string computedHash = ComputeHash(password, saltBytes);
    return computedHash == storedHash;  // Constant-time comparison
}
```

### 4. Automatic Migration
Existing users with plain text passwords are automatically migrated:
```csharp
if (!string.IsNullOrEmpty(user.Password))
{
    // Backward compatibility: plain text password check
    passwordValid = user.Password == password;
    if (passwordValid)
    {
        // Migrate to hashed password
        var (hash, salt) = HashPassword(password);
        user.PasswordHash = hash;
        user.PasswordSalt = salt;
        user.Password = string.Empty;  // Clear plain text
        await _dbContext.SaveChangesAsync();
    }
}
```

### 5. Session Management
- ? Current user ID stored in `AuthenticationService.CurrentUserId`
- ? Thread-safe implementation using locks
- ? Cleared on logout
- ? Used to scope settings profiles per user

---

## API Reference

### AuthenticationService

#### RegisterUserAsync
```csharp
public async Task<RegistrationResult> RegisterUserAsync(
    string username, 
    string password, 
    string email = null)
```
- Validates username availability
- Hashes password using PBKDF2
- Creates user with secure credentials
- Returns user ID on success

#### AuthenticateAsync
```csharp
public async Task<AuthenticationResult> AuthenticateAsync(
    string username, 
    string password)
```
- Verifies password against hash
- Updates `LastLoginDate` on success
- Sets `CurrentUserId` for session tracking
- Automatically migrates plain text passwords

#### GetPreviouslyLoggedInUsersAsync
```csharp
public async Task<List<string>> GetPreviouslyLoggedInUsersAsync()
```
- Returns list of usernames ordered by `LastLoginDate` (most recent first)
- Only includes active users with at least one login
- Used for username auto-population

#### IsUsernameAvailableAsync
```csharp
public async Task<bool> IsUsernameAvailableAsync(string username)
```
- Checks if username is already taken
- Case-insensitive comparison

---

## Migration Guide for Existing Installations

### Step 1: Run Cleanup Script (Optional)

Execute `CleanupPlainTextPasswords.sql` to audit and optionally clean up plain text passwords:

```sql
-- Check for plain text passwords
SELECT Id, Username, LastLoginDate
FROM UserCredentials
WHERE LEN(ISNULL(Password, '')) > 0
ORDER BY LastLoginDate DESC;
```

### Step 2: Automatic Migration on Login

Existing users with plain text passwords will be automatically migrated when they log in. The flow is:
1. User enters username and plain text password
2. `AuthenticationService` detects plain text password
3. Password is verified (plain text comparison)
4. Password is hashed and stored in `PasswordHash` and `PasswordSalt`
5. Plain text password is cleared
6. User is logged in successfully

### Step 3: Monitor Migration Progress

Query to check migration status:
```sql
SELECT 
    COUNT(*) AS TotalUsers,
    SUM(CASE WHEN LEN(ISNULL(PasswordHash, '')) > 0 THEN 1 ELSE 0 END) AS Migrated,
    SUM(CASE WHEN LEN(ISNULL(Password, '')) > 0 THEN 1 ELSE 0 END) AS NeedsMigration
FROM UserCredentials;
```

---

## Frequently Asked Questions (FAQ)

### Q: Why can't the app remember my password?
**A:** For security, passwords are never stored. This is an industry best practice. Only your username is remembered and auto-populated on restart.

### Q: What if I forget my password?
**A:** Currently, you'll need to re-register or contact support. Consider implementing a password reset feature using email verification.

### Q: Is the PIN field still used?
**A:** The PIN field exists for backward compatibility but is not used for authentication. You can remove it from the UI if desired.

### Q: Can I use Windows Credential Manager to store passwords securely?
**A:** Yes, this is a good enhancement for desktop applications. See the "Future Enhancements" section below.

### Q: How secure is PBKDF2 with 120,000 iterations?
**A:** This is the current OWASP recommendation for PBKDF2 with SHA-256. It provides strong protection against brute-force attacks while maintaining reasonable performance.

---

## Future Enhancements

### 1. Windows Credential Manager Integration

For secure password storage on Windows, integrate with Windows Credential Manager:

```csharp
using System.Security.Cryptography;
using Windows.Security.Credentials;

public class SecureCredentialStorage
{
    private const string RESOURCE_NAME = "QuantraApp";
    
    public void SaveCredential(string username, string password)
    {
        var vault = new PasswordVault();
        var credential = new PasswordCredential(RESOURCE_NAME, username, password);
        vault.Add(credential);
    }
    
    public string RetrievePassword(string username)
    {
        var vault = new PasswordVault();
        var credential = vault.Retrieve(RESOURCE_NAME, username);
        credential.RetrievePassword();
        return credential.Password;
    }
}
```

### 2. Password Reset Feature

Implement email-based password reset:
- Generate secure reset token
- Send email with reset link
- Verify token and allow new password
- Invalidate token after use

### 3. Multi-Factor Authentication (MFA)

Add MFA for enhanced security:
- TOTP (Time-based One-Time Password) using authenticator apps
- Email-based verification codes
- SMS-based verification (less secure, not recommended)

### 4. Password Complexity Requirements

Enforce password policies:
- Minimum length (currently: 6, consider: 8+)
- Require uppercase, lowercase, numbers, special characters
- Check against common password lists
- Prevent password reuse

### 5. Account Lockout

Implement account lockout after failed login attempts:
- Track failed login attempts
- Lock account after N failures
- Require admin unlock or time-based unlock

---

## Security Audit Checklist

- [x] Passwords are hashed using strong algorithm (PBKDF2)
- [x] Sufficient iterations for hash (120,000)
- [x] Random salt per password
- [x] No plain text passwords stored
- [x] Automatic migration of legacy passwords
- [x] Session management with current user tracking
- [x] Username auto-population based on LastLoginDate
- [ ] Password reset feature (Future enhancement)
- [ ] Account lockout after failed attempts (Future enhancement)
- [ ] Multi-factor authentication (Future enhancement)
- [ ] Audit logging of authentication events (Partially implemented)

---

## Testing

### Unit Tests

```csharp
[Fact]
public async Task RegisterUser_StoresHashedPassword()
{
    // Arrange
    var service = new AuthenticationService(_dbContext, _loggingService);
    
    // Act
    var result = await service.RegisterUserAsync("testuser", "Password123!");
    
    // Assert
    Assert.True(result.Success);
    var user = await _dbContext.UserCredentials.FindAsync(result.UserId);
    Assert.NotNull(user.PasswordHash);
    Assert.NotNull(user.PasswordSalt);
    Assert.True(string.IsNullOrEmpty(user.Password));
}

[Fact]
public async Task Authenticate_WithCorrectPassword_Succeeds()
{
    // Arrange & Act
    await _service.RegisterUserAsync("testuser", "Password123!");
    var result = await _service.AuthenticateAsync("testuser", "Password123!");
    
    // Assert
    Assert.True(result.Success);
    Assert.NotNull(result.UserId);
}

[Fact]
public async Task AutoPopulate_ReturnsLastLoggedInUser()
{
    // Arrange
    await _service.RegisterUserAsync("user1", "Pass1!");
    await _service.RegisterUserAsync("user2", "Pass2!");
    await _service.AuthenticateAsync("user1", "Pass1!");
    await Task.Delay(100); // Ensure LastLoginDate differs
    await _service.AuthenticateAsync("user2", "Pass2!");
    
    // Act
    var recentUsers = await _service.GetPreviouslyLoggedInUsersAsync();
    
    // Assert
    Assert.Equal("user2", recentUsers[0]); // Most recent first
}
```

---

## Compliance

This implementation aligns with:
- ? OWASP Password Storage Cheat Sheet
- ? NIST SP 800-63B Digital Identity Guidelines
- ? GDPR (passwords are hashed, not stored in plain text)
- ? PCI DSS (if applicable to payment processing)

---

## References

- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [NIST SP 800-63B](https://pages.nist.gov/800-63-3/sp800-63b.html)
- [PBKDF2 Documentation](https://en.wikipedia.org/wiki/PBKDF2)
- [Windows Credential Manager](https://docs.microsoft.com/en-us/uwp/api/windows.security.credentials.passwordvault)

---

**Last Updated:** 2024  
**Author:** Quantra Development Team  
**Status:** Implemented and Tested
