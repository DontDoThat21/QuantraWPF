-- ========================================
-- Cleanup Plain Text Passwords Script
-- ========================================
-- This script removes plain text passwords from the UserCredentials table
-- for security purposes. Only hashed passwords should be stored.
--
-- RUN THIS SCRIPT ONCE to clean up any existing plain text passwords.
-- After running this script, users will need to re-register or use
-- the password reset feature if one exists.
--
-- Date: 2024
-- Author: Quantra Development Team
-- ========================================

USE [Quantra]
GO

-- Step 1: Check for plain text passwords (where Password is not empty but PasswordHash is empty)
SELECT 
    Id,
    Username,
    CASE 
        WHEN LEN(ISNULL(Password, '')) > 0 THEN 'Has Plain Text Password'
        ELSE 'No Plain Text Password'
    END AS PasswordStatus,
    CASE 
        WHEN LEN(ISNULL(PasswordHash, '')) > 0 THEN 'Has Hashed Password'
        ELSE 'No Hashed Password'
    END AS HashStatus,
    LastLoginDate,
    CreatedDate
FROM UserCredentials
WHERE LEN(ISNULL(Password, '')) > 0
ORDER BY LastLoginDate DESC;

-- Step 2: OPTIONAL - Clear plain text passwords for users who already have hashed passwords
-- UNCOMMENT THE FOLLOWING LINES TO EXECUTE THE CLEANUP
/*
UPDATE UserCredentials
SET Password = ''
WHERE LEN(ISNULL(PasswordHash, '')) > 0  -- Only clear if hash exists
  AND LEN(ISNULL(Password, '')) > 0;     -- And plain text exists

PRINT 'Plain text passwords cleared for users with hashed passwords.';
*/

-- Step 3: OPTIONAL - For users without hashed passwords, you may want to:
-- Option A: Force them to re-register (delete their accounts)
-- Option B: Manually reset their passwords using the AuthenticationService
-- Option C: Leave them as-is and let the AuthenticationService migrate them on next login

-- Example: List users who need password migration
SELECT 
    Id,
    Username,
    Email,
    CreatedDate,
    LastLoginDate,
    'Needs Password Migration' AS Status
FROM UserCredentials
WHERE (LEN(ISNULL(PasswordHash, '')) = 0 OR PasswordHash IS NULL)
  AND LEN(ISNULL(Password, '')) > 0
ORDER BY LastLoginDate DESC;

-- ========================================
-- Notes:
-- ========================================
-- 1. The AuthenticationService automatically migrates plain text passwords
--    to hashed passwords when users log in successfully.
--
-- 2. After migration, the Password column is set to empty string.
--
-- 3. New registrations always use hashed passwords (PasswordHash + PasswordSalt).
--
-- 4. For maximum security, run Step 2 after confirming all active users
--    have successfully logged in at least once post-migration.
--
-- 5. Consider implementing a password reset feature for users who
--    cannot log in after this cleanup.
-- ========================================

-- Verification Query - Run this after cleanup to verify
SELECT 
    COUNT(*) AS TotalUsers,
    SUM(CASE WHEN LEN(ISNULL(PasswordHash, '')) > 0 THEN 1 ELSE 0 END) AS UsersWithHashedPasswords,
    SUM(CASE WHEN LEN(ISNULL(Password, '')) > 0 THEN 1 ELSE 0 END) AS UsersWithPlainTextPasswords,
    SUM(CASE WHEN LEN(ISNULL(PasswordHash, '')) > 0 AND LEN(ISNULL(Password, '')) = 0 THEN 1 ELSE 0 END) AS SecureUsersOnly
FROM UserCredentials;

GO
