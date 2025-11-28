# Fix for AlertEmail and AlphaVantageApiKey Not Saving

## Problem
The `AlertEmail` and `AlphaVantageApiKey` fields were not being saved to the database when updating settings profiles because the `SettingsProfiles` table was missing these columns.

## Solution
The following changes have been made:

### 1. Database Schema Update
Added the following columns to the `SettingsProfiles` table:
- `AlertEmail` (NVARCHAR(255))
- `AlphaVantageApiKey` (NVARCHAR(255))
- `EnableEmailAlerts` (BIT)
- `EnableStandardAlertEmails` (BIT)
- `EnableOpportunityAlertEmails` (BIT)
- `EnablePredictionAlertEmails` (BIT)
- `EnableGlobalAlertEmails` (BIT)
- `EnableSystemHealthAlertEmails` (BIT)
- `EnableVixMonitoring` (BIT)

### 2. Code Changes
Updated the following files:
- `Quantra.DAL/Data/Entities/SettingsProfile.cs` - Added missing properties to the entity
- `Quantra.DAL/Services/SettingsService.cs` - Updated mapping methods to include the new fields:
  - `MapToEntity()` - Maps from `DatabaseSettingsProfile` to `SettingsProfile` entity
  - `UpdateEntityFromProfile()` - Updates entity with new field values
  - `MapFromEntity()` - Maps from entity back to `DatabaseSettingsProfile`

## How to Apply

### Option 1: Run the Migration Script (Recommended)
1. Open SQL Server Management Studio (SSMS)
2. Connect to your database server
3. Open the file `AddAlertEmailAndApiKeyToSettingsProfile.sql`
4. Execute the script against your database

The script is safe to run multiple times as it checks if columns already exist before adding them.

### Option 2: Let EF Core Handle It
If you're using EF Core migrations:
1. Stop the application
2. Delete the database (if in development)
3. Run the application again - EF Core will recreate the database with the new schema

### Option 3: Manual Update
Execute the following SQL commands in SSMS:

```sql
USE [YourDatabaseName]
GO

ALTER TABLE [dbo].[SettingsProfiles] ADD [AlertEmail] NVARCHAR(255) NULL
ALTER TABLE [dbo].[SettingsProfiles] ADD [AlphaVantageApiKey] NVARCHAR(255) NULL
ALTER TABLE [dbo].[SettingsProfiles] ADD [EnableEmailAlerts] BIT NOT NULL DEFAULT 0
ALTER TABLE [dbo].[SettingsProfiles] ADD [EnableStandardAlertEmails] BIT NOT NULL DEFAULT 0
ALTER TABLE [dbo].[SettingsProfiles] ADD [EnableOpportunityAlertEmails] BIT NOT NULL DEFAULT 0
ALTER TABLE [dbo].[SettingsProfiles] ADD [EnablePredictionAlertEmails] BIT NOT NULL DEFAULT 0
ALTER TABLE [dbo].[SettingsProfiles] ADD [EnableGlobalAlertEmails] BIT NOT NULL DEFAULT 0
ALTER TABLE [dbo].[SettingsProfiles] ADD [EnableSystemHealthAlertEmails] BIT NOT NULL DEFAULT 0
ALTER TABLE [dbo].[SettingsProfiles] ADD [EnableVixMonitoring] BIT NOT NULL DEFAULT 1

-- Set default values for existing rows
UPDATE [dbo].[SettingsProfiles] SET [AlertEmail] = 'test@gmail.com' WHERE [AlertEmail] IS NULL
UPDATE [dbo].[SettingsProfiles] SET [AlphaVantageApiKey] = '' WHERE [AlphaVantageApiKey] IS NULL
GO
```

## Testing
After applying the migration:
1. Run the application
2. Open Settings window
3. Update the `AlertEmail` field
4. Update the `AlphaVantageApiKey` field (in the password box)
5. Close the settings window
6. Verify the changes are saved by checking the database or reopening the settings window

## Verification Query
Run this query to verify the columns were added:

```sql
SELECT 
    COLUMN_NAME, 
    DATA_TYPE, 
    IS_NULLABLE, 
    CHARACTER_MAXIMUM_LENGTH
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'SettingsProfiles'
AND COLUMN_NAME IN ('AlertEmail', 'AlphaVantageApiKey', 'EnableEmailAlerts', 
                     'EnableStandardAlertEmails', 'EnableOpportunityAlertEmails',
                     'EnablePredictionAlertEmails', 'EnableGlobalAlertEmails', 
                     'EnableSystemHealthAlertEmails', 'EnableVixMonitoring')
ORDER BY COLUMN_NAME
```

Expected result: 9 rows showing the new columns.

## Additional Notes
- The `AlertEmail` field is now stored in the database per profile
- The `AlphaVantageApiKey` is also stored per profile and set as an environment variable when saved
- All email alert enable/disable flags are now properly persisted
- The fix is backward compatible - existing profiles will work with default values

## Rollback (if needed)
If you need to rollback the changes:

```sql
USE [YourDatabaseName]
GO

ALTER TABLE [dbo].[SettingsProfiles] DROP COLUMN [AlertEmail]
ALTER TABLE [dbo].[SettingsProfiles] DROP COLUMN [AlphaVantageApiKey]
ALTER TABLE [dbo].[SettingsProfiles] DROP COLUMN [EnableEmailAlerts]
ALTER TABLE [dbo].[SettingsProfiles] DROP COLUMN [EnableStandardAlertEmails]
ALTER TABLE [dbo].[SettingsProfiles] DROP COLUMN [EnableOpportunityAlertEmails]
ALTER TABLE [dbo].[SettingsProfiles] DROP COLUMN [EnablePredictionAlertEmails]
ALTER TABLE [dbo].[SettingsProfiles] DROP COLUMN [EnableGlobalAlertEmails]
ALTER TABLE [dbo].[SettingsProfiles] DROP COLUMN [EnableSystemHealthAlertEmails]
ALTER TABLE [dbo].[SettingsProfiles] DROP COLUMN [EnableVixMonitoring]
GO
```

Then revert the code changes in the mentioned files.
