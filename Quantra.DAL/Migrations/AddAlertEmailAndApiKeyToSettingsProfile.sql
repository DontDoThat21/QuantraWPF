-- Migration script to add AlertEmail, AlphaVantageApiKey, and email alert settings to SettingsProfiles table
-- Run this script against your database to add the missing columns

USE [master]
GO

-- Check if columns exist before adding them
IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'AlertEmail')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [AlertEmail] NVARCHAR(255) NULL
    
    -- Set default value for existing rows
    UPDATE [dbo].[SettingsProfiles]
    SET [AlertEmail] = 'test@gmail.com'
    WHERE [AlertEmail] IS NULL
    
    PRINT 'Added AlertEmail column'
END
GO

IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'AlphaVantageApiKey')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [AlphaVantageApiKey] NVARCHAR(255) NULL
    
    -- Set default value for existing rows
    UPDATE [dbo].[SettingsProfiles]
    SET [AlphaVantageApiKey] = ''
    WHERE [AlphaVantageApiKey] IS NULL
    
    PRINT 'Added AlphaVantageApiKey column'
END
GO

IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'EnableEmailAlerts')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [EnableEmailAlerts] BIT NOT NULL DEFAULT 0
    
    PRINT 'Added EnableEmailAlerts column'
END
GO

IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'EnableStandardAlertEmails')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [EnableStandardAlertEmails] BIT NOT NULL DEFAULT 0
    
    PRINT 'Added EnableStandardAlertEmails column'
END
GO

IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'EnableOpportunityAlertEmails')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [EnableOpportunityAlertEmails] BIT NOT NULL DEFAULT 0
    
    PRINT 'Added EnableOpportunityAlertEmails column'
END
GO

IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'EnablePredictionAlertEmails')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [EnablePredictionAlertEmails] BIT NOT NULL DEFAULT 0
    
    PRINT 'Added EnablePredictionAlertEmails column'
END
GO

IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'EnableGlobalAlertEmails')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [EnableGlobalAlertEmails] BIT NOT NULL DEFAULT 0
    
    PRINT 'Added EnableGlobalAlertEmails column'
END
GO

IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'EnableSystemHealthAlertEmails')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [EnableSystemHealthAlertEmails] BIT NOT NULL DEFAULT 0
    
    PRINT 'Added EnableSystemHealthAlertEmails column'
END
GO

IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'EnableVixMonitoring')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [EnableVixMonitoring] BIT NOT NULL DEFAULT 1
    
    PRINT 'Added EnableVixMonitoring column'
END
GO

PRINT 'Migration completed successfully'
GO
