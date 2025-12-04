-- Migration script to add AlphaVantageApiCallsPerMinute column to SettingsProfiles table
-- This column stores the user's Alpha Vantage API plan tier rate limit
-- Run this script against your database to add the column

USE [master]
GO

-- Check if column exists before adding it
IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'AlphaVantageApiCallsPerMinute')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [AlphaVantageApiCallsPerMinute] INT NOT NULL DEFAULT 75

    -- Set default value for existing rows (75 = standard plan)
    UPDATE [dbo].[SettingsProfiles]
    SET [AlphaVantageApiCallsPerMinute] = 75
    WHERE [AlphaVantageApiCallsPerMinute] IS NULL OR [AlphaVantageApiCallsPerMinute] = 0

    PRINT 'Added AlphaVantageApiCallsPerMinute column with default value of 75 (standard plan)'
END
ELSE
BEGIN
    PRINT 'AlphaVantageApiCallsPerMinute column already exists'
END
GO

PRINT 'Migration completed successfully'
GO
