-- Migration script to add EnableStockExplorerAutoRefresh column to SettingsProfiles table
-- This column stores the user's preference for Stock Explorer auto-refresh functionality
-- Run this script against your database to add the column

USE [master]
GO

-- Check if column exists before adding it
IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[SettingsProfiles]') AND name = 'EnableStockExplorerAutoRefresh')
BEGIN
    ALTER TABLE [dbo].[SettingsProfiles]
    ADD [EnableStockExplorerAutoRefresh] BIT NOT NULL DEFAULT 0

    -- Set default value for existing rows (0 = disabled by default)
    UPDATE [dbo].[SettingsProfiles]
    SET [EnableStockExplorerAutoRefresh] = 0
    WHERE [EnableStockExplorerAutoRefresh] IS NULL

    PRINT 'Added EnableStockExplorerAutoRefresh column with default value of 0 (disabled)'
END
ELSE
BEGIN
    PRINT 'EnableStockExplorerAutoRefresh column already exists'
END
GO

PRINT 'Migration completed successfully'
GO
