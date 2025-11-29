-- Migration script to add ChatHistoryId and UserQuery columns to StockPredictions table
-- These columns support the MarketChat AI integration feature
-- Run this script against your database to add the missing columns

USE [master]
GO

-- Check if ChatHistoryId column exists before adding it
IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[StockPredictions]') AND name = 'ChatHistoryId')
BEGIN
    ALTER TABLE [dbo].[StockPredictions]
    ADD [ChatHistoryId] INT NULL
    
    PRINT 'Added ChatHistoryId column to StockPredictions table'
END
ELSE
BEGIN
    PRINT 'ChatHistoryId column already exists in StockPredictions table'
END
GO

-- Check if UserQuery column exists before adding it
IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[dbo].[StockPredictions]') AND name = 'UserQuery')
BEGIN
    ALTER TABLE [dbo].[StockPredictions]
    ADD [UserQuery] NVARCHAR(1000) NULL
    
    PRINT 'Added UserQuery column to StockPredictions table'
END
ELSE
BEGIN
    PRINT 'UserQuery column already exists in StockPredictions table'
END
GO

-- Create index on ChatHistoryId for better query performance
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE object_id = OBJECT_ID(N'[dbo].[StockPredictions]') AND name = N'IX_StockPredictions_ChatHistoryId')
BEGIN
    CREATE NONCLUSTERED INDEX [IX_StockPredictions_ChatHistoryId]
    ON [dbo].[StockPredictions] ([ChatHistoryId])
    
    PRINT 'Created index IX_StockPredictions_ChatHistoryId'
END
ELSE
BEGIN
    PRINT 'Index IX_StockPredictions_ChatHistoryId already exists'
END
GO

-- Add foreign key constraint to ChatHistory table (if ChatHistory table exists)
IF EXISTS (SELECT * FROM sys.tables WHERE name = 'ChatHistory')
BEGIN
    IF NOT EXISTS (SELECT * FROM sys.foreign_keys WHERE name = 'FK_StockPredictions_ChatHistory')
    BEGIN
        ALTER TABLE [dbo].[StockPredictions]
        ADD CONSTRAINT [FK_StockPredictions_ChatHistory]
        FOREIGN KEY ([ChatHistoryId])
        REFERENCES [dbo].[ChatHistory] ([Id])
        ON DELETE SET NULL
        
        PRINT 'Added foreign key constraint FK_StockPredictions_ChatHistory'
    END
    ELSE
    BEGIN
        PRINT 'Foreign key constraint FK_StockPredictions_ChatHistory already exists'
    END
END
ELSE
BEGIN
    PRINT 'WARNING: ChatHistory table does not exist - skipping foreign key constraint'
    PRINT 'You may need to create the ChatHistory table before adding the foreign key'
END
GO

PRINT 'Migration completed successfully'
PRINT 'ChatHistoryId and UserQuery columns have been added to StockPredictions table'
GO
