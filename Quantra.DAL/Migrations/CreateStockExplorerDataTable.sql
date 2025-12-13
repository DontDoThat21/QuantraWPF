-- Migration: Create StockExplorerData Table
-- Date: 2025-12-13
-- Description: Creates the StockExplorerData table for storing stock data displayed in StockExplorer view
-- This table is separate from StockDataCache which is used for predictions

-- Check if table exists before creating
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[StockExplorerData]') AND type in (N'U'))
BEGIN
    CREATE TABLE [dbo].[StockExplorerData] (
        [Id] INT IDENTITY(1,1) PRIMARY KEY,
        [Symbol] NVARCHAR(20) NOT NULL,
        [Name] NVARCHAR(500) NULL,
        [Price] FLOAT NOT NULL DEFAULT 0,
        [Change] FLOAT NOT NULL DEFAULT 0,
        [ChangePercent] FLOAT NOT NULL DEFAULT 0,
        [DayHigh] FLOAT NOT NULL DEFAULT 0,
        [DayLow] FLOAT NOT NULL DEFAULT 0,
        [MarketCap] FLOAT NOT NULL DEFAULT 0,
        [Volume] FLOAT NOT NULL DEFAULT 0,
        [Sector] NVARCHAR(200) NULL,
        [RSI] FLOAT NOT NULL DEFAULT 0,
        [PERatio] FLOAT NOT NULL DEFAULT 0,
        [VWAP] FLOAT NOT NULL DEFAULT 0,
        [Date] DATETIME NOT NULL DEFAULT GETDATE(),
        [LastUpdated] DATETIME NOT NULL DEFAULT GETDATE(),
        [LastAccessed] DATETIME NOT NULL DEFAULT GETDATE(),
        [Timestamp] DATETIME NOT NULL DEFAULT GETDATE(),
        [CacheTime] DATETIME NULL,
        CONSTRAINT [UQ_StockExplorerData_Symbol] UNIQUE ([Symbol])
    );

    -- Create index on Symbol for faster lookups
    CREATE INDEX [IX_StockExplorerData_Symbol] ON [dbo].[StockExplorerData] ([Symbol]);

    -- Create index on LastUpdated for auto-refresh queries
    CREATE INDEX [IX_StockExplorerData_LastUpdated] ON [dbo].[StockExplorerData] ([LastUpdated]);

    PRINT 'StockExplorerData table created successfully.';
END
ELSE
BEGIN
    PRINT 'StockExplorerData table already exists.';
END
GO
