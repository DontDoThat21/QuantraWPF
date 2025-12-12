-- Migration script to add SavedFilters table to store user's saved filter configurations
-- Run this script against your database to create the table

USE [master]
GO

-- Create SavedFilters table
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[SavedFilters]') AND type in (N'U'))
BEGIN
    CREATE TABLE [dbo].[SavedFilters](
        [Id] INT IDENTITY(1,1) NOT NULL,
        [Name] NVARCHAR(100) NOT NULL,
        [Description] NVARCHAR(500) NULL,
        [UserId] INT NULL,
        [IsSystemFilter] BIT NOT NULL DEFAULT 0,
        [SymbolFilter] NVARCHAR(100) NULL,
        [PriceFilter] NVARCHAR(100) NULL,
        [PeRatioFilter] NVARCHAR(100) NULL,
        [VwapFilter] NVARCHAR(100) NULL,
        [RsiFilter] NVARCHAR(100) NULL,
        [ChangePercentFilter] NVARCHAR(100) NULL,
        [MarketCapFilter] NVARCHAR(100) NULL,
        [CreatedDate] DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        [ModifiedDate] DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        CONSTRAINT [PK_SavedFilters] PRIMARY KEY CLUSTERED ([Id] ASC),
        CONSTRAINT [FK_SavedFilters_UserCredentials] FOREIGN KEY([UserId])
            REFERENCES [dbo].[UserCredentials] ([Id])
            ON DELETE SET NULL
    )

    PRINT 'SavedFilters table created successfully'
END
ELSE
BEGIN
    PRINT 'SavedFilters table already exists'
END
GO

-- Create index on UserId for faster queries
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_SavedFilters_UserId' AND object_id = OBJECT_ID('dbo.SavedFilters'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_SavedFilters_UserId]
    ON [dbo].[SavedFilters] ([UserId] ASC)

    PRINT 'Index IX_SavedFilters_UserId created successfully'
END
GO

-- Create index on Name for faster lookups
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_SavedFilters_Name' AND object_id = OBJECT_ID('dbo.SavedFilters'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_SavedFilters_Name]
    ON [dbo].[SavedFilters] ([Name] ASC)

    PRINT 'Index IX_SavedFilters_Name created successfully'
END
GO

-- Insert some default system filters
IF NOT EXISTS (SELECT * FROM [dbo].[SavedFilters] WHERE [Name] = 'RSI Oversold' AND [IsSystemFilter] = 1)
BEGIN
    INSERT INTO [dbo].[SavedFilters] ([Name], [Description], [IsSystemFilter], [RsiFilter], [CreatedDate], [ModifiedDate])
    VALUES ('RSI Oversold', 'Stocks with RSI below 30 (oversold)', 1, '<30', GETUTCDATE(), GETUTCDATE())

    PRINT 'Default filter "RSI Oversold" created'
END
GO

IF NOT EXISTS (SELECT * FROM [dbo].[SavedFilters] WHERE [Name] = 'RSI Overbought' AND [IsSystemFilter] = 1)
BEGIN
    INSERT INTO [dbo].[SavedFilters] ([Name], [Description], [IsSystemFilter], [RsiFilter], [CreatedDate], [ModifiedDate])
    VALUES ('RSI Overbought', 'Stocks with RSI above 70 (overbought)', 1, '>70', GETUTCDATE(), GETUTCDATE())

    PRINT 'Default filter "RSI Overbought" created'
END
GO

IF NOT EXISTS (SELECT * FROM [dbo].[SavedFilters] WHERE [Name] = 'Low P/E' AND [IsSystemFilter] = 1)
BEGIN
    INSERT INTO [dbo].[SavedFilters] ([Name], [Description], [IsSystemFilter], [PeRatioFilter], [CreatedDate], [ModifiedDate])
    VALUES ('Low P/E', 'Stocks with P/E ratio below 15', 1, '<15', GETUTCDATE(), GETUTCDATE())

    PRINT 'Default filter "Low P/E" created'
END
GO

IF NOT EXISTS (SELECT * FROM [dbo].[SavedFilters] WHERE [Name] = 'High P/E Growth' AND [IsSystemFilter] = 1)
BEGIN
    INSERT INTO [dbo].[SavedFilters] ([Name], [Description], [IsSystemFilter], [PeRatioFilter], [CreatedDate], [ModifiedDate])
    VALUES ('High P/E Growth', 'Stocks with P/E ratio above 30', 1, '>30', GETUTCDATE(), GETUTCDATE())

    PRINT 'Default filter "High P/E Growth" created'
END
GO

IF NOT EXISTS (SELECT * FROM [dbo].[SavedFilters] WHERE [Name] = 'Mid-Cap Value' AND [IsSystemFilter] = 1)
BEGIN
    INSERT INTO [dbo].[SavedFilters] ([Name], [Description], [IsSystemFilter], [MarketCapFilter], [PeRatioFilter], [CreatedDate], [ModifiedDate])
    VALUES ('Mid-Cap Value', 'Mid-cap stocks with low P/E', 1, '>2000000000&<10000000000', '<20', GETUTCDATE(), GETUTCDATE())

    PRINT 'Default filter "Mid-Cap Value" created'
END
GO

PRINT 'Migration completed successfully'
GO
