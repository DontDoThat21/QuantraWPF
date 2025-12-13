-- Migration: Create StockConfigurations Table
-- Date: 2025-12-13
-- Description: Creates the StockConfigurations table for predefined stock symbol configurations

-- Check if table exists before creating
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[StockConfigurations]') AND type in (N'U'))
BEGIN
    CREATE TABLE [dbo].[StockConfigurations] (
        [Id] INT IDENTITY(1,1) PRIMARY KEY,
        [Symbol] NVARCHAR(20) NOT NULL,
        [Name] NVARCHAR(500) NULL,
        [Sector] NVARCHAR(200) NULL,
        [IsActive] BIT NOT NULL DEFAULT 1,
        [CreatedDate] DATETIME NOT NULL DEFAULT GETDATE(),
        [LastModified] DATETIME NULL,
        CONSTRAINT [UQ_StockConfigurations_Symbol] UNIQUE ([Symbol])
    );

    -- Create index on Symbol for faster lookups
    CREATE INDEX [IX_StockConfigurations_Symbol] ON [dbo].[StockConfigurations] ([Symbol]);

    -- Create index on IsActive for filtering
    CREATE INDEX [IX_StockConfigurations_IsActive] ON [dbo].[StockConfigurations] ([IsActive]);

    PRINT 'StockConfigurations table created successfully.';
END
ELSE
BEGIN
    PRINT 'StockConfigurations table already exists.';
END
GO
