-- =============================================
-- Create Table: EarningsCalendar
-- Description: Stores earnings calendar data for stock symbols.
--              Used for TFT model known future inputs - earnings dates we know ahead of time.
-- =============================================

-- Drop table if it exists (optional - comment out if you want to preserve existing data)
-- DROP TABLE IF EXISTS [dbo].[EarningsCalendar];

-- Create the EarningsCalendar table
CREATE TABLE [dbo].[EarningsCalendar] (
    [Id] INT IDENTITY(1,1) NOT NULL,
    [Symbol] NVARCHAR(10) NOT NULL,
    [EarningsDate] DATETIME2 NOT NULL,
    [FiscalQuarter] NVARCHAR(10) NULL,
    [EPSEstimate] DECIMAL(10, 4) NULL,
    [LastUpdated] DATETIME2 NOT NULL,
    
    -- Primary Key
    CONSTRAINT [PK_EarningsCalendar] PRIMARY KEY CLUSTERED ([Id] ASC)
);

-- Create index on Symbol for faster lookups
CREATE NONCLUSTERED INDEX [IX_EarningsCalendar_Symbol] 
    ON [dbo].[EarningsCalendar] ([Symbol] ASC);

-- Create index on EarningsDate for date-based queries
CREATE NONCLUSTERED INDEX [IX_EarningsCalendar_EarningsDate] 
    ON [dbo].[EarningsCalendar] ([EarningsDate] ASC);

-- Create composite index for common query pattern (Symbol + EarningsDate)
CREATE NONCLUSTERED INDEX [IX_EarningsCalendar_Symbol_EarningsDate] 
    ON [dbo].[EarningsCalendar] ([Symbol] ASC, [EarningsDate] ASC);

GO
