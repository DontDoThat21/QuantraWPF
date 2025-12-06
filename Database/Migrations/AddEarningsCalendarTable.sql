-- Migration: Add EarningsCalendar table for TFT model known future inputs
-- This table stores earnings calendar data for stocks to enable TFT model
-- to use known future inputs like days until next earnings announcement

-- Create the EarningsCalendar table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'EarningsCalendar')
BEGIN
    CREATE TABLE EarningsCalendar (
        Id INT PRIMARY KEY IDENTITY(1,1),
        Symbol NVARCHAR(10) NOT NULL,
        EarningsDate DATE NOT NULL,
        FiscalQuarter NVARCHAR(10),
        EPSEstimate DECIMAL(10, 4),
        LastUpdated DATETIME2 NOT NULL
    );

    -- Create index on Symbol for faster lookups by stock symbol
    CREATE INDEX IX_EarningsCalendar_Symbol ON EarningsCalendar (Symbol);

    -- Create index on EarningsDate for date-based queries
    CREATE INDEX IX_EarningsCalendar_Date ON EarningsCalendar (EarningsDate);

    -- Create composite index for common query pattern (symbol + date)
    CREATE INDEX IX_EarningsCalendar_Symbol_Date ON EarningsCalendar (Symbol, EarningsDate);

    PRINT 'Created EarningsCalendar table with indexes';
END
ELSE
BEGIN
    PRINT 'EarningsCalendar table already exists';
END
