-- Migration: Add MarketContext Table for Caching Market Context Data
-- Purpose: Store cached market context data (S&P 500, VIX, Treasury Yields, Market Breadth) 
--          for TFT model predictions, refreshed every 15 minutes during market hours
-- Date: 2024

-- Create MarketContext table
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'MarketContext')
BEGIN
    CREATE TABLE MarketContext (
        Id INT PRIMARY KEY IDENTITY(1,1),
        
        -- Timestamp for the market context data point
        Timestamp DATETIME2 NOT NULL,
        
        -- S&P 500 (SPY) data
        SP500_Price DECIMAL(10, 2) NULL,
        SP500_Return DECIMAL(10, 6) NULL,
        
        -- VIX (Volatility Index)
        VIX DECIMAL(10, 2) NULL,
        VolatilityRegime INT NULL,  -- 0=Low, 1=Normal, 2=Elevated, 3=High
        
        -- Treasury Yields
        TreasuryYield_10Y DECIMAL(10, 4) NULL,
        TreasuryYield_2Y DECIMAL(10, 4) NULL,  -- For yield curve analysis
        
        -- Market Breadth
        MarketBreadth DECIMAL(10, 4) NULL,
        AdvancingCount INT NULL,
        DecliningCount INT NULL,
        
        -- Federal Funds Rate
        FederalFundsRate DECIMAL(10, 4) NULL,
        
        -- Metadata
        LastUpdated DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        -- Index on Timestamp for efficient time-based queries
        INDEX IX_MarketContext_Timestamp (Timestamp),
        INDEX IX_MarketContext_LastUpdated (LastUpdated)
    );

    PRINT 'Created MarketContext table';
END
ELSE
BEGIN
    PRINT 'MarketContext table already exists';
END

-- Create SectorETFContext table for sector-specific market context
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'SectorETFContext')
BEGIN
    CREATE TABLE SectorETFContext (
        Id INT PRIMARY KEY IDENTITY(1,1),
        
        -- Timestamp for the sector data point
        Timestamp DATETIME2 NOT NULL,
        
        -- Sector identification
        SectorCode INT NOT NULL,  -- Matches AlphaVantageService.GetSectorCode() output
        SectorName NVARCHAR(100) NOT NULL,
        ETFSymbol NVARCHAR(10) NOT NULL,
        
        -- Price data
        ETFPrice DECIMAL(10, 2) NULL,
        ETFReturn DECIMAL(10, 6) NULL,  -- 1-day return percentage
        
        -- Metadata
        LastUpdated DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        -- Indexes for efficient queries
        INDEX IX_SectorETFContext_Timestamp (Timestamp),
        INDEX IX_SectorETFContext_SectorCode (SectorCode),
        INDEX IX_SectorETFContext_ETFSymbol (ETFSymbol),
        INDEX IX_SectorETFContext_LastUpdated (LastUpdated)
    );

    PRINT 'Created SectorETFContext table';
END
ELSE
BEGIN
    PRINT 'SectorETFContext table already exists';
END

-- Insert initial sector ETF mappings for reference
-- Note: These are the standard Select Sector SPDR ETFs
/*
Sector ETF Mappings:
- Technology (0): XLK
- Healthcare (1): XLV
- Financial (2): XLF
- Consumer Discretionary (3): XLY
- Consumer Staples (4): XLP
- Industrials (5): XLI
- Energy (6): XLE
- Materials (7): XLB
- Real Estate (8): XLRE
- Utilities (9): XLU
- Communication Services (10): XLC
- Default/Unknown: SPY (S&P 500)
*/

PRINT 'MarketContext migration completed successfully';
