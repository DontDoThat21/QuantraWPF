-- Migration: Add CompanyOverviews Table
-- Purpose: Cache company overview data for TFT static metadata features (Sector, MarketCap, Exchange)
-- Data is cached for 7 days to avoid excessive API calls
-- Date: 2025-12-06

-- Check if table exists before creating
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[CompanyOverviews]') AND type in (N'U'))
BEGIN
    CREATE TABLE [dbo].[CompanyOverviews] (
        [Id] INT IDENTITY(1,1) NOT NULL,
        [Symbol] NVARCHAR(10) NOT NULL,
        [Name] NVARCHAR(255) NULL,
        [Sector] NVARCHAR(50) NULL,
        [Industry] NVARCHAR(100) NULL,
        [MarketCapitalization] BIGINT NULL,
        [Exchange] NVARCHAR(50) NULL,
        [Country] NVARCHAR(50) NULL,
        [Currency] NVARCHAR(10) NULL,
        [Description] NVARCHAR(MAX) NULL,
        [FiscalYearEnd] NVARCHAR(20) NULL,
        [PERatio] DECIMAL(18,4) NULL,
        [PEGRatio] DECIMAL(18,4) NULL,
        [BookValue] DECIMAL(18,4) NULL,
        [DividendPerShare] DECIMAL(18,4) NULL,
        [DividendYield] DECIMAL(18,6) NULL,
        [EPS] DECIMAL(18,4) NULL,
        [Beta] DECIMAL(18,4) NULL,
        [Week52High] DECIMAL(18,4) NULL,
        [Week52Low] DECIMAL(18,4) NULL,
        [Day50MovingAverage] DECIMAL(18,4) NULL,
        [Day200MovingAverage] DECIMAL(18,4) NULL,
        [SharesOutstanding] BIGINT NULL,
        [LastUpdated] DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        CONSTRAINT [PK_CompanyOverviews] PRIMARY KEY CLUSTERED ([Id] ASC),
        CONSTRAINT [UQ_CompanyOverviews_Symbol] UNIQUE NONCLUSTERED ([Symbol] ASC)
    );

    PRINT 'Created CompanyOverviews table';
END
ELSE
BEGIN
    PRINT 'CompanyOverviews table already exists';
END
GO

-- Create index on Sector for filtering by sector
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_CompanyOverviews_Sector' AND object_id = OBJECT_ID('dbo.CompanyOverviews'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_CompanyOverviews_Sector] 
    ON [dbo].[CompanyOverviews] ([Sector] ASC);
    
    PRINT 'Created index IX_CompanyOverviews_Sector';
END
GO

-- Create index on Exchange for filtering by exchange
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_CompanyOverviews_Exchange' AND object_id = OBJECT_ID('dbo.CompanyOverviews'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_CompanyOverviews_Exchange] 
    ON [dbo].[CompanyOverviews] ([Exchange] ASC);
    
    PRINT 'Created index IX_CompanyOverviews_Exchange';
END
GO

-- Create index on MarketCapitalization for filtering by market cap category
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_CompanyOverviews_MarketCap' AND object_id = OBJECT_ID('dbo.CompanyOverviews'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_CompanyOverviews_MarketCap] 
    ON [dbo].[CompanyOverviews] ([MarketCapitalization] ASC);
    
    PRINT 'Created index IX_CompanyOverviews_MarketCap';
END
GO

-- Create index on LastUpdated for cache expiration queries
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_CompanyOverviews_LastUpdated' AND object_id = OBJECT_ID('dbo.CompanyOverviews'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_CompanyOverviews_LastUpdated] 
    ON [dbo].[CompanyOverviews] ([LastUpdated] ASC);
    
    PRINT 'Created index IX_CompanyOverviews_LastUpdated';
END
GO

-- Stored procedure to get or refresh company overview with cache logic
IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[usp_GetCompanyOverview]') AND type in (N'P'))
    DROP PROCEDURE [dbo].[usp_GetCompanyOverview];
GO

CREATE PROCEDURE [dbo].[usp_GetCompanyOverview]
    @Symbol NVARCHAR(10),
    @CacheDays INT = 7
AS
BEGIN
    SET NOCOUNT ON;

    -- Return cached data if valid (within cache period)
    SELECT 
        [Id],
        [Symbol],
        [Name],
        [Sector],
        [Industry],
        [MarketCapitalization],
        [Exchange],
        [Country],
        [Currency],
        [Description],
        [FiscalYearEnd],
        [PERatio],
        [PEGRatio],
        [BookValue],
        [DividendPerShare],
        [DividendYield],
        [EPS],
        [Beta],
        [Week52High],
        [Week52Low],
        [Day50MovingAverage],
        [Day200MovingAverage],
        [SharesOutstanding],
        [LastUpdated],
        CASE WHEN DATEDIFF(DAY, [LastUpdated], GETUTCDATE()) <= @CacheDays THEN 1 ELSE 0 END AS [IsCacheValid]
    FROM [dbo].[CompanyOverviews]
    WHERE [Symbol] = @Symbol;
END
GO

-- Stored procedure to upsert company overview data
IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[usp_UpsertCompanyOverview]') AND type in (N'P'))
    DROP PROCEDURE [dbo].[usp_UpsertCompanyOverview];
GO

CREATE PROCEDURE [dbo].[usp_UpsertCompanyOverview]
    @Symbol NVARCHAR(10),
    @Name NVARCHAR(255) = NULL,
    @Sector NVARCHAR(50) = NULL,
    @Industry NVARCHAR(100) = NULL,
    @MarketCapitalization BIGINT = NULL,
    @Exchange NVARCHAR(50) = NULL,
    @Country NVARCHAR(50) = NULL,
    @Currency NVARCHAR(10) = NULL,
    @Description NVARCHAR(MAX) = NULL,
    @FiscalYearEnd NVARCHAR(20) = NULL,
    @PERatio DECIMAL(18,4) = NULL,
    @PEGRatio DECIMAL(18,4) = NULL,
    @BookValue DECIMAL(18,4) = NULL,
    @DividendPerShare DECIMAL(18,4) = NULL,
    @DividendYield DECIMAL(18,6) = NULL,
    @EPS DECIMAL(18,4) = NULL,
    @Beta DECIMAL(18,4) = NULL,
    @Week52High DECIMAL(18,4) = NULL,
    @Week52Low DECIMAL(18,4) = NULL,
    @Day50MovingAverage DECIMAL(18,4) = NULL,
    @Day200MovingAverage DECIMAL(18,4) = NULL,
    @SharesOutstanding BIGINT = NULL
AS
BEGIN
    SET NOCOUNT ON;

    MERGE [dbo].[CompanyOverviews] AS target
    USING (SELECT @Symbol AS Symbol) AS source
    ON (target.[Symbol] = source.[Symbol])
    WHEN MATCHED THEN
        UPDATE SET
            [Name] = @Name,
            [Sector] = @Sector,
            [Industry] = @Industry,
            [MarketCapitalization] = @MarketCapitalization,
            [Exchange] = @Exchange,
            [Country] = @Country,
            [Currency] = @Currency,
            [Description] = @Description,
            [FiscalYearEnd] = @FiscalYearEnd,
            [PERatio] = @PERatio,
            [PEGRatio] = @PEGRatio,
            [BookValue] = @BookValue,
            [DividendPerShare] = @DividendPerShare,
            [DividendYield] = @DividendYield,
            [EPS] = @EPS,
            [Beta] = @Beta,
            [Week52High] = @Week52High,
            [Week52Low] = @Week52Low,
            [Day50MovingAverage] = @Day50MovingAverage,
            [Day200MovingAverage] = @Day200MovingAverage,
            [SharesOutstanding] = @SharesOutstanding,
            [LastUpdated] = GETUTCDATE()
    WHEN NOT MATCHED THEN
        INSERT (
            [Symbol], [Name], [Sector], [Industry], [MarketCapitalization],
            [Exchange], [Country], [Currency], [Description], [FiscalYearEnd],
            [PERatio], [PEGRatio], [BookValue], [DividendPerShare], [DividendYield],
            [EPS], [Beta], [Week52High], [Week52Low], [Day50MovingAverage],
            [Day200MovingAverage], [SharesOutstanding], [LastUpdated]
        )
        VALUES (
            @Symbol, @Name, @Sector, @Industry, @MarketCapitalization,
            @Exchange, @Country, @Currency, @Description, @FiscalYearEnd,
            @PERatio, @PEGRatio, @BookValue, @DividendPerShare, @DividendYield,
            @EPS, @Beta, @Week52High, @Week52Low, @Day50MovingAverage,
            @Day200MovingAverage, @SharesOutstanding, GETUTCDATE()
        );
END
GO

-- Stored procedure to clean up expired cache entries
IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[usp_CleanupExpiredCompanyOverviews]') AND type in (N'P'))
    DROP PROCEDURE [dbo].[usp_CleanupExpiredCompanyOverviews];
GO

CREATE PROCEDURE [dbo].[usp_CleanupExpiredCompanyOverviews]
    @CacheDays INT = 30  -- Default: clean entries older than 30 days
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @RowsDeleted INT;

    DELETE FROM [dbo].[CompanyOverviews]
    WHERE DATEDIFF(DAY, [LastUpdated], GETUTCDATE()) > @CacheDays;

    SET @RowsDeleted = @@ROWCOUNT;

    SELECT @RowsDeleted AS RowsDeleted;
END
GO

PRINT 'Migration complete: AddCompanyOverviewsTable';
