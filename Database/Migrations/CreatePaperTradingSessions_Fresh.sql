-- =====================================================
-- Create PaperTradingSessions Table - Fresh Install
-- For the enhanced paper trading system
-- =====================================================

USE [QuantraRelational]
GO

-- Drop existing table if it exists
IF OBJECT_ID('dbo.PaperTradingSessions', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing PaperTradingSessions table...'
    DROP TABLE dbo.PaperTradingSessions
    PRINT '  ? Table dropped'
END
GO

-- Create the PaperTradingSessions table
PRINT 'Creating PaperTradingSessions table...'

CREATE TABLE dbo.PaperTradingSessions (
    Id INT PRIMARY KEY IDENTITY(1,1),
    SessionId NVARCHAR(36) NOT NULL,
    [Name] NVARCHAR(200) NULL,
    InitialCash DECIMAL(18,4) NOT NULL,
    CashBalance DECIMAL(18,4) NOT NULL,
    RealizedPnL DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    IsActive BIT NOT NULL DEFAULT 1,
    StartedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    EndedAt DATETIME2 NULL,
    LastUpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE()
)
GO

-- Create unique index on SessionId
CREATE UNIQUE NONCLUSTERED INDEX IX_PaperTradingSessions_SessionId 
ON dbo.PaperTradingSessions(SessionId)
GO

-- Create index on IsActive for filtering active sessions
CREATE NONCLUSTERED INDEX IX_PaperTradingSessions_IsActive 
ON dbo.PaperTradingSessions(IsActive) 
INCLUDE (SessionId, StartedAt)
GO

PRINT '  ? PaperTradingSessions table created successfully!'
PRINT ''

-- Verify the schema
PRINT 'Current PaperTradingSessions schema:'
SELECT 
    c.name AS ColumnName,
    t.name AS DataType,
    c.max_length AS MaxLength,
    c.precision AS [Precision],
    c.scale AS Scale,
    c.is_nullable AS IsNullable,
    CASE WHEN pk.column_id IS NOT NULL THEN 'YES' ELSE 'NO' END AS IsPrimaryKey
FROM sys.columns c
INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
LEFT JOIN (
    SELECT ic.object_id, ic.column_id
    FROM sys.indexes i
    INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
    WHERE i.is_primary_key = 1
) pk ON c.object_id = pk.object_id AND c.column_id = pk.column_id
WHERE c.object_id = OBJECT_ID('dbo.PaperTradingSessions')
ORDER BY c.column_id
GO

-- Show indexes
PRINT ''
PRINT 'Indexes on PaperTradingSessions:'
SELECT 
    i.name AS IndexName,
    i.type_desc AS IndexType,
    i.is_unique AS IsUnique,
    STRING_AGG(c.name, ', ') WITHIN GROUP (ORDER BY ic.key_ordinal) AS Columns
FROM sys.indexes i
INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
WHERE i.object_id = OBJECT_ID('dbo.PaperTradingSessions')
AND i.type > 0  -- Exclude heap
GROUP BY i.name, i.type_desc, i.is_unique, i.index_id
ORDER BY i.index_id
GO

PRINT ''
PRINT '??? PaperTradingSessions table creation complete! ???'
GO
