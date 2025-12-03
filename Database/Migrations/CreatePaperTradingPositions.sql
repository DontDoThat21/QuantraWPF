-- =====================================================
-- Create PaperTradingPositions Table
-- For the enhanced paper trading system
-- =====================================================

USE [QuantraRelational]
GO

-- Drop existing table if it exists
IF OBJECT_ID('dbo.PaperTradingPositions', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing PaperTradingPositions table...'
    DROP TABLE dbo.PaperTradingPositions
    PRINT '  ? Table dropped'
END
GO

-- Create the PaperTradingPositions table
PRINT 'Creating PaperTradingPositions table...'

CREATE TABLE dbo.PaperTradingPositions (
    Id INT PRIMARY KEY IDENTITY(1,1),
    SessionId INT NOT NULL,
    Symbol NVARCHAR(20) NOT NULL,
    Quantity INT NOT NULL,
    AverageCost DECIMAL(18,4) NOT NULL,
    CurrentPrice DECIMAL(18,4) NOT NULL,
    UnrealizedPnL DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    RealizedPnL DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    AssetType NVARCHAR(20) NOT NULL,
    IsClosed BIT NOT NULL DEFAULT 0,
    OpenedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    ClosedAt DATETIME2 NULL,
    LastUpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    -- Foreign key constraint
    CONSTRAINT FK_PaperTradingPositions_Session 
        FOREIGN KEY (SessionId) 
        REFERENCES dbo.PaperTradingSessions(Id) 
        ON DELETE CASCADE
)
GO

-- Create index on SessionId for efficient lookups
CREATE NONCLUSTERED INDEX IX_PaperTradingPositions_SessionId 
ON dbo.PaperTradingPositions(SessionId)
INCLUDE (Symbol, Quantity, IsClosed)
GO

-- Create index on SessionId and IsClosed for filtering open positions
CREATE NONCLUSTERED INDEX IX_PaperTradingPositions_SessionId_IsClosed 
ON dbo.PaperTradingPositions(SessionId, IsClosed)
INCLUDE (Symbol, Quantity, AverageCost, CurrentPrice, UnrealizedPnL, RealizedPnL)
GO

-- Create index on Symbol for symbol-based lookups
CREATE NONCLUSTERED INDEX IX_PaperTradingPositions_Symbol 
ON dbo.PaperTradingPositions(Symbol)
INCLUDE (SessionId, Quantity, IsClosed)
GO

-- Create composite index for unique open positions per session and symbol
CREATE UNIQUE NONCLUSTERED INDEX IX_PaperTradingPositions_Session_Symbol_Open 
ON dbo.PaperTradingPositions(SessionId, Symbol) 
WHERE IsClosed = 0
GO

PRINT '  ? PaperTradingPositions table created successfully!'
PRINT ''

-- Verify the schema
PRINT 'Current PaperTradingPositions schema:'
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
WHERE c.object_id = OBJECT_ID('dbo.PaperTradingPositions')
ORDER BY c.column_id
GO

-- Show indexes
PRINT ''
PRINT 'Indexes on PaperTradingPositions:'
SELECT 
    i.name AS IndexName,
    i.type_desc AS IndexType,
    i.is_unique AS IsUnique,
    STRING_AGG(c.name, ', ') WITHIN GROUP (ORDER BY ic.key_ordinal) AS Columns
FROM sys.indexes i
INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
WHERE i.object_id = OBJECT_ID('dbo.PaperTradingPositions')
AND i.type > 0  -- Exclude heap
GROUP BY i.name, i.type_desc, i.is_unique, i.index_id
ORDER BY i.index_id
GO

-- Show foreign keys
PRINT ''
PRINT 'Foreign Keys on PaperTradingPositions:'
SELECT 
    fk.name AS ForeignKeyName,
    OBJECT_NAME(fk.parent_object_id) AS TableName,
    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ColumnName,
    OBJECT_NAME(fk.referenced_object_id) AS ReferencedTable,
    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS ReferencedColumn
FROM sys.foreign_keys fk
INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
WHERE fk.parent_object_id = OBJECT_ID('dbo.PaperTradingPositions')
GO

PRINT ''
PRINT '??? PaperTradingPositions table creation complete! ???'
GO
