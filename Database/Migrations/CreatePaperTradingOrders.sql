-- =====================================================
-- Create PaperTradingOrders Table
-- For the enhanced paper trading system
-- =====================================================

USE [QuantraRelational]
GO

-- Drop existing table if it exists
IF OBJECT_ID('dbo.PaperTradingOrders', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing PaperTradingOrders table...'
    DROP TABLE dbo.PaperTradingOrders
    PRINT '  ? Table dropped'
END
GO

-- Create the PaperTradingOrders table
PRINT 'Creating PaperTradingOrders table...'

CREATE TABLE dbo.PaperTradingOrders (
    Id INT PRIMARY KEY IDENTITY(1,1),
    OrderId NVARCHAR(36) NOT NULL,
    SessionId INT NOT NULL,
    Symbol NVARCHAR(20) NOT NULL,
    OrderType NVARCHAR(20) NOT NULL,
    Side NVARCHAR(10) NOT NULL,
    [State] NVARCHAR(20) NOT NULL,
    Quantity INT NOT NULL,
    FilledQuantity INT NOT NULL DEFAULT 0,
    LimitPrice DECIMAL(18,4) NULL,
    StopPrice DECIMAL(18,4) NULL,
    AvgFillPrice DECIMAL(18,4) NULL,
    TimeInForce NVARCHAR(10) NOT NULL,
    AssetType NVARCHAR(20) NOT NULL,
    RejectReason NVARCHAR(500) NULL,
    Notes NVARCHAR(1000) NULL,
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    SubmittedAt DATETIME2 NULL,
    FilledAt DATETIME2 NULL,
    ExpirationTime DATETIME2 NULL,
    LastUpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    -- Foreign key constraint
    CONSTRAINT FK_PaperTradingOrders_Session 
        FOREIGN KEY (SessionId) 
        REFERENCES dbo.PaperTradingSessions(Id) 
        ON DELETE CASCADE
)
GO

-- Create unique index on OrderId
CREATE UNIQUE NONCLUSTERED INDEX IX_PaperTradingOrders_OrderId 
ON dbo.PaperTradingOrders(OrderId)
GO

-- Create index on SessionId for efficient lookups
CREATE NONCLUSTERED INDEX IX_PaperTradingOrders_SessionId 
ON dbo.PaperTradingOrders(SessionId)
INCLUDE (Symbol, OrderType, Side, [State], CreatedAt)
GO

-- Create index on SessionId and State for filtering active orders
CREATE NONCLUSTERED INDEX IX_PaperTradingOrders_SessionId_State 
ON dbo.PaperTradingOrders(SessionId, [State])
INCLUDE (OrderId, Symbol, OrderType, Side, Quantity, FilledQuantity)
GO

-- Create index on Symbol for symbol-based lookups
CREATE NONCLUSTERED INDEX IX_PaperTradingOrders_Symbol 
ON dbo.PaperTradingOrders(Symbol)
INCLUDE (SessionId, [State], CreatedAt)
GO

-- Create index on CreatedAt for chronological queries
CREATE NONCLUSTERED INDEX IX_PaperTradingOrders_CreatedAt 
ON dbo.PaperTradingOrders(CreatedAt DESC)
INCLUDE (SessionId, OrderId, Symbol, [State])
GO

PRINT '  ? PaperTradingOrders table created successfully!'
PRINT ''

-- Verify the schema
PRINT 'Current PaperTradingOrders schema:'
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
WHERE c.object_id = OBJECT_ID('dbo.PaperTradingOrders')
ORDER BY c.column_id
GO

-- Show indexes
PRINT ''
PRINT 'Indexes on PaperTradingOrders:'
SELECT 
    i.name AS IndexName,
    i.type_desc AS IndexType,
    i.is_unique AS IsUnique,
    STRING_AGG(c.name, ', ') WITHIN GROUP (ORDER BY ic.key_ordinal) AS Columns
FROM sys.indexes i
INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
WHERE i.object_id = OBJECT_ID('dbo.PaperTradingOrders')
AND i.type > 0  -- Exclude heap
GROUP BY i.name, i.type_desc, i.is_unique, i.index_id
ORDER BY i.index_id
GO

-- Show foreign keys
PRINT ''
PRINT 'Foreign Keys on PaperTradingOrders:'
SELECT 
    fk.name AS ForeignKeyName,
    OBJECT_NAME(fk.parent_object_id) AS TableName,
    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ColumnName,
    OBJECT_NAME(fk.referenced_object_id) AS ReferencedTable,
    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS ReferencedColumn
FROM sys.foreign_keys fk
INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
WHERE fk.parent_object_id = OBJECT_ID('dbo.PaperTradingOrders')
GO

PRINT ''
PRINT '??? PaperTradingOrders table creation complete! ???'
GO
