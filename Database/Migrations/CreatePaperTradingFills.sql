-- =====================================================
-- Create PaperTradingFills Table
-- For the enhanced paper trading system
-- =====================================================

USE [QuantraRelational]
GO

-- Drop existing table if it exists
IF OBJECT_ID('dbo.PaperTradingFills', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing PaperTradingFills table...'
    DROP TABLE dbo.PaperTradingFills
    PRINT '  ? Table dropped'
END
GO

-- Create the PaperTradingFills table
PRINT 'Creating PaperTradingFills table...'

CREATE TABLE dbo.PaperTradingFills (
    Id INT PRIMARY KEY IDENTITY(1,1),
    FillId NVARCHAR(36) NOT NULL,
    OrderEntityId INT NOT NULL,
    PositionEntityId INT NULL,
    Symbol NVARCHAR(20) NOT NULL,
    Quantity INT NOT NULL,
    Price DECIMAL(18,4) NOT NULL,
    Side NVARCHAR(10) NOT NULL,
    Commission DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    Slippage DECIMAL(18,4) NOT NULL DEFAULT 0.0000,
    Exchange NVARCHAR(50) NULL,
    FillTime DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    -- Foreign key constraints
    CONSTRAINT FK_PaperTradingFills_Order 
        FOREIGN KEY (OrderEntityId) 
        REFERENCES dbo.PaperTradingOrders(Id) 
        ON DELETE CASCADE,
    
    CONSTRAINT FK_PaperTradingFills_Position 
        FOREIGN KEY (PositionEntityId) 
        REFERENCES dbo.PaperTradingPositions(Id) 
        ON DELETE NO ACTION
)
GO

-- Create unique index on FillId
CREATE UNIQUE NONCLUSTERED INDEX IX_PaperTradingFills_FillId 
ON dbo.PaperTradingFills(FillId)
GO

-- Create index on OrderEntityId for efficient lookups
CREATE NONCLUSTERED INDEX IX_PaperTradingFills_OrderEntityId 
ON dbo.PaperTradingFills(OrderEntityId)
INCLUDE (FillId, Symbol, Quantity, Price, FillTime)
GO

-- Create index on PositionEntityId for efficient lookups
CREATE NONCLUSTERED INDEX IX_PaperTradingFills_PositionEntityId 
ON dbo.PaperTradingFills(PositionEntityId)
INCLUDE (Symbol, Quantity, Price, FillTime)
WHERE PositionEntityId IS NOT NULL
GO

-- Create index on Symbol for symbol-based lookups
CREATE NONCLUSTERED INDEX IX_PaperTradingFills_Symbol 
ON dbo.PaperTradingFills(Symbol)
INCLUDE (OrderEntityId, Quantity, Price, FillTime)
GO

-- Create index on FillTime for chronological queries
CREATE NONCLUSTERED INDEX IX_PaperTradingFills_FillTime 
ON dbo.PaperTradingFills(FillTime DESC)
INCLUDE (FillId, OrderEntityId, Symbol, Quantity, Price)
GO

PRINT '  ? PaperTradingFills table created successfully!'
PRINT ''

-- Verify the schema
PRINT 'Current PaperTradingFills schema:'
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
WHERE c.object_id = OBJECT_ID('dbo.PaperTradingFills')
ORDER BY c.column_id
GO

-- Show indexes
PRINT ''
PRINT 'Indexes on PaperTradingFills:'
SELECT 
    i.name AS IndexName,
    i.type_desc AS IndexType,
    i.is_unique AS IsUnique,
    STRING_AGG(c.name, ', ') WITHIN GROUP (ORDER BY ic.key_ordinal) AS Columns
FROM sys.indexes i
INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
WHERE i.object_id = OBJECT_ID('dbo.PaperTradingFills')
AND i.type > 0  -- Exclude heap
GROUP BY i.name, i.type_desc, i.is_unique, i.index_id
ORDER BY i.index_id
GO

-- Show foreign keys
PRINT ''
PRINT 'Foreign Keys on PaperTradingFills:'
SELECT 
    fk.name AS ForeignKeyName,
    OBJECT_NAME(fk.parent_object_id) AS TableName,
    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ColumnName,
    OBJECT_NAME(fk.referenced_object_id) AS ReferencedTable,
    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS ReferencedColumn
FROM sys.foreign_keys fk
INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
WHERE fk.parent_object_id = OBJECT_ID('dbo.PaperTradingFills')
GO

PRINT ''
PRINT '??? PaperTradingFills table creation complete! ???'
GO
