# Insider Transactions "Load All Symbols" Feature

## Overview
Implemented functionality to load insider transactions for multiple symbols and cache them in a database table.

## Features Implemented

### 1. Database Table Creation
- **File**: `Quantra.DAL/SQL/CreateInsiderTransactionsTable.sql`
- **Table Name**: `InsiderTransactions`
- **Columns**:
  - `Id` - Primary key
  - `Symbol` - Stock ticker
  - `FilingDate` - Date transaction was filed
  - `TransactionDate` - Date transaction occurred
  - `OwnerName` - Name of insider
  - `OwnerCik` - CIK number
  - `OwnerTitle` - Position/title
  - `SecurityType` - Type of security
  - `TransactionCode` - Transaction type code (P, S, M, etc.)
  - `SharesTraded` - Number of shares
  - `PricePerShare` - Price per share
  - `SharesOwnedFollowing` - Shares owned after transaction
  - `AcquisitionOrDisposal` - A or D indicator
  - `LastUpdated` - Timestamp of last update
  
- **Indexes**:
  - Symbol index for fast lookups
  - TransactionDate index for date-range queries
  - LastUpdated index for cache expiry checks
  
- **Unique Constraint**: Prevents duplicate transactions (Symbol, FilingDate, TransactionDate, OwnerCik)

### 2. UI Changes
- **File**: `Quantra\Views\Intelligence\InsiderTransactionsControl.xaml`
- Added "Load All Symbols" checkbox
- Symbol textbox now disables when "Load All Symbols" is checked
- Added Symbol column to DataGrid (useful when showing transactions from multiple symbols)

### 3. Code-Behind Changes
- **File**: `Quantra\Views\Intelligence\InsiderTransactionsControl.xaml.cs`

#### New Methods:

**Database Caching**:
- `InitializeInsiderTransactionsTable()` - Creates table and indexes if they don't exist
- `SaveTransactionsToCache(List<InsiderTransactionData>)` - Saves transactions to database
- `LoadTransactionsFromCache(string symbol = null)` - Loads cached transactions (all or by symbol)
- `GetCachedSymbols()` - Gets list of all symbols in cache

**Load All Symbols**:
- `LoadAllSymbolsInsiderTransactions()` - Main method to fetch transactions for multiple symbols
- `GetSymbolsToFetch()` - Returns list of symbols to fetch (from user settings or default S&P 500 list)
- `LoadAllSymbolsCheckBox_Changed()` - Event handler to enable/disable symbol textbox

#### Modified Methods:
- `LoadButton_Click()` - Now checks if "Load All Symbols" is checked and calls appropriate method
- Constructor - Initializes database connection and creates table

## How It Works

### Single Symbol Mode (Default)
1. User enters a symbol in the textbox
2. Clicks "Load" button
3. Fetches data from Alpha Vantage API
4. Displays results in the grid

### Load All Symbols Mode
1. User checks "Load All Symbols" checkbox
2. Symbol textbox is disabled
3. Clicks "Load" button
4. System:
   - First loads all cached transactions from database
   - Displays cached data immediately
   - Fetches fresh data for all configured symbols (default: top 50 S&P 500 stocks)
   - Shows progress: "Loading AAPL... (1/50)"
   - Saves new transactions to database cache
   - Merges and displays all transactions
5. Shows summary: "Loaded X total transactions (Y symbols successful, Z errors)"

### Caching Benefits
- **Reduced API calls**: Data is cached in database, avoiding repeated API requests
- **Faster loading**: Cached data displayed immediately while fresh data is fetched in background
- **Deduplication**: UNIQUE constraint prevents storing duplicate transactions
- **Date filtering**: Indexes optimize date-range queries

## SQL Table Creation Script

To manually create the table, run this SQL against your SQLite database:

```sql
CREATE TABLE IF NOT EXISTS InsiderTransactions (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Symbol TEXT NOT NULL,
    FilingDate TEXT NOT NULL,
    TransactionDate TEXT NOT NULL,
    OwnerName TEXT,
    OwnerCik TEXT,
    OwnerTitle TEXT,
    SecurityType TEXT,
    TransactionCode TEXT,
    SharesTraded INTEGER NOT NULL DEFAULT 0,
    PricePerShare REAL NOT NULL DEFAULT 0.0,
    SharesOwnedFollowing INTEGER NOT NULL DEFAULT 0,
    AcquisitionOrDisposal TEXT,
    LastUpdated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(Symbol, FilingDate, TransactionDate, OwnerCik)
);

CREATE INDEX IF NOT EXISTS idx_insider_transactions_symbol ON InsiderTransactions(Symbol);
CREATE INDEX IF NOT EXISTS idx_insider_transactions_date ON InsiderTransactions(TransactionDate);
CREATE INDEX IF NOT EXISTS idx_insider_transactions_lastupdated ON InsiderTransactions(LastUpdated);
```

**Note**: The application automatically creates this table on first use of the Insider Transactions control.

## Configuration

### Custom Symbol List
To specify which symbols to load, add a user preference in the database:

```sql
INSERT INTO UserPreferences (UserId, PreferenceKey, PreferenceValue)
VALUES (1, 'InsiderTransactionsSymbols', 'AAPL,MSFT,GOOGL,AMZN,META');
```

### Default Symbols
If no custom list is configured, the system defaults to these 50 S&P 500 stocks:
- AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK.B, UNH, XOM
- JNJ, V, PG, MA, HD, CVX, MRK, ABBV, PEP, COST
- AVGO, KO, LLY, TMO, ADBE, MCD, ACN, CSCO, ABT, WMT
- DHR, NKE, NEE, TXN, DIS, INTC, VZ, PM, CRM, CMCSA
- NFLX, UNP, AMD, QCOM, UPS, ORCL, HON, BMY, RTX, BA

## Rate Limiting
- The system includes an 800ms delay between API calls to avoid rate limiting
- Shows real-time progress for each symbol being loaded
- Continues loading remaining symbols even if some fail

## Error Handling
- Failed symbol loads are logged but don't stop the process
- Status bar shows count of successful and failed symbol loads
- Logging service records all errors for debugging

## Build Notes
- The SQL file validation errors from Visual Studio can be ignored - they're for SQL Server syntax
- SQLite fully supports `CREATE TABLE IF NOT EXISTS` and `AUTOINCREMENT`
- The code is tested and functional with SQLite databases
