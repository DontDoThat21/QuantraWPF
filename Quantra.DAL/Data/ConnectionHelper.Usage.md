# ConnectionHelper Usage Guide

The `ConnectionHelper` class provides a centralized way to manage SQL Server LocalDB connections for the QuantraRelational database.

## Prerequisites

- SQL Server LocalDB installed (part of Visual Studio or SQL Server Express)
- Microsoft.Data.SqlClient NuGet package (version 5.2.2 or later)

## Basic Usage

### Get a Connection (Not Opened)

```csharp
using Quantra.DAL.Data;

// Get a connection instance (not opened)
var connection = ConnectionHelper.GetConnection();
try
{
    connection.Open();
  // Use the connection...
}
finally
{
 connection.Dispose();
}
```

### Get an Open Connection

```csharp
using Quantra.DAL.Data;

// Get an already-opened connection
using (var connection = ConnectionHelper.GetOpenConnection())
{
    // Use the connection directly
    using (var command = connection.CreateCommand())
    {
        command.CommandText = "SELECT * FROM YourTable";
        using (var reader = command.ExecuteReader())
        {
  while (reader.Read())
         {
     // Process results...
      }
        }
    }
}
```

### Test Connection

```csharp
using Quantra.DAL.Data;

if (ConnectionHelper.TestConnection())
{
    Console.WriteLine("Database connection successful!");
}
else
{
    Console.WriteLine("Database connection failed.");
}
```

### Ensure Database Exists

```csharp
using Quantra.DAL.Data;

// Creates the QuantraRelational database if it doesn't exist
bool success = ConnectionHelper.EnsureDatabaseExists();
if (success)
{
    Console.WriteLine("Database is ready.");
}
```

### Get Database Information

```csharp
using Quantra.DAL.Data;

string dbName = ConnectionHelper.GetDatabaseName();
string serverName = ConnectionHelper.GetServerName();

Console.WriteLine($"Database: {dbName}");
Console.WriteLine($"Server: {serverName}");
```

### Custom Connection String

```csharp
using Quantra.DAL.Data;

// Override the default connection string
ConnectionHelper.ConnectionString = 
    "Data Source=(localdb)\\MSSQLLocalDB;" +
    "Initial Catalog=MyCustomDatabase;" +
    "Integrated Security=True;";

using (var connection = ConnectionHelper.GetOpenConnection())
{
    // Now connects to MyCustomDatabase instead
}
```

## Default Connection String

The default connection string is:

```
Data Source=(localdb)\MSSQLLocalDB;
Initial Catalog=QuantraRelational;
Integrated Security=True;
Persist Security Info=False;
Pooling=False;
MultipleActiveResultSets=False;
Encrypt=True;
TrustServerCertificate=False;
Application Name="Quantra Trading Platform";
Command Timeout=0
```

## Using with Dapper

```csharp
using Quantra.DAL.Data;
using Dapper;

using (var connection = ConnectionHelper.GetOpenConnection())
{
    var results = connection.Query<MyModel>("SELECT * FROM MyTable WHERE Id = @Id", new { Id = 1 });
    // Process results...
}
```

## Error Handling

All methods log errors using `LoggingService`. Methods that open connections will throw `InvalidOperationException` if the connection cannot be established.

```csharp
using Quantra.DAL.Data;

try
{
    using (var connection = ConnectionHelper.GetOpenConnection())
    {
        // Use connection...
    }
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"Failed to connect: {ex.Message}");
    // Check inner exception for SQL Server specific errors
}
```

## Integration with Entity Framework Core

While this helper is for ADO.NET connections, you can use it alongside Entity Framework:

```csharp
// For raw SQL queries alongside EF Core
using (var connection = ConnectionHelper.GetOpenConnection())
{
    // Execute raw SQL that EF Core doesn't support well
    var command = connection.CreateCommand();
 command.CommandText = "EXEC sp_SomeStoredProcedure @Param1";
    // ...
}

// Continue using EF Core for normal operations
using (var dbContext = new YourDbContext())
{
    // EF Core operations...
}
```

## Best Practices

1. **Always use `using` statements** to ensure connections are properly disposed
2. **Test the connection** on application startup using `TestConnection()`
3. **Create the database** on first run using `EnsureDatabaseExists()`
4. **Use connection pooling** in production (set `Pooling=True` in connection string)
5. **Log all database operations** - the helper automatically logs to `LoggingService`
