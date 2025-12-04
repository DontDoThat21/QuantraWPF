# Quick Reference: Avoiding DbContext Threading Issues

## The Golden Rules

### ? DO:
1. **Use `AsNoTracking()` for read-only queries**
   ```csharp
   var users = await _dbContext.Users.AsNoTracking().ToListAsync();
   ```

2. **Make all database operations sequential (one at a time)**
   ```csharp
   var users = await _dbContext.Users.ToListAsync();
   var orders = await _dbContext.Orders.ToListAsync();
   ```

3. **Register DbContext as Scoped**
   ```csharp
   services.AddDbContext<MyDbContext>(options => /*...*/);  // Scoped by default
   ```

4. **Use async/await consistently**
   ```csharp
   public async Task<User> GetUserAsync(int id)
   {
       return await _dbContext.Users.FindAsync(id);
   }
   ```

### ? DON'T:
1. **Never run concurrent queries on the same DbContext**
   ```csharp
   // ? BAD
   var task1 = _dbContext.Users.ToListAsync();
   var task2 = _dbContext.Orders.ToListAsync();
   await Task.WhenAll(task1, task2);
   ```

2. **Don't use AsNoTracking() when you need to update/delete**
   ```csharp
   // ? BAD
   var user = await _dbContext.Users.AsNoTracking().FirstAsync();
   user.Name = "New Name";
   await _dbContext.SaveChangesAsync();  // Won't work!
   ```

3. **Don't store DbContext in singletons**
   ```csharp
   // ? BAD
   services.AddSingleton<MyService>(); // If MyService has DbContext
   ```

4. **Don't mix sync and async**
   ```csharp
   // ? BAD
   var user = _dbContext.Users.ToList();  // Sync
   var orders = await _dbContext.Orders.ToListAsync();  // Async
   ```

---

## Common Scenarios

### Reading Data (No Updates)
```csharp
// ? CORRECT
public async Task<List<UserDto>> GetUsersAsync()
{
    return await _dbContext.Users
        .AsNoTracking()  // ? Add this for read-only
        .Select(u => new UserDto { Name = u.Name })
        .ToListAsync();
}
```

### Updating Data
```csharp
// ? CORRECT
public async Task UpdateUserAsync(int id, string newName)
{
    var user = await _dbContext.Users.FindAsync(id);  // Tracking enabled
    user.Name = newName;
    await _dbContext.SaveChangesAsync();
}
```

### Multiple Sequential Operations
```csharp
// ? CORRECT
public async Task ProcessOrderAsync(int orderId)
{
    var order = await _dbContext.Orders.FindAsync(orderId);
    var user = await _dbContext.Users.FindAsync(order.UserId);
    var products = await _dbContext.Products
        .Where(p => order.ProductIds.Contains(p.Id))
        .ToListAsync();
    
    // Process...
}
```

### Parallel Operations (Use Separate Contexts)
```csharp
// ? CORRECT
public class MyService
{
    private readonly IServiceScopeFactory _scopeFactory;
    
    public async Task ParallelOperationsAsync()
    {
        var task1 = Task.Run(async () =>
        {
            using var scope = _scopeFactory.CreateScope();
            var context = scope.ServiceProvider.GetRequiredService<MyDbContext>();
            return await context.Users.ToListAsync();
        });
        
        var task2 = Task.Run(async () =>
        {
            using var scope = _scopeFactory.CreateScope();
            var context = scope.ServiceProvider.GetRequiredService<MyDbContext>();
            return await context.Orders.ToListAsync();
        });
        
        await Task.WhenAll(task1, task2);
    }
}
```

---

## Exception Messages to Watch For

```
? "A second operation was started on this context instance before a previous operation completed."
   ? You have concurrent operations on the same DbContext

? "The instance of entity type '...' cannot be tracked because another instance with the same key value..."
   ? You're tracking the same entity twice (use AsNoTracking or DetachLocal)

? "An error occurred while updating the entries. See the inner exception for details."
   ? Often happens after AsNoTracking when trying to save changes
```

---

## Quick Checklist for Code Review

- [ ] All read-only queries use `.AsNoTracking()`
- [ ] All async methods use `async`/`await` (no `.Result` or `.Wait()`)
- [ ] No `Task.WhenAll()` or parallel operations on same context
- [ ] DbContext is injected as scoped (not singleton)
- [ ] No fire-and-forget operations that access DbContext concurrently
- [ ] Update/Delete operations do NOT use `AsNoTracking()`

---

## Performance Tips

1. **Project to DTOs early**
   ```csharp
   // Better performance
   var users = await _dbContext.Users
       .AsNoTracking()
       .Select(u => new UserDto { Id = u.Id, Name = u.Name })
       .ToListAsync();
   ```

2. **Use compiled queries for hot paths**
   ```csharp
   private static readonly Func<MyDbContext, int, Task<User>> GetUserById =
       EF.CompileAsyncQuery((MyDbContext ctx, int id) => 
           ctx.Users.FirstOrDefault(u => u.Id == id));
   ```

3. **Batch operations when possible**
   ```csharp
   // Better than multiple SaveChanges calls
   foreach (var user in users)
   {
       user.IsActive = true;
   }
   await _dbContext.SaveChangesAsync();  // Single save
   ```

---

## Need Help?

- See `DBCONTEXT_MULTITHREADING_FIX.md` for detailed explanation
- Check [EF Core docs](https://docs.microsoft.com/ef/core/querying/tracking)
- Ask in team chat with error message and code snippet

---

**Remember:** When in doubt, use `AsNoTracking()` for reads and keep operations sequential!
