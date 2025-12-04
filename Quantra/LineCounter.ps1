Get-ChildItem -Recurse -File -Include *.py,*.cs,*.xaml,*.xaml.cs,*.resx,*.config | Where-Object { -not ($_.Name -like '*.g.cs') -and -not ($_.Name -like '*.g.i.cs') -and -not ($_.Name -like '*.Designer.cs') } | ForEach-Object {
    $lineCount = (Get-Content $_.FullName -ErrorAction SilentlyContinue).Count
    [PSCustomObject]@{
        File = $_.FullName
        Lines = $lineCount
    }
} | Sort-Object Lines -Descending | Format-Table -AutoSize