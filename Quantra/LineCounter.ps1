Get-ChildItem -Recurse -File -Include *.cs,*.xaml | Where-Object { -not ($_.Name -like '*.g.cs') } | ForEach-Object {
    $lineCount = (Get-Content $_.FullName -ErrorAction SilentlyContinue).Count
    [PSCustomObject]@{
        File = $_.FullName
        Lines = $lineCount
    }
} | Sort-Object Lines -Descending | Format-Table -AutoSize
