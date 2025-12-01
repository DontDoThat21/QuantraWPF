using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;

namespace Quantra.Repositories
{
    public class TabRepository
    {
        private readonly IDbConnection _connection;

        public TabRepository(IDbConnection connection)
        {
            _connection = connection;
        }

        /*public List<(string TabName, int TabOrder)> GetTabs()
        {
            _connection.Open();
            var tabs = _connection.Query<(string TabName, int TabOrder)>(
                "SELECT TabName, TabOrder FROM UserAppSettings ORDER BY TabOrder").ToList();
            return tabs;
        }

        public void InsertTab(string tabName, int tabOrder, int rows, int columns)
        {
            var query = @"
                INSERT INTO UserAppSettings 
                    (TabName, TabOrder, GridRows, GridColumns) 
                VALUES 
                    (@TabName, @TabOrder, @GridRows, @GridColumns)";
            _connection.Execute(query, new { TabName = tabName, TabOrder = tabOrder, GridRows = rows, GridColumns = columns });
        }

        public void UpdateTabName(string oldTabName, string newTabName)
        {
            var query = "UPDATE UserAppSettings SET TabName = @NewTabName WHERE TabName = @OldTabName";
            _connection.Execute(query, new { NewTabName = newTabName, OldTabName = oldTabName });
        }

        public void DeleteTab(string tabName)
        {
            var query = "DELETE FROM UserAppSettings WHERE TabName = @TabName";
            _connection.Execute(query, new { TabName = tabName });
        }

        public void UpdateTabOrder(string tabName, int tabOrder)
        {
            var query = "UPDATE UserAppSettings SET TabOrder = @TabOrder WHERE TabName = @TabName";
            _connection.Execute(query, new { TabOrder = tabOrder, TabName = tabName });
        }

        public void UpdateControlPosition(string tabName, int controlIndex, int row, int column, int rowSpan, int columnSpan)
        {
            var query = @"
                UPDATE ControlPositions 
                SET Row = @Row, Column = @Column, RowSpan = @RowSpan, ColumnSpan = @ColumnSpan 
                WHERE TabName = @TabName AND ControlIndex = @ControlIndex";
            _connection.Execute(query, new { TabName = tabName, ControlIndex = controlIndex, Row = row, Column = column, RowSpan = rowSpan, ColumnSpan = columnSpan });
        }*/
    }
}
