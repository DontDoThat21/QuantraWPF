using System;

namespace Quantra.Models
{
    /// <summary>
    /// Model class to represent the saved form state including position and screen information
    /// </summary>
    public class WindowState
    {
        public int Id { get; set; }
        public string FormName { get; set; }
        public double Left { get; set; }
        public double Top { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }
        public int FormState { get; set; }  // 0:Normal, 1:Minimized, 2:Maximized
        public string ScreenDeviceName { get; set; }
        public double ScreenWidth { get; set; }
        public double ScreenHeight { get; set; }
        public DateTime LastUpdated { get; set; }
    }
}
