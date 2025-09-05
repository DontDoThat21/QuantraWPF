using System;
using System.Runtime.InteropServices;

namespace Quantra
{
    /// <summary>
    /// Utility class for flashing a window to get user attention
    /// Moved to Helpers namespace for better organization
    /// </summary>
    public static class FlashWindow
    {
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool FlashWindowEx(ref FLASHWINFO pwfi);

        [StructLayout(LayoutKind.Sequential)]
        private struct FLASHWINFO
        {
            public uint cbSize;
            public IntPtr hwnd;
            public uint dwFlags;
            public uint uCount;
            public uint dwTimeout;
        }

        private const uint FLASHW_ALL = 3;
        private const uint FLASHW_TIMERNOFG = 12;

        public static bool Flash(IntPtr hwnd, uint count = 5)
        {
            FLASHWINFO fi = new FLASHWINFO
            {
                cbSize = Convert.ToUInt32(Marshal.SizeOf(typeof(FLASHWINFO))),
                hwnd = hwnd,
                dwFlags = FLASHW_ALL | FLASHW_TIMERNOFG,
                uCount = count,
                dwTimeout = 0
            };
            return FlashWindowEx(ref fi);
        }
    }
}
