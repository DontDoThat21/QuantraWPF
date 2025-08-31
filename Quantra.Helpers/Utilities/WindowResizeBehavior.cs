using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;

namespace Quantra
{
    /// <summary>
    /// Helper class for adding custom window resize behavior to borderless WPF windows
    /// </summary>
    public static class WindowResizeBehavior
    {
        // WM_NCHITTEST values for window resizing from WinUser.h
        private const int HTLEFT = 10;
        private const int HTRIGHT = 11;
        private const int HTTOP = 12;
        private const int HTTOPLEFT = 13;
        private const int HTTOPRIGHT = 14;
        private const int HTBOTTOM = 15;
        private const int HTBOTTOMLEFT = 16;
        private const int HTBOTTOMRIGHT = 17;
        private const int HTCAPTION = 2;

        [DllImport("user32.dll")]
        private static extern IntPtr DefWindowProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam);

        [StructLayout(LayoutKind.Sequential)]
        private struct POINT
        {
            public int x;
            public int y;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct RECT
        {
            public int left;
            public int top;
            public int right;
            public int bottom;
        }

        /// <summary>
        /// Attaches resize behavior to a window
        /// </summary>
        /// <param name="window">The window to make resizable</param>
        /// <param name="resizeBorderThickness">Thickness of the resize border area in pixels</param>
        public static void AttachResizeBehavior(Window window, int resizeBorderThickness = 8)
        {
            if (window == null) return;

            // Ensure minimal size is set
            if (window.MinWidth <= 0) window.MinWidth = 300;
            if (window.MinHeight <= 0) window.MinHeight = 200;

            // Use SourceInitialized event to ensure the window handle is created
            window.SourceInitialized += (sender, e) =>
            {
                var hwndSource = PresentationSource.FromVisual(window) as HwndSource;
                if (hwndSource != null)
                {
                    hwndSource.AddHook(new HwndSourceHook(WndProc));
                }
            };

            // Helper function for window procedure hook
            IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled)
            {
                // Handle WM_NCHITTEST message to enable resizing
                if (msg == 0x0084) // WM_NCHITTEST
                {
                    // Get the point coordinates for the hit test
                    int screenX = (short)((int)lParam & 0xFFFF);
                    int screenY = (short)((int)lParam >> 16);

                    // Get the window rectangle
                    RECT windowRect = new RECT();
                    GetWindowRect(hwnd, ref windowRect);

                    // Calculate window edges for hit-testing
                    int leftEdge = windowRect.left + resizeBorderThickness;
                    int rightEdge = windowRect.right - resizeBorderThickness;
                    int topEdge = windowRect.top + resizeBorderThickness;
                    int bottomEdge = windowRect.bottom - resizeBorderThickness;

                    // Determine if the hit test is for resizing (on the border edge)

                    // Left edge
                    if (screenX < leftEdge)
                    {
                        if (screenY < topEdge)
                        {
                            handled = true;
                            return new IntPtr(HTTOPLEFT);
                        }
                        if (screenY >= bottomEdge)
                        {
                            handled = true;
                            return new IntPtr(HTBOTTOMLEFT);
                        }
                        handled = true;
                        return new IntPtr(HTLEFT);
                    }

                    // Right edge
                    if (screenX >= rightEdge)
                    {
                        if (screenY < topEdge)
                        {
                            handled = true;
                            return new IntPtr(HTTOPRIGHT);
                        }
                        if (screenY >= bottomEdge)
                        {
                            handled = true;
                            return new IntPtr(HTBOTTOMRIGHT);
                        }
                        handled = true;
                        return new IntPtr(HTRIGHT);
                    }

                    // Top edge
                    if (screenY < topEdge)
                    {
                        handled = true;
                        return new IntPtr(HTTOP);
                    }

                    // Bottom edge
                    if (screenY >= bottomEdge)
                    {
                        handled = true;
                        return new IntPtr(HTBOTTOM);
                    }
                }

                return IntPtr.Zero;
            }
        }

        [DllImport("user32.dll")]
        private static extern bool GetWindowRect(IntPtr hwnd, ref RECT rect);
    }
}
