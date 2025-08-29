using System.Windows;
using System.Windows.Media;

namespace Quantra.Utilities
{
    /// <summary>
    /// Provides utility methods for working with the WPF visual tree
    /// </summary>
    public static class VisualTreeUtility
    {
        /// <summary>
        /// Finds an ancestor of a specified type in the visual tree
        /// </summary>
        /// <typeparam name="T">The type of ancestor to find</typeparam>
        /// <param name="current">The element to start searching from</param>
        /// <returns>The ancestor of type T if found; otherwise, null</returns>
        public static T FindAncestor<T>(DependencyObject current) where T : DependencyObject
        {
            while (current != null)
            {
                if (current is T)
                {
                    return (T)current;
                }
                current = VisualTreeHelper.GetParent(current);
            }
            return null;
        }
    }
}
