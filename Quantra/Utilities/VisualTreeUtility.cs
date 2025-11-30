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
                current = GetParent(current);
            }
            return null;
        }

        /// <summary>
        /// Gets the parent of a DependencyObject, handling both visual and logical tree elements
        /// </summary>
        /// <param name="element">The element to get the parent of</param>
        /// <returns>The parent element if found; otherwise, null</returns>
        private static DependencyObject GetParent(DependencyObject element)
        {
            if (element is Visual || element is System.Windows.Media.Media3D.Visual3D)
            {
                return VisualTreeHelper.GetParent(element);
            }
            else if (element is FrameworkContentElement contentElement)
            {
                return contentElement.Parent;
            }
            return null;
        }
    }
}
