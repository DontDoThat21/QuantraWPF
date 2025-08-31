using System.Windows;
using System.Windows.Media;

namespace Quantra.Utilities
{
    public static class VisualTreeHelperExtensions
    {
        public static T GetParentOfType<T>(DependencyObject child) where T : DependencyObject
        {
            DependencyObject parentObject = VisualTreeHelper.GetParent(child);

            if (parentObject == null)
            {
                return null;
            }

            if (parentObject is T parent)
            {
                return parent;
            }
            else
            {
                return GetParentOfType<T>(parentObject);
            }
        }
    }
}
