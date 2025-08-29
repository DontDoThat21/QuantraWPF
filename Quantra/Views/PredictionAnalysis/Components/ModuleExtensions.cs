using System.Windows;
using Quantra.Views.PredictionAnalysis.Components;

namespace Quantra.Views.PredictionAnalysis
{
    public static class ModuleExtensions
    {
        public static T WithMargin<T>(this T module, Thickness margin) where T : PredictionModuleBase
        {
            module.Margin = margin;
            return module;
        }
    }
}