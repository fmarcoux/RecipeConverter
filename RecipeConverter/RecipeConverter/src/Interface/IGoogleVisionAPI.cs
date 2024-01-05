using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Google.Cloud.Vision.V1;

namespace RecipeConverter.src.Interface
{
    internal interface IGoogleVisionAPI
    {
        Task<IReadOnlyList<EntityAnnotation>> DetectImage(string path_);
        IReadOnlyList<Task<IReadOnlyList<EntityAnnotation>>> DetectImages(IReadOnlyList<string> paths_);
    }
}
