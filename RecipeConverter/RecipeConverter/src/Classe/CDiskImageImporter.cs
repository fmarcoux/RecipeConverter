using Newtonsoft.Json.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace RecipeConverter.src.Classe
{
    public class CDiskImageImporter : IImageImporter
    {
        private readonly IReadOnlyCollection<string> m_allowedExtension;

        static public CDiskImageImporter AllImageTypeFactory() {
            return new CDiskImageImporter(new List<string>
            {
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".webp",
                ".tiff",
                ".psd",
                ".raw",
                ".bmp",
                ".heif",
                ".indd",
                ".svg",
                ".ai",
                ".eps",
                ".pdf",
            });
        }

        private CDiskImageImporter(IReadOnlyCollection<string> allowedExtension_) => m_allowedExtension = allowedExtension_;

        public IReadOnlyCollection<String> GetImages(String directory) 
        {
            List<String> imagesPath = new List<String>();
            foreach (var file in Directory.GetFiles(directory))
            {
                var a = m_allowedExtension.FirstOrDefault(extension => file.Contains(extension, StringComparison.OrdinalIgnoreCase));
                if(a != default)
                {
                    imagesPath.Add(file);
                }
            }
            return imagesPath;
        }
    }
}
