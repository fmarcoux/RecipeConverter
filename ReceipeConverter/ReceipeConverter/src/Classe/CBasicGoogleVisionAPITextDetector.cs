using Google.Cloud.Vision.V1;
using ReceipeConverter.src.Interface;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReceipeConverter.src.Classe
{
    internal class CBasicGoogleVisionAPITextDetector : IGoogleVisionAPI
    {
        private readonly ImageAnnotatorClient m_imageAnnotatorClient;
        private readonly ImageContext m_imageContext;
        public CBasicGoogleVisionAPITextDetector() 
        {
            m_imageAnnotatorClient = ImageAnnotatorClient.Create();
            m_imageContext = new()
            {
                TextDetectionParams = new TextDetectionParams(),
            };
            m_imageContext.TextDetectionParams.EnableTextDetectionConfidenceScore = true;
        }

        public Task<IReadOnlyList<EntityAnnotation>> DetectImage(string path_)
        {
            return m_imageAnnotatorClient.DetectTextAsync(Image.FromFile(path_));
        }

        public IReadOnlyList<Task<IReadOnlyList<EntityAnnotation>>> DetectImages(IReadOnlyList<string> paths_)
        {
            List<Task<IReadOnlyList<EntityAnnotation>>> annotations = new();
            foreach (var path in paths_)
            {
                annotations.Add( DetectImage(path));
            }
            return annotations;
        }
    }
}
