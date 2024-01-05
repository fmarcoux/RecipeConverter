using Google.Cloud.Vision.V1;
using RecipeConverter.src.Interface;

namespace RecipeConverter.src.Classe
{
    public class CBasicGoogleVisionAPITextDetector : IGoogleVisionAPI
    {
        private readonly ImageAnnotatorClient m_imageAnnotatorClient;
        private readonly ImageContext m_imageContext;
        public CBasicGoogleVisionAPITextDetector() 
        {
            m_imageAnnotatorClient = ImageAnnotatorClient.Create();
            m_imageContext = new()
            {
                TextDetectionParams = new TextDetectionParams(),
                LanguageHints = { "fr" },
            };
            m_imageContext.TextDetectionParams.EnableTextDetectionConfidenceScore = true;
        }

        public CBasicGoogleVisionAPITextDetector(ImageAnnotatorClient client)
        {
            m_imageAnnotatorClient = client;
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
