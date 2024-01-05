using Google.Apis.Auth.OAuth2;
using Google.Cloud.Vision.V1;
using RecipeConverter.src.Classe;
using Grpc.Auth;

namespace RecipeConverterTest
{
    [TestClass]
    public class GoogleVisionTextDetectorTest
    {
        private static readonly string m_projectDirectory = DirectoryUtils.GetProjectRootDiretory();
        private static readonly string m_imageDirectory = Path.Combine(m_projectDirectory, "TestImages");  
        private static readonly string m_credentialsPath = Path.Combine(Directory.GetParent(m_projectDirectory)!.FullName,"RecipeConverter", "applicationdefaultcreds.json");    

        [TestMethod]
        public void GoogleVisionTextDetector_TestSingleImageDetection()
        {
            CBasicGoogleVisionAPITextDetector client = CreateBasicGoogleVisionApi();
            var response = client.DetectImage(Path.Combine(m_imageDirectory,"gauffresBelges.jpg"));

            foreach (var annotation in response.Result)
            {
                if (annotation.Description != null)
                {
                    Console.WriteLine(annotation.Description);
                }
            }

            Assert.IsNotNull(response.Result);
            Assert.IsTrue(response.Result[0].Description.Length > 100);
            Assert.IsTrue(response.Result.Count > 0);  
        }


        private CBasicGoogleVisionAPITextDetector CreateBasicGoogleVisionApi() {
            
            // Create a ChannelCredentials instance using the JSON key file
            var channelCredentials = GoogleCredential.FromFile(m_credentialsPath).ToChannelCredentials();

            // Create an ImageAnnotatorClient using the channelCredentials
            var clientBuilder = new ImageAnnotatorClientBuilder
            {
                ChannelCredentials = channelCredentials,
                Endpoint = ImageAnnotatorClient.DefaultEndpoint.ToString()
            };

            // Create an ImageAnnotatorClient instance
            var client = clientBuilder.Build();

            return  new CBasicGoogleVisionAPITextDetector(client);
        }
    }
}
