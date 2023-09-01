using Google.Cloud.Vision.V1;
using ReceipeConverter.src.Classe;
using System;
using System.Security.Cryptography;

namespace VisionApiDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            CBasicGoogleVisionAPITextDetector client = new CBasicGoogleVisionAPITextDetector();
            CDiskImageImporter importer = CDiskImageImporter.AllImageTypeFactory();           
            var response = client.DetectImage("C:\\TestFilesForVisionAPI\\recipe1.jpeg");
            foreach (var annotation in response.Result)
            {
                if (annotation.Description != null)
                {
                    Console.WriteLine(annotation.Description);
                }
            }
        }
    }
}
