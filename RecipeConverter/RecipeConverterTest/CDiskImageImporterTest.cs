using RecipeConverter.src.Classe;

namespace RecipeConverterTest
{
    [TestClass]
    public class CDiskImageImporterTests
    {
        
        private static readonly string m_imageDirectory = Path.Combine(Directory.GetCurrentDirectory(),@"..\..\..\","TestImages");              
        private static readonly string m_notValidImageDirectory = Path.Combine(m_imageDirectory,"NotValid");              
        [TestMethod]
        public void CDiskImageImporter_AllImageTypeFactory_ImportAllValidTypes()
        {
            Console.WriteLine(m_imageDirectory);
            CDiskImageImporter imageImporter = CDiskImageImporter.AllImageTypeFactory();
            var paths = imageImporter.GetImages(m_imageDirectory);
            Assert.AreEqual(paths.Count,19);
        }

        [TestMethod]
        public void CDiskImageImporter_AllImageTypeFactory_DoNotImportValidTypes()
        {
            CDiskImageImporter imageImporter = CDiskImageImporter.AllImageTypeFactory();
            var paths = imageImporter.GetImages(m_notValidImageDirectory);
            Assert.AreEqual(paths.Count,0);
        }

        [TestMethod]
        public void CDiskImageImporter_AllImageTypeFactory_ImportedImagesIsAValidPath()
        {
            CDiskImageImporter imageImporter = CDiskImageImporter.AllImageTypeFactory();
            var paths = imageImporter.GetImages(m_imageDirectory);
            Console.WriteLine(paths.First());
            Assert.IsTrue(File.Exists(paths.First()));

        }
    }
}