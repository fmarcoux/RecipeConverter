using Newtonsoft.Json;
using System;
using System.Text.Json;
using ReceipeConverter.src.DTO;
using ReceipeConverter.src.DTO.TandoorRecipe;

namespace RecipeConverterTest
{
    [TestClass]
    public class TandoorDTOTest
    {

        private static readonly string m_jsonDirectory = Path.Combine(Directory.GetCurrentDirectory(), @"..\..\..\", "TestJSON");
        private static readonly string m_newReceipeJson = Path.Combine(m_jsonDirectory, "NewRecipe");

        [TestMethod]
        public void TandoorDTOTest_FromMinimalJsonCreateDTO()
        {
            Recipe? recipe = JsonConvert.DeserializeObject<Recipe>(System.IO.File.ReadAllText(Path.Combine(m_newReceipeJson, "minimal.json")));
            Assert.IsNotNull(recipe);
            Assert.AreEqual(recipe?.Name, "CHANGE THE NAME BRO");

        }
    }
}
