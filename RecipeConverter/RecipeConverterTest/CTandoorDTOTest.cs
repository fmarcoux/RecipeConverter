
using System;
using System.Text.Json;
using System.Text.RegularExpressions;
using RecipeConverter.src.DTO;
using RecipeConverter.src.DTO.TandoorRecipe;

namespace RecipeConverterTest
{
    [TestClass]
    public class CTandoorDTOTest
    {

        private static readonly string m_jsonDirectory = Path.Combine(Directory.GetCurrentDirectory(), @"..\..\..\", "TestJSON");
        private static readonly string m_newReceipeJson = Path.Combine(m_jsonDirectory, "NewRecipe");

        [TestMethod]
        public void TandoorDTOTest_FromMinimalJsonCreateDTO()
        {
            Recipe? recipe = JsonSerializer.Deserialize<Recipe>(System.IO.File.ReadAllText(Path.Combine(m_newReceipeJson, "minimal.json")));
            Assert.IsNotNull(recipe);
            Assert.AreEqual(recipe.Name, "Ceci est un nom de fucking test yo2");
            Assert.IsInstanceOfType( recipe.Name, typeof(string));
            Assert.AreEqual(recipe.Steps.Count, 1);
            Assert.AreEqual(recipe.Steps.First().Name, "");
            Assert.IsTrue(recipe.Steps.First().Instruction.Length>100);
            Assert.AreEqual(recipe.Steps.First().Ingredients.Count, 2);
            Assert.AreEqual(recipe.Steps.First().Ingredients.First().Food.Name, "YII pelee et rapee");
            Assert.AreEqual(recipe.Steps.First().Ingredients.First().Food.Description, "");
            Assert.IsFalse(recipe.Steps.First().Ingredients.First().Food.IgnoreShopping );
            Assert.AreEqual(recipe.Steps.First().Ingredients.First().Unit.Name,"g");
            Assert.AreEqual(recipe.Steps.First().Ingredients.First().Amount, 250.0d);
            Assert.AreEqual(recipe.Steps.First().Ingredients.First().Note,"44 tasses");
            Assert.AreEqual(recipe.Steps.First().Ingredients.First().OriginalText, "250 g (2 tasses) de patate douce pelee et rapee");
        }
    }
}
