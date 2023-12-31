using System.Text.Json.Serialization;

namespace ReceipeConverter.src.DTO.TandoorRecipe
{

    public class ChildInheritField
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("field")]
        public string Field { get; set; }
    }

    public class File
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("file")]
        public string sourceFile { get; set; }

        [JsonPropertyName("id")]
        public int Id { get; set; }
    }

    public class Food
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("plural_name")]
        public string? PluralName { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("recipe")]
        public LinkedRecipe? usedInRecipe { get; set; }

        [JsonPropertyName("food_onhand")]
        public string? FoodOnhand { get; set; }

        [JsonPropertyName("supermarket_category")]
        public SupermarketCategory? SupermarketCategory { get; set; }

        [JsonPropertyName("inherit_fields")]
        public List<InheritField>? InheritFields { get; set; }

        [JsonPropertyName("ignore_shopping")]
        public bool IgnoreShopping { get; set; } = false;

        [JsonPropertyName("substitute")]
        public List<Substitute>? Substitute { get; set; }

        [JsonPropertyName("substitute_siblings")]
        public bool? SubstituteSiblings { get; set; }

        [JsonPropertyName("substitute_children")]
        public bool? SubstituteChildren { get; set; }

        [JsonPropertyName("child_inherit_fields")]
        public List<ChildInheritField>? ChildInheritFields { get; set; }
    }

    public class Ingredient
    {
        [JsonPropertyName("food")]
        public Food Food { get; set; }

        [JsonPropertyName("unit")]
        public Unit Unit { get; set; }

        [JsonPropertyName("amount")]
        public string Amount { get; set; }

        [JsonPropertyName("note")]
        public string? Note { get; set; }

        [JsonPropertyName("order")]
        public int? Order { get; set; }

        [JsonPropertyName("is_header")]
        public bool? IsHeader { get; set; }

        [JsonPropertyName("no_amount")]
        public bool? NoAmount { get; set; }

        [JsonPropertyName("original_text")]
        public string? OriginalText { get; set; }

        [JsonPropertyName("always_use_plural_unit")]
        public bool? AlwaysUsePluralUnit { get; set; }

        [JsonPropertyName("always_use_plural_food")]
        public bool? AlwaysUsePluralFood { get; set; }
    }

    public class InheritField
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("field")]
        public string Field { get; set; }
    }

    public class Keyword
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("icon")]
        public string Icon { get; set; }

        [JsonPropertyName("description")]
        public string Description { get; set; }
    }

    public class Nutrition
    {
        [JsonPropertyName("carbohydrates")]
        public string Carbohydrates { get; set; }

        [JsonPropertyName("fats")]
        public string Fats { get; set; }

        [JsonPropertyName("proteins")]
        public string Proteins { get; set; }

        [JsonPropertyName("calories")]
        public string Calories { get; set; }

        [JsonPropertyName("source")]
        public string Source { get; set; }
    }

    public class LinkedRecipe
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }
    }

    public class Recipe
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("description")]
        public string Description { get; set; }

        [JsonPropertyName("keywords")]
        public List<Keyword> Keywords { get; set; }

        [JsonPropertyName("steps")]
        public List<Step> Steps { get; set; }

        [JsonPropertyName("working_time")]
        public int WorkingTime { get; set; }

        [JsonPropertyName("waiting_time")]
        public int WaitingTime { get; set; }

        [JsonPropertyName("source_url")]
        public string? SourceUrl { get; set; }

        [JsonPropertyName("internal")]
        public bool? Internal { get; set; }

        [JsonPropertyName("show_ingredient_overview")]
        public bool ShowIngredientOverview { get; set; } = true;

        [JsonPropertyName("nutrition")]
        public Nutrition? Nutrition { get; set; }

        [JsonPropertyName("servings")]
        public int Servings { get; set; }

        [JsonPropertyName("file_path")]
        public string? FilePath { get; set; }

        [JsonPropertyName("servings_text")]
        public string? ServingsText { get; set; }

        [JsonPropertyName("private")]
        public bool Private { get; set; } = false;

        [JsonPropertyName("shared")]
        public List<Shared>? Shared { get; set; }
    }

    public class Shared
    {
        [JsonPropertyName("first_name")]
        public string FirstName { get; set; }

        [JsonPropertyName("last_name")]
        public string LastName { get; set; }
    }

    public class Step
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("instruction")]
        public string Instruction { get; set; }

        [JsonPropertyName("ingredients")]
        public List<Ingredient> Ingredients { get; set; }

        [JsonPropertyName("time")]
        public int? Time { get; set; }

        [JsonPropertyName("order")]
        public int? Order { get; set; }

        [JsonPropertyName("show_as_header")]
        public bool? ShowAsHeader { get; set; }

        [JsonPropertyName("file")]
        public File? File { get; set; }

        [JsonPropertyName("step_recipe")]
        public int? StepRecipe { get; set; }

    }

    public class Substitute
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("plural_name")]
        public string PluralName { get; set; }
    }

    public class SupermarketCategory
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("description")]
        public string Description { get; set; }
    }

    public class Unit
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("plural_name")]
        public string? PluralName { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }
    }

}
