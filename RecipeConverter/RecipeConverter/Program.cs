using Microsoft.Extensions.Configuration;

using RecipeConverter.src.Classe;

namespace VisionApiDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            IConfiguration configuration = new ConfigurationBuilder()
                .SetBasePath(DirectoryUtils.GetProjectRootDiretory())
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                .Build(); 

        }

    }
}
