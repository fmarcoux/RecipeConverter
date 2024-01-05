using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecipeConverter.src.Classe
{
    public static class DirectoryUtils
    {
        public static string GetProjectRootDiretory()
        {
            string projectDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.FullName;
            
            if (projectDir == null)
            {
                throw new DirectoryNotFoundException("Project directory not found");
            }

            return projectDir;
        }
    }
}
