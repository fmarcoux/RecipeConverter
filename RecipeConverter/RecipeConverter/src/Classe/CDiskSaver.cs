using RecipeConverter.src.Interface;
using System.Text;


namespace RecipeConverter.src.Classe
{
    internal class CDiskSaver : ITextSaver
    {
        private readonly DirectoryInfo m_destinationRepository ;

        public CDiskSaver(string repositoryPath)
        {
            if (!IsValidRepository(repositoryPath))
            {
                m_destinationRepository = Directory.CreateDirectory(repositoryPath);
            }
            else
            {
                //Le directory existe déjà
                m_destinationRepository = new DirectoryInfo(repositoryPath);
            }
             
        }

        public bool Save(IEnumerable<string> text)
        {
            if (!text.Any<string>()) return false;
            return Save(String.Join(" ", text), text.First());
        }

        public bool Save(string wholeText, string fileName)
        {
            if (String.IsNullOrEmpty(wholeText)) return false;
            FileStream f = new FileStream($"{m_destinationRepository.FullName}{Path.PathSeparator}{fileName}", FileMode.Create);
            f.Write(Encoding.UTF8.GetBytes(wholeText));
            return true;
        }

        public int SaveMultiple(IReadOnlyList<IReadOnlyList<string>> multipleText)
        {
            int failCounter = 0;
            foreach (var text in multipleText)
            {
                try
                {
                    Save(text);
                }
                catch { failCounter++; }
            }
            return failCounter;
        }

        /// <summary>
        /// Validate if a repository exists
        /// </summary>
        /// <param name="repositoryPath">the repository to validate</param>
        /// <returns></returns>
        private bool IsValidRepository(string repositoryPath)
        {
            return Directory.Exists(repositoryPath); 
        }
    }
}
