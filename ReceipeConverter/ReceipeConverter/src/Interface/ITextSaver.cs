using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReceipeConverter.src.Interface
{
    internal interface ITextSaver
    {
        /// <summary>
        /// Saves the strings in a single entity
        /// </summary>
        /// <param name="text">List of string to save</param>
        /// <returns></returns>
        public bool Save(IEnumerable<string> text);


        /// <summary>
        /// Saves the string in a single entity
        /// </summary>
        /// <param name="wholeText"></param>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public bool Save(string wholeText,string fileName);

        /// <summary>
        /// Saves the data in multiple entities
        /// </summary>
        /// <param name="multipleText"> List containing Lists of data to be saved</param>
        /// <returns>the number of entities that have not been saved successfully</returns>
        public int SaveMultiple(IReadOnlyList<IReadOnlyList<string>> multipleText);
        //TODO : Améliorer le error handling pour cette fonction, il serait probablement intéressant de savoir la/lesquelles on fail
    }
}
