using RecipeConverter.src.DTO.ChatGptRequest;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecipeConverter.src.Interface
{
    public interface IChatGptClient
    {
        public Task<string> QueryChaGPT(ChatGPTRequestDTO text);
    }
}
